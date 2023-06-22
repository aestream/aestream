#pragma once

#include <optional>

#include "../aer.hpp"
#include "../generator.hpp"

#include "utils.hpp"

struct EVT3 : FileBase
{

    struct RawEvent
    {
        unsigned int type : 4;
        unsigned int content : 12;
    } __attribute__((packed));

    struct EventCoordinate
    {
        unsigned int type : 4;
        bool meta : 1; // Either system type (EVT_ADDR_Y) or polarity (EVT_ADDR_X)
        unsigned int coordinate : 11;
    } __attribute__((packed));

    struct VECT8
    {
        unsigned int type : 4;
        unsigned int unused : 4;
        unsigned int valid : 8;
    } __attribute__((packed));

    struct VECT12
    {
        unsigned int type : 4;
        bool meta : 1; // Either system type (EVT_ADDR_Y) or polarity (EVT_ADDR_X)
        unsigned int valid : 11;
    } __attribute__((packed));

    struct EventTime
    {
        unsigned int type : 4;
        unsigned int time : 12;
        static uint64_t decode_time_high(const uint16_t *event, const uint64_t time)
        {
            const EventTime *time_high = reinterpret_cast<const EventTime *>(event);
            return (time & ~(0b111111111111ull << 12)) | (time_high->time << 12);
        }
    } __attribute__((packed));

    // https://docs.prophesee.ai/stable/data/encoding_formats/evt3.html
    enum EventType
    {
        EVT_ADDR_Y = 0b0000,
        EVT_ADDR_X = 0b0010,
        VEC_BASE_X = 0b0011,
        VECT_12 = 0b0100,
        VECT_8 = 0b0101,
        EVT_TIME_LOW = 0b0110,
        CONTINUED_4 = 0b0111,
        EVT_TIME_HIGH = 0x1000,
        EXT_TRIDDER = 0b1010,
        OTHERS = 0b1110,
        CONTINUED_12 = 0b1111
    };

    std::tuple<std::vector<AER::Event>, size_t> read_events(const int64_t n_events = -1)
    {
        static const size_t READ_BUFFER_SIZE = 4096;
        const size_t buffer_size = n_events > 0 ? n_events : READ_BUFFER_SIZE;
        const size_t event_array_size =
            n_events > 0 ? n_events : 512;

        std::vector<uint16_t> buffer_vector = std::vector<uint16_t>();
        buffer_vector.reserve(buffer_size);
        uint16_t *buffer = buffer_vector.data();

        std::vector<AER::Event> events{};
        events.reserve(event_array_size);

        size_t size = 0, index = 0;
        do
        {
            size = fread(buffer, sizeof(uint16_t), buffer_size, fp.get());
            if (size == 0 && !feof(fp.get()))
            {
                throw std::runtime_error("Error when processing .evt3 file");
            }

            if (size > n_events - index)
            {
                fseek(fp.get(), size - (n_events - index), SEEK_CUR); // Re-align file
                size = n_events - index;
            }

            auto offset = decode_event_buffer(buffer, size, events, n_events - events.size());
            fseek(fp.get(), offset, SEEK_CUR);

            index = index + size;
        } while (size > 0 && n_events - events.size() > 0);

        return {events, events.size()};
    }

    Generator<AER::Event> stream(const int64_t n_events = -1)
    {
        static const size_t STREAM_BUFFER_SIZE = 4096;
        uint64_t buffer[STREAM_BUFFER_SIZE];
        uint64_t timestep = 0;
        size_t overflows = 0, size = 0, count = 0;
        do
        {
            auto ret = read_events(STREAM_BUFFER_SIZE);
            auto events = std::get<0>(ret);
            size = std::get<1>(ret);
            for (size_t i = 0; i < size; i++)
            {
                co_yield events[i];
                count++;
            }
        } while (size == STREAM_BUFFER_SIZE &&
                 (n_events < 0 || (n_events - count >= 0)));
    }

    explicit EVT3(const std::string &filename) : EVT3(open_file(filename)) {}
    explicit EVT3(file_t &&fp) : fp(std::move(fp)) {}

private:
    const file_t fp;
    uint32_t time_low = 0, time_high = 0, time_overflow = 0; //, base_x = 0; // State
    uint64_t current_time = 0;
    std::optional<EventCoordinate *> base_x_event = {};
    std::optional<EventCoordinate *> y_event = {};

    inline void process_vector_event(uint16_t bits, uint16_t size, std::vector<AER::Event> &events)
    {
        if (base_x_event.has_value())
        {
            uint16_t y = (y_event.has_value()) ? y_event.value()->coordinate : 0;
            for (int i = 0; i < size; i++)
            {
                const uint16_t x = static_cast<uint16_t>(base_x_event.value()->coordinate++); // Increment coordinate irregardless of validity
                if (bits & i)
                {
                    events.push_back({current_time, x, y, base_x_event.value()->meta});
                }
            }
        }
    }

    int32_t decode_event_buffer(uint16_t *buffer, size_t buffer_size, std::vector<AER::Event> &events, int64_t remaining_events)
    {
        const size_t max_remaining_limit = events.size() + remaining_events; // Maximum events to emit

        for (size_t i = 0; i < buffer_size; ++i)
        {
            auto raw_event = reinterpret_cast<RawEvent *>(&buffer[i]);
            if (events.size() >= max_remaining_limit) // Test limit has been reached
            {
                return i - buffer_size; // Return remaining offset
            }

            if (raw_event->type == EVT3::EventType::EVT_ADDR_Y)
            {
                y_event = reinterpret_cast<EventCoordinate *>(raw_event);
            }
            else if (raw_event->type == EVT3::EventType::EVT_ADDR_X)
            {
                const auto x_event = reinterpret_cast<EventCoordinate *>(raw_event);
                const AER::Event event = {current_time, static_cast<uint16_t>(x_event->coordinate), static_cast<uint16_t>(y_event.value()->coordinate), x_event->meta};
                events.push_back(event);
            }
            else if (raw_event->type == EventType::EVT_TIME_HIGH)
            {
                static constexpr uint64_t TIMESTAMP_MAX = 1ULL << 11;
                const auto event_time_high = reinterpret_cast<const EventTime *>(raw_event);
                const auto time_high = event_time_high->time << 12;

                if (event_time_high->time > TIMESTAMP_MAX + current_time)
                {
                    time_overflow++;
                }
                if (current_time & time_high != time_high)
                {
                    time_low = 0;
                }
                current_time = (time_overflow << 22) & (time_high << 11) & (time_low);
            }
            else if (raw_event->type == EventType::EVT_TIME_LOW)
            {
                const auto event_time_low = reinterpret_cast<const EventTime *>(raw_event);
                time_low += event_time_low->time;
                current_time = (time_overflow << 22) | (time_high << 11) | (time_low);
            }
            else if (raw_event->type == EventType::VEC_BASE_X)
            {
                base_x_event = reinterpret_cast<EventCoordinate *>(raw_event);
            }
            else if (raw_event->type == EventType::VECT_8)
            {
                const auto vect_event = reinterpret_cast<const VECT8 *>(raw_event);
                process_vector_event(vect_event->valid, 8, events);
            }
            else if (raw_event->type == EventType::VECT_12)
            {
                const auto vect_event = reinterpret_cast<const VECT12 *>(raw_event);
                process_vector_event(vect_event->valid, 12, events);
            }
            //  for (; cur_raw_ev != raw_ev_end;)
            //  {
            //      const uint16_t type = cur_raw_ev->type;
            //      if (type == static_cast<EventTypesUnderlying_t>(EventTypesEnum::EVT_ADDR_X))
            //      {
            //          if (is_valid)
            //          {
            //              const Evt3Raw::Event_PosX *ev_posx = reinterpret_cast<const Evt3Raw::Event_PosX *>(cur_raw_ev);
            //              if (validator.validate_event_cd(cur_raw_ev))
            //              {
            //                  cd_forwarder.forward(static_cast<unsigned short>(ev_posx->x),
            //                                       state[(int)EventTypesEnum::EVT_ADDR_Y], static_cast<short>(ev_posx->pol),
            //                                       last_timestamp<DO_TIMESHIFT>());
            //              }
            //          }
            //          ++cur_raw_ev;
            //      }
            //      else if (type == static_cast<EventTypesUnderlying_t>(EventTypesEnum::VECT_12))
            //      {
            //          constexpr uint32_t vect12_size = sizeof(Evt3Raw::Event_Vect12_12_8) / sizeof(RawEvent);
            //          if (cur_raw_ev + vect12_size > raw_ev_end)
            //          {
            //              // Not enough raw data to decode the vect12_12_8 events. Stop decoding this buffer and return the
            //              // amount of data missing to wait for to be able to decode on the next call
            //              return std::distance(raw_ev_end, cur_raw_ev + vect12_size);
            //          }
            //          if (!is_valid)
            //          {
            //              cur_raw_ev += vect12_size;
            //              continue;
            //          }

            //         const uint16_t nb_bits = 32;
            //         int next_offset;
            //         if (validator.validate_vect_12_12_8_pattern(
            //                 cur_raw_ev, state[(int)EventTypesEnum::VECT_BASE_X] & NOT_POLARITY_MASK, next_offset))
            //         {
            //             cd_forwarder.reserve(32);

            //             const Evt3Raw::Event_Vect12_12_8 *ev_vect12_12_8 =
            //                 reinterpret_cast<const Evt3Raw::Event_Vect12_12_8 *>(cur_raw_ev);

            //             Evt3Raw::Mask m;
            //             m.m.valid1 = ev_vect12_12_8->valid1;
            //             m.m.valid2 = ev_vect12_12_8->valid2;
            //             m.m.valid3 = ev_vect12_12_8->valid3;

            //             uint32_t valid = m.valid;

            //             uint16_t last_x = state[(int)EventTypesEnum::VECT_BASE_X] & NOT_POLARITY_MASK;
            //             uint16_t off = 0;
            //             while (valid)
            //             {
            //                 off = ctz_not_zero(valid);
            //                 valid &= ~(1 << off);
            //                 cd_forwarder.forward_unsafe(last_x + off, state[(int)EventTypesEnum::EVT_ADDR_Y],
            //                                             (bool)(state[(int)EventTypesEnum::VECT_BASE_X] & POLARITY_MASK),
            //                                             last_timestamp<DO_TIMESHIFT>());
            //             }
            //         }
            //         if (validator.has_valid_vect_base())
            //         {
            //             state[(int)EventTypesEnum::VECT_BASE_X] += nb_bits;
            //         }
            //         cur_raw_ev += next_offset;
            //     }
            //     else if (type == static_cast<EventTypesUnderlying_t>(EventTypesEnum::EVT_TIME_HIGH))
            //     {
            //         const Evt3Raw::Event_Time *ev_timehigh = reinterpret_cast<const Evt3Raw::Event_Time *>(cur_raw_ev);
            //         static constexpr timestamp max_timestamp_ = 1ULL << 11;

            //         validator.validate_time_high(last_timestamp_.bitfield_time.high, ev_timehigh->time);

            //         last_timestamp_.bitfield_time.loop +=
            //             (bool)(last_timestamp_.bitfield_time.high >= max_timestamp_ + ev_timehigh->time);
            //         last_timestamp_.bitfield_time.low =
            //             (last_timestamp_.bitfield_time.high == ev_timehigh->time ? last_timestamp_.bitfield_time.low : 0); // avoid momentary time discrepancies when decoding event per events. Time low comes
            //                                                                                                                // right after to correct the value (note that the timestamp here is not good if we don't
            //                                                                                                                // do that either)
            //         last_timestamp_.bitfield_time.high = ev_timehigh->time;

            //         ++cur_raw_ev;
            //     }
            //     else if (type == static_cast<EventTypesUnderlying_t>(EventTypesEnum::EXT_TRIGGER))
            //     {
            //         if (validator.validate_ext_trigger(cur_raw_ev))
            //         {
            //             const Evt3Raw::Event_ExtTrigger *ev_exttrigger =
            //                 reinterpret_cast<const Evt3Raw::Event_ExtTrigger *>(cur_raw_ev);
            //             trigger_forwarder.forward(static_cast<short>(ev_exttrigger->pol), last_timestamp<DO_TIMESHIFT>(),
            //                                       static_cast<short>(ev_exttrigger->id));
            //         }
            //         ++cur_raw_ev;
            //     }
            //     else if (type == static_cast<EventTypesUnderlying_t>(EventTypesEnum::OTHERS))
            //     {
            //         const uint16_t master_type = cur_raw_ev->content;
            //         bool is_out_count_evt = false;

            //         switch (master_type)
            //         {
            //         case static_cast<uint16_t>(Evt3MasterEventTypes::MASTER_RATE_CONTROL_CD_EVENT_COUNT):
            //             is_out_count_evt = true;
            //             [[fallthrough]];
            //         case static_cast<uint16_t>(Evt3MasterEventTypes::MASTER_IN_CD_EVENT_COUNT):
            //         {
            //             constexpr uint32_t evt_size = 1 + sizeof(Evt3Raw::Event_Continue12_12_4) / sizeof(RawEvent);
            //             if (cur_raw_ev + evt_size > raw_ev_end)
            //             {
            //                 // Not enough raw data to decode the continue events. Stop decoding this buffer and return the
            //                 // amount of data missing to wait for to be able to decode on the next call
            //                 return std::distance(raw_ev_end, cur_raw_ev + evt_size);
            //             }
            //             ++cur_raw_ev;
            //             int next_offset;
            //             if (validator.validate_continue_12_12_4_pattern(cur_raw_ev, next_offset))
            //             {
            //                 const Evt3Raw::Event_Continue12_12_4 *data =
            //                     reinterpret_cast<const Evt3Raw::Event_Continue12_12_4 *>(cur_raw_ev);
            //                 erc_count_forwarder.forward(last_timestamp<DO_TIMESHIFT>(),
            //                                             Evt3Raw::Event_Continue12_12_4::decode(*data), is_out_count_evt);
            //             }
            //             cur_raw_ev += next_offset;
            //             break;
            //         }
            //         default:
            //             // Unhandled sys event
            //             ++cur_raw_ev;
            //             break;
            //         }
            //     }
            //     else
            //     {
            //         // The objective is to reduce the number of possible cases
            //         // The content of each type is store into a state because the encoding is stateful
            //         state[type] = cur_raw_ev->content;
            //         // Here the type of event is saved (CD vs EM) to know when a EVT_ADDR_X or VECT_BASE_X arrives
            //         // if the event is a CD or EM
            //         is_cd = type >= 2 ? is_cd : !(bool)type;
            //         // Some event outside of the sensor may occur, to limit the number of test the check is done
            //         // every EVT_ADDR_Y
            //         is_valid = is_cd && state[(int)EventTypesEnum::EVT_ADDR_Y] < height_;

            //         last_timestamp_.bitfield_time.low =
            //             type != static_cast<EventTypesUnderlying_t>(EventTypesEnum::EVT_TIME_LOW) ? last_timestamp_.bitfield_time.low : state[static_cast<EventTypesUnderlying_t>(EventTypesEnum::EVT_TIME_LOW)];
            //         last_timestamp_set_ = true;

            //         validator.state_update(cur_raw_ev);

            //         ++cur_raw_ev;
            //     }
        }
        return 0;
    }
};