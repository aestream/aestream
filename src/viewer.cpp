#include "aedat.hpp"
#include "dvs_gesture.hpp"

#include <SDL.h>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>

uint32_t render_frame(SDL_Renderer *renderer, SDL_Texture *frame_texture,
                      AEDAT4 &data, uint32_t frame_index, uint64_t timestep) {
  if (frame_index >= data.frames.size()) {
    return 0;
  }

  while ((data.frames[frame_index].time < timestep) &&
         (frame_index < data.frames.size() - 1)) {
    frame_index++;
  }

  SDL_UpdateTexture(frame_texture, nullptr, &data.frames[frame_index].pixels[0],
                    3 * data.frames[frame_index].width);
  SDL_RenderCopy(renderer, frame_texture, nullptr, nullptr);

  return frame_index;
}

uint32_t
render_polarity_events(SDL_Renderer *renderer,
                       std::vector<AEDAT::PolarityEvent> &polarity_events,
                       SDL_Point top, uint32_t event_index, uint32_t timestep) {
  std::vector<SDL_Point> positive_polarity_points;
  std::vector<SDL_Point> negative_polarity_points;

  if (event_index >= polarity_events.size()) {
    return 0;
  }

  while ((polarity_events[event_index].timestamp < timestep) &&
         (event_index < polarity_events.size())) {
    if (polarity_events[event_index].polarity == 1) {
      positive_polarity_points.push_back(
          {top.x + polarity_events[event_index].x,
           top.y + polarity_events[event_index].y});
    } else {
      negative_polarity_points.push_back(
          {top.x + polarity_events[event_index].x,
           top.y + polarity_events[event_index].y});
    }
    event_index++;
  }

  SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255);
  SDL_RenderDrawPoints(renderer, &positive_polarity_points[0],
                       positive_polarity_points.size());
  SDL_SetRenderDrawColor(renderer, 0, 0, 255, 255);
  SDL_RenderDrawPoints(renderer, &negative_polarity_points[0],
                       negative_polarity_points.size());

  return event_index;
}

int main(int argc, char *argv[]) {
  int window_width;
  int window_height;
  int num_row;
  int num_column;
  int num_classes;

  AEDAT data;
  AEDAT4 data4;
  dvs_gesture::DataSet dataset;
  bool gesture_dataset = false;
  std::vector<std::vector<AEDAT::PolarityEvent>> events;
  std::vector<uint32_t> event_index;
  std::vector<uint32_t> timestep;
  int64_t video_timestep;

  if (argc == 2) {
    data4.load(argv[1]);
    window_width = data4.outinfos[0].size_x;
    window_height = data4.outinfos[0].size_y;
    events.push_back(data4.polarity_events);
    event_index.push_back(0);
    num_row = 1;
    num_column = 1;
    num_classes = 1;
    timestep.push_back(data4.polarity_events[0].timestamp + 16000);
  } else if (argc == 3) {
    dataset.load(argv[1], argv[2]);
    num_row = 3;
    num_column = 4;
    window_width = num_column * 128;
    window_height = num_row * 128;
    num_classes = 11;

    for (auto data : dataset.datapoints) {
      events.push_back(data.events);
      event_index.push_back(0);

      timestep.push_back(data.events[0].timestamp + 16000);
    }
  } else {
    return 0;
  }

  SDL_Event event;
  SDL_Renderer *renderer;
  SDL_Window *window;

  SDL_Init(SDL_INIT_VIDEO);

  SDL_CreateWindowAndRenderer(window_width, window_height, 0, &window,
                              &renderer);

  auto frame_texture =
      SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGB888,
                        SDL_TEXTUREACCESS_STATIC, window_width, window_height);
  if (!frame_texture) {
    std::cout << SDL_GetError() << std::endl;
  }

  bool has_video = data4.frames.size() > 0;

  if (has_video) {
    SDL_UpdateTexture(frame_texture, nullptr, &data4.frames[0].pixels[0],
                      3 * window_width);
    video_timestep = data4.frames[0].time;
  }

  uint32_t frame_idx = 0;
  uint32_t video_frame_index = 0;
  uint32_t ticks = SDL_GetTicks();
  while (1) {
    uint32_t next_ticks = SDL_GetTicks();
    if (next_ticks - ticks < 16) {
      continue;
    }
    ticks = next_ticks;

    if (SDL_PollEvent(&event) && event.type == SDL_QUIT)
      break;

    if (has_video) {
      if (video_frame_index >= data4.frames.size() - 1) {
        video_frame_index = 0;
        video_timestep = data4.frames[0].time;
      }

      video_frame_index = render_frame(renderer, frame_texture, data4,
                                       video_frame_index, video_timestep);
      video_timestep += 16000;
    }

    for (int i = 0; i < num_row; i++) {
      for (int j = 0; j < num_column; j++) {
        if (num_column * i + j >= num_classes) {
          break;
        }
        event_index[num_column * i + j] = render_polarity_events(
            renderer, events[num_column * i + j], {128 * i, 128 * j},
            event_index[num_column * i + j], timestep[num_column * i + j]);
        if (event_index[num_column * i + j] == 0) {
          timestep[num_column * i + j] =
              events[num_column * i + j][0].timestamp;
        }
        timestep[num_column * i + j] += 16000;
      }
    }
    SDL_RenderPresent(renderer);
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 0);
    SDL_RenderClear(renderer);

    frame_idx++;
  }

  SDL_DestroyRenderer(renderer);
  SDL_DestroyWindow(window);
  SDL_Quit();
  return EXIT_SUCCESS;
}
