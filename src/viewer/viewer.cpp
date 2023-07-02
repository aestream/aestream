#include <iomanip>
#include <iostream>
#include <locale>
#include <vector>

#include <SDL2/SDL.h>

#include "viewer.hpp"

void render_polarity_events(SDL_Renderer *renderer,
                            std::vector<SDL_Point> &positive_events,
                            std::vector<SDL_Point> &negative_events) {
  SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255);
  SDL_RenderDrawPoints(renderer, &positive_events[0], positive_events.size());
  SDL_SetRenderDrawColor(renderer, 0, 0, 255, 255);
  SDL_RenderDrawPoints(renderer, &negative_events[0], negative_events.size());
}

struct NumberGrouping : std::numpunct<char> {
  std::string do_grouping() const { return "\3"; }
};

int view_stream(Generator<AER::Event> &generator, size_t width, size_t height,
                size_t frame_duration, bool quiet) {
  SDL_Event sdl_event;
  SDL_Renderer *renderer;
  SDL_Window *window;

  SDL_Init(SDL_INIT_VIDEO);

  SDL_CreateWindowAndRenderer(width, height, 0, &window, &renderer);

  auto frame_texture =
      SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGB888,
                        SDL_TEXTUREACCESS_STATIC, width, height);
  if (!frame_texture) {
    std::cout << SDL_GetError() << std::endl;
  }

  size_t frame_idx = 0;
  size_t event_count = 0;
  size_t event_count_positive = 0;
  size_t event_count_positive_last_fps = 0;
  size_t event_count_negative = 0;
  size_t event_count_negative_last_fps = 0;
  size_t event_count_last_fps = 0;
  // Ticks are value in ms since initialization
  size_t ticks_frame = SDL_GetTicks();
  size_t ticks_text = SDL_GetTicks();

  std::vector<SDL_Point> positive_buffer;
  std::vector<SDL_Point> negative_buffer;

  for (auto event : generator) {
    if (event.polarity) {
      positive_buffer.push_back({event.x, event.y});
      event_count_positive++;
    } else {
      negative_buffer.push_back({event.x, event.y});
      event_count_negative++;
    }
    event_count++;

    uint32_t next_ticks = SDL_GetTicks();
    if (next_ticks - ticks_frame <= frame_duration) {
      continue;
    }
    ticks_frame = next_ticks;

    if (!quiet && next_ticks - ticks_text >= 1000) {
      auto eps = event_count - event_count_last_fps;
      auto eps_pos = event_count_positive - event_count_positive_last_fps;
      auto eps_neg = event_count_negative - event_count_negative_last_fps;
      event_count_last_fps = event_count;
      event_count_positive = event_count_positive_last_fps;
      event_count_negative = event_count_negative_last_fps;
      std::cout.imbue(std::locale(std::cout.getloc(), new NumberGrouping));
      std::cout << "Events per second: " << std::setw(15) << eps << "  (+"
                << std::setw(15) << eps_pos << "/ -" << std::setw(15) << eps_neg
                << ")" << std::endl;
      ticks_text = next_ticks;
    }

    if (SDL_PollEvent(&sdl_event) && sdl_event.type == SDL_QUIT)
      break;

    render_polarity_events(renderer, positive_buffer, negative_buffer);
    SDL_RenderPresent(renderer);
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 0);
    SDL_RenderClear(renderer);
    positive_buffer.clear();
    negative_buffer.clear();
    frame_idx++;
  }

  SDL_DestroyRenderer(renderer);
  SDL_DestroyWindow(window);
  SDL_Quit();
  return EXIT_SUCCESS;
}
