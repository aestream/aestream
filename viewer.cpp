#include "aedat.hpp"

#include <SDL.h>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>

int main(int argc, char *argv[]) {
  const int WINDOW_WIDTH = 128;
  AEDAT data;

  if (argc > 0) {
    data.load(argv[1]);
  } else {
    return 0;
  }

  SDL_Event event;
  SDL_Renderer *renderer;
  SDL_Window *window;

  SDL_Init(SDL_INIT_VIDEO);
  SDL_CreateWindowAndRenderer(WINDOW_WIDTH, WINDOW_WIDTH, 0, &window,
                              &renderer);

  uint32_t timestep = data.polarity_events[0].timestamp + 16000;
  uint32_t event_index = 0;
  uint32_t frame_idx = 0;

  uint32_t ticks = SDL_GetTicks();
  while (1) {
    uint32_t next_ticks = SDL_GetTicks();
    if (next_ticks - ticks < 16) {
      continue;
    }
    ticks = next_ticks;

    if (SDL_PollEvent(&event) && event.type == SDL_QUIT)
      break;

    std::vector<SDL_Point> positive_polarity_points;
    std::vector<SDL_Point> negative_polarity_points;

    while (data.polarity_events[event_index].timestamp < timestep) {
      if (data.polarity_events[event_index].polarity == 1) {
        positive_polarity_points.push_back(
            {data.polarity_events[event_index].y,
             data.polarity_events[event_index].x});
      } else {
        negative_polarity_points.push_back(
            {data.polarity_events[event_index].y,
             data.polarity_events[event_index].x});
      }
      event_index++;
    }

    SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255);
    SDL_RenderDrawPoints(renderer, &positive_polarity_points[0],
                         positive_polarity_points.size());
    SDL_SetRenderDrawColor(renderer, 0, 0, 255, 255);
    SDL_RenderDrawPoints(renderer, &negative_polarity_points[0],
                         negative_polarity_points.size());
    SDL_RenderPresent(renderer);
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 0);
    SDL_RenderClear(renderer);

    negative_polarity_points.clear();
    positive_polarity_points.clear();
    timestep += 16000;
    frame_idx++;
  }

  SDL_DestroyRenderer(renderer);
  SDL_DestroyWindow(window);
  SDL_Quit();
  return EXIT_SUCCESS;
}