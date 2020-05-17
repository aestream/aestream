#include "aedat.hpp"
#include "convert.hpp"

int main(int argc, char *argv[]) {
  AEDAT data;

  if (argc > 0) {
    data.load(argv[1]);
  } else {
    return 0;
  }

  auto events = convert_polarity_events(data.polarity_events);
  std::cout << events.sizes() << std::endl;
}