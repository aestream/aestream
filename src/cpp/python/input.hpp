#include "aer.hpp"
#include "types.hpp"

struct PyInput {
  virtual tensor_t all();
  virtual tensor_t read();
}