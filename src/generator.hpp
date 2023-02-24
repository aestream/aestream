#pragma once

#if USE_CLANG
#include <experimental/coroutine>
namespace coroutinestd = std::experimental;
#else
#include <coroutine>
namespace coroutinestd = std;
#endif
#include <iostream>
#include <optional>

template <std::movable T> class Generator {
  // Thanks to https://en.cppreference.com/w/cpp/coroutine/coroutine_handle
public:
  struct promise_type {
    std::optional<T> current_value;
    std::exception_ptr current_exception;

    Generator<T> get_return_object() {
      return Generator{Handle::from_promise(*this)};
    }
    void return_void() {}
    // use coroutinestd instead of {std, std::experimental}
    static coroutinestd::suspend_always initial_suspend() noexcept { return {}; }
    static coroutinestd::suspend_always final_suspend() noexcept { return {}; }
    coroutinestd::suspend_always yield_value(T value) noexcept {
      current_value = std::move(value);
      return {};
    }
    // Disallow co_await in generator coroutines.
    void await_transform() = delete;
    void unhandled_exception() noexcept { current_exception = std::current_exception(); }
  };

  // use coroutinestd instead of {std, std::experimental}
  using Handle = coroutinestd::coroutine_handle<promise_type>;

  explicit Generator(const Handle coroutine) : m_coroutine{coroutine} {}

  Generator() = default;
  ~Generator() {
    if (m_coroutine) {
      m_coroutine.destroy();
    }
  }

  Generator(const Generator &) = delete;
  Generator &operator=(const Generator &) = delete;

  Generator(Generator &&other) noexcept : m_coroutine{other.m_coroutine} {
    other.m_coroutine = {};
  }
  Generator &operator=(Generator &&other) noexcept {
    if (this != &other) {
      if (m_coroutine) {
        m_coroutine.destroy();
      }
      m_coroutine = other.m_coroutine;
      other.m_coroutine = {};
    }
    return *this;
  }

  // Range-based for loop support.
  class Iter {
  public:
    void operator++() { m_coroutine.resume(); }
    const T &operator*() const {
      promise_type promise = m_coroutine.promise();
      if (promise.current_exception) {
        std::rethrow_exception(promise.current_exception);
      } else {
        return *promise.current_value;
      }
    }
    bool operator==(std::default_sentinel_t) const {
      return !m_coroutine || m_coroutine.done();
    }

    explicit Iter(const Handle coroutine) : m_coroutine{coroutine} {}

  private:
    Handle m_coroutine;
  };

  Iter begin() {
    if (m_coroutine) {
      m_coroutine.resume();
    }
    return Iter{m_coroutine};
  }
  std::default_sentinel_t end() { return {}; }

  // private:
  Handle m_coroutine;
};
