#pragma once

#if !defined USE_CLANG
  #include <experimental/coroutine>
#else
  #include <coroutine>
#endif
#include <iostream>
#include <optional>

template <std::movable T> class Generator {
  // Thanks to https://en.cppreference.com/w/cpp/coroutine/coroutine_handle
public:
  struct promise_type {
    Generator<T> get_return_object() {
      return Generator{Handle::from_promise(*this)};
    }
    void return_void() {}
    // hack to resolve std / std::experimental
#if !defined USE_CLANG
    static std::experimental::suspend_always initial_suspend() noexcept { return {}; }
    static std::experimental::suspend_always final_suspend() noexcept { return {}; }
    std::experimental::suspend_always yield_value(T value) noexcept {
      current_value = std::move(value);
      return {};
    }
#else
    static std::suspend_always initial_suspend() noexcept { return {}; }
    static std::suspend_always final_suspend() noexcept { return {}; }
    std::suspend_always yield_value(T value) noexcept {
      current_value = std::move(value);
      return {};
    }
#endif
    // Disallow co_await in generator coroutines.
    void await_transform() = delete;
    [[noreturn]] static void unhandled_exception() { throw; }

    std::optional<T> current_value;
  };

  // hack to resolve std / std::experimental
#if !defined USE_CLANG
  using Handle = std::experimental::coroutine_handle<promise_type>;
#else
  using Handle = std::coroutine_handle<promise_type>;
#endif

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
    const T &operator*() const { return *m_coroutine.promise().current_value; }
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

private:
  Handle m_coroutine;
};
