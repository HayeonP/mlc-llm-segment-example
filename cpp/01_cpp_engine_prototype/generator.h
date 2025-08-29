#pragma once
#include <coroutine>
#include <exception>
#include <iostream>
#include <chrono>
#include <thread>

template<typename T>
class Generator{
public:
    class promise_type;
    using handle_type = std::coroutine_handle<promise_type>;

    class promise_type{
    public:
        T current_value;
        std::exception_ptr exception;

        Generator get_return_object(){
            return Generator{ handle_type::from_promise(*this) };
        }

        std::suspend_always initial_suspend() { return{}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        std::suspend_always yield_value(T value){
            current_value = value;
            return {};
        }
        void return_void() {}
        void unhandled_exception() {
            exception = std::current_exception();
        }

    };

    handle_type coroutine;

    explicit Generator(handle_type h) : coroutine(h) {};
    ~Generator() { if (coroutine) coroutine.destroy(); }

    Generator(const Generator&) = delete;
    Generator& operator=(const Generator&) = delete;
    Generator(Generator&& other) noexcept : coroutine(other.coroutine) { other.coroutine = nullptr; }
    Generator& operator=(Generator&& other) noexcept {
        if (this != &other){
            if (coroutine) coroutine.destroy();
            coroutine = other.coroutine;
            other.coroutine = nullptr;
        }
        return *this;
    }

    bool move_next(){
        coroutine.resume();
        if (coroutine.done()){
            if (coroutine.promise().exception)
                std::rethrow_exception(coroutine.promise().exception);
            return false;
        }
        return true;
    }

    T current_value(){
        return coroutine.promise().current_value;
    }
};


Generator<int> simple_generator(int limit){
    int count = 0;
    while (count < limit){
        co_yield count++;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

int generator_example() {
    auto gen = simple_generator(5);
    while (gen.move_next()){
        std::cout << gen.current_value() << std::endl;
    }

    return 0;
}

