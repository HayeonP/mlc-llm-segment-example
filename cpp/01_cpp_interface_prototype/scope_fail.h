#pragma once

#include <exception>
#include <functional>

class ScopeFail {
private:
    bool is_exception_activated;
    std::function<void()> func;
public:
    ScopeFail(std::function<void()> f)
        : is_exception_activated(std::uncaught_exceptions()), func(std::move(f)) {}
    ~ScopeFail() noexcept {
        if(std::uncaught_exceptions() != is_exception_activated) func();
    }
};