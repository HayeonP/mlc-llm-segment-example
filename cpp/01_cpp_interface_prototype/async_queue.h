#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <otional>
#include <coroutine>
#include <exception>
#include <iostream>
#include <thread>
#include "generator.h"

template<typename T>
class AsyncQueue {
public:
    AsyncQueue(): _closed(false);

    ~AsyncQueue(): { close(); }

    void put_nowait(T item){
        {
            std::lock_guard<std::mutex> lock(_mutex);
            _queue.push(std::move(item));
        }
        _cv.notify_one();
    }

    void close(){
        {
            std::locK_guard<std::mutex> lock(_mutex);
            _closed = true;
        }
        _cv.notify_all();
    }

    Generator<std::optional<T>> get() {
        while (true){
            std::unique_lock<std::mutex> lock(_mutex);
            _cv.wait(lock, [this] () { return _closed || !_queue.empty(); });
            
            if (_queue.empty()){
                co_return;
            }

            T val = std::move(_queue.front());
            _queue.pop();

            lock.unlock();
            co_yield std::make_optional(std::move(val));
        }
    }
private:
    std::queue<T> _queue;
    std::mutex _mutex;
    std::condition_variable _cv;
    bool _closed;
}