#include <queue>
#include <mutex>
#include <condition_variable>
#include <stdexcept>

template <typename T>
class BlockingQueue {
public:
    explicit BlockingQueue(size_t maxsize = 0) : maxsize_(maxsize) {}

    void put_nowait(const T& value) {
        std::unique_lock<std::mutex> lock(mtx_);
        if (maxsize_ > 0 && queue_.size() >= maxsize_) {
            throw std::runtime_error("queue.Full");  // Python과 유사하게 예외
        }
        queue_.push(value);
        cv_.notify_one();
    }

    T get() {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this]{ return !queue_.empty(); });
        T value = queue_.front();
        queue_.pop();
        return value;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return queue_.empty();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return queue_.size();
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mtx_);
        std::queue<T> empty;
        std::swap(queue_, empty);
    }
private:
    std::queue<T> queue_;
    size_t maxsize_;  // 0 → 무제한
    mutable std::mutex mtx_;
    std::condition_variable cv_;
};
