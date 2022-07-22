/*******************************************************************************
 * @file:   logger.h
 *
 * @author: Daniel Seemaier
 * @date:   21.09.21
 * @brief:  Helper for console output.
 ******************************************************************************/
#pragma once

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <csignal>
#include <iostream>
#include <memory>
#include <sstream>
#include <type_traits>
#include <utility>
#include <vector>

#include <tbb/spin_mutex.h>

#define LOG  (kaminpar::Logger())
#define LLOG (kaminpar::Logger(std::cout, ""))

namespace kaminpar {
namespace logger {
template <typename T, typename = void>
struct is_iterable : std::false_type {};

template <typename T>
struct is_iterable<T, std::void_t<decltype(std::begin(std::declval<T>())), decltype(std::end(std::declval<T>()))>>
    : std::true_type {};

template <typename T>
constexpr bool is_container_v = !std::is_same_v<std::decay_t<T>, std::string> && //
                                !std::is_same_v<std::decay_t<T>, char*> &&       //
                                !std::is_same_v<std::decay_t<T>, const char*> && //
                                is_iterable<T>::value;

class ContainerFormatter {
public:
    virtual ~ContainerFormatter()                                                          = default;
    virtual void print(const std::vector<std::string>& container, std::ostream& out) const = 0;
};

class CompactContainerFormatter : public ContainerFormatter {
public:
    constexpr explicit CompactContainerFormatter(std::string_view sep) noexcept : _sep{sep} {}
    void print(const std::vector<std::string>& container, std::ostream& out) const final;

private:
    std::string_view _sep;
};

class Table : public ContainerFormatter {
public:
    constexpr explicit Table(std::size_t width) noexcept : _width{width} {}
    void print(const std::vector<std::string>& container, std::ostream& out) const final;

private:
    std::size_t _width;
};

class TextFormatter {
public:
    virtual ~TextFormatter()                                             = default;
    virtual void print(const std::string& text, std::ostream& out) const = 0;
};

class DefaultTextFormatter final : public TextFormatter {
public:
    void print(const std::string& text, std::ostream& out) const final;
};

class Colorized : public TextFormatter {
public:
    enum class Color { RED, GREEN, MAGENTA, ORANGE, CYAN, RESET };

    constexpr explicit Colorized(Color color) noexcept : _color{color} {}
    void print(const std::string& text, std::ostream& out) const final;

private:
    Color _color;
};

template <typename T>
constexpr bool is_text_formatter_v = std::is_base_of_v<TextFormatter, std::decay_t<T>>;

template <typename T>
constexpr bool is_container_formatter_v = std::is_base_of_v<ContainerFormatter, std::decay_t<T>>;

template <typename T>
constexpr bool is_default_log_arg_v = !is_container_v<T> && !is_text_formatter_v<T> && !is_container_formatter_v<T>;

extern DefaultTextFormatter      DEFAULT_TEXT;
extern Colorized                 RED;
extern Colorized                 GREEN;
extern Colorized                 MAGENTA;
extern Colorized                 ORANGE;
extern Colorized                 CYAN;
extern Colorized                 RESET;
extern CompactContainerFormatter DEFAULT_CONTAINER;
extern CompactContainerFormatter COMPACT;
extern Table                     TABLE;
} // namespace logger

class Logger {
public:
    Logger();
    explicit Logger(std::ostream& out, std::string append = "\n");

    Logger(const Logger&)            = delete;
    Logger& operator=(const Logger&) = delete;
    Logger(Logger&&) noexcept        = default;
    Logger& operator=(Logger&&)      = delete;
    virtual ~Logger() {
        flush();
    };

    template <typename Arg, std::enable_if_t<logger::is_default_log_arg_v<Arg>, bool> = true>
    Logger& operator<<(Arg&& arg) {
        std::stringstream ss;
        ss << arg;
        _text_formatter->print(ss.str(), _buffer);
        return *this;
    }

    template <typename Formatter, std::enable_if_t<logger::is_text_formatter_v<Formatter>, bool> = true>
    Logger& operator<<(Formatter&& formatter) {
        _text_formatter = std::make_unique<std::decay_t<Formatter>>(formatter);
        return *this;
    }

    template <typename Formatter, std::enable_if_t<logger::is_container_formatter_v<Formatter>, bool> = true>
    Logger& operator<<(Formatter&& formatter) {
        _container_formatter = std::make_unique<std::decay_t<Formatter>>(formatter);
        return *this;
    }

    template <typename T, std::enable_if_t<logger::is_container_v<T>, bool> = true>
    Logger& operator<<(T&& container) {
        std::vector<std::string> str;
        for (const auto& element: container) {
            std::stringstream ss;
            ss << element;
            str.push_back(ss.str());
        }
        _container_formatter->print(str, _buffer);
        return *this;
    }

    template <typename K, typename V>
    Logger& operator<<(const std::pair<K, V>&& pair) {
        (*this) << "<" << pair.first << ", " << pair.second << ">";
        return *this;
    }

    void flush();

    static void set_quiet_mode(bool quiet);

private:
    static bool _quiet;

    static tbb::spin_mutex& flush_mutex();

    std::unique_ptr<logger::TextFormatter> _text_formatter{
        std::make_unique<std::decay_t<decltype(logger::DEFAULT_TEXT)>>(logger::DEFAULT_TEXT)};
    std::unique_ptr<logger::ContainerFormatter> _container_formatter{
        std::make_unique<std::decay_t<decltype(logger::DEFAULT_CONTAINER)>>(logger::DEFAULT_CONTAINER)};

    std::ostringstream _buffer;
    std::ostream&      _out;
    std::string        _append;
    bool               _flushed{false};
};
} // namespace kaminpar
