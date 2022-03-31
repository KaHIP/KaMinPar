/*******************************************************************************
 * @file:   console_io.h
 *
 * @author: Daniel Seemaier
 * @date:   21.09.21
 * @brief:  Helper functions for console IO.
 ******************************************************************************/
#pragma once

#include <mutex>

#include "kaminpar/definitions.h"

namespace kaminpar::cio {
void print_banner(const std::string& title);

class ProgressBar {
    constexpr static std::size_t COLS          = 80;
    constexpr static std::size_t TITLE_COLS    = 20;
    constexpr static std::size_t TITLE_PADDING = 1;

public:
    explicit ProgressBar(std::size_t n, const std::string& title = "", bool silent = false);

    void set_step(std::size_t step);

    void step(const std::string& description = "");

    void update(std::size_t i, const std::string& description = "");

    void stop();

private:
    [[nodiscard]] static std::string make_title(const std::string& title);

    std::string          _title;
    std::size_t          _n;
    bool                 _silent;
    std::size_t          _current{0};
    std::size_t          _step{std::max<std::size_t>(1, _n / 100)};
    std::recursive_mutex _update_mutex;
};
} // namespace kaminpar::cio