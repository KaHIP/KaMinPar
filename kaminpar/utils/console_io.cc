/*******************************************************************************
 * @file:   console_io.cc
 *
 * @author: Daniel Seemaier
 * @date:   21.09.21
 * @brief:  Helper functions for console IO.
 ******************************************************************************/
#include "kaminpar/utils/console_io.h"

namespace kaminpar::cio {
void print_banner(const std::string& title) {
    LOG;
    LOG << std::string(80, '*');
    LOG << "* " << title << std::string(80 - 4 - title.size(), ' ') << " *";
    LOG << std::string(80, '*');
    LOG;
}

ProgressBar::ProgressBar(const std::size_t n, const std::string& title, const bool silent)
    : _title{make_title(title)},
      _n{n},
      _silent{silent} {}

void ProgressBar::set_step(const std::size_t step) {
    _step = step;
}

void ProgressBar::step(const std::string& description) {
    std::scoped_lock lk(_update_mutex);

#ifndef KAMINPAR_EXPERIMENTS_MODE
    update(_current + 1, description);
#else  // KAMINPAR_EXPERIMENTS_MODE
    (void)description;
#endif // KAMINPAR_EXPERIMENTS_MODE
}

void ProgressBar::update(const std::size_t i, const std::string& description) {
    std::scoped_lock lk(_update_mutex);

#ifndef KAMINPAR_EXPERIMENTS_MODE
    ASSERT(i <= _n);
    if (_silent) {
        return;
    }

    _current = i;
    if ((i % _step) == 0 || i == _n) {
        const std::size_t full     = COLS - 2;
        const std::size_t progress = full * i / _n;
        std::cout << "\033[1K\r";
        std::cout << _title;
        std::cout << "[" << std::string(progress, '#') << std::string(full - progress, ' ') << "] " << i << " / " << _n
                  << " -- " << (100 * i / _n) << "%";
        if (!description.empty()) {
            std::cout << " -- " << description;
        }
        std::cout << std::flush;
    }
#else  // KAMINPAR_EXPERIMENTS_MODE
    (void)i;
    (void)description;
#endif // KAMINPAR_EXPERIMENTS_MODE
}

void ProgressBar::stop() {
    std::scoped_lock lk(_update_mutex);

#ifndef KAMINPAR_EXPERIMENTS_MODE
    if (_silent) {
        return;
    }
    update(_n, "done");
    std::cout << std::endl;
#endif // KAMINPAR_EXPERIMENTS_MODE
}

[[nodiscard]] std::string ProgressBar::make_title(const std::string& title) {
    if (title.empty()) {
        return "";
    }
    const std::string shrunk_title = (title.size() + 2 * TITLE_PADDING > TITLE_COLS)
                                         ? title.substr(0, TITLE_COLS - 2 * TITLE_PADDING - 3) + "..."
                                         : title + ' ';
    return shrunk_title + std::string(1 + TITLE_COLS - shrunk_title.size() - 2 * TITLE_PADDING, '.') + ' ';
}
} // namespace kaminpar::cio