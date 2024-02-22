/*******************************************************************************
 * Helper class for console logging.
 *
 * @file:   logger.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#include "kaminpar-common/logger.h"

#include <cmath>

namespace kaminpar {
namespace logger {
void CompactContainerFormatter::print(const std::vector<std::string> &container, std::ostream &out)
    const {
  bool first{true};
  for (const auto &element : container) {
    if (!first) {
      out << _sep;
    }
    out << element;
    first = false;
  }
}

void Table::print(const std::vector<std::string> &container, std::ostream &out) const {
  using namespace std::literals;

  const std::size_t width = (_width == 0) ? std::sqrt(container.size()) : _width;
  std::size_t max_width = 0;
  for (const auto &element : container) {
    max_width = std::max(max_width, element.length());
  }

  std::stringstream ss_h_del;
  for (std::size_t i = 0; i < width; ++i) {
    ss_h_del << "+" << std::string(max_width + 2, '-');
  }
  ss_h_del << "+";
  const std::string h_del = ss_h_del.str();

  out << h_del << "\n";
  std::size_t column = 0;
  for (const auto &element : container) {
    out << "| " << element << std::string(max_width + 1 - element.length(), ' ');
    if (++column == width) {
      out << "|\n" << h_del << "\n";
      column = 0;
    }
  }

  if (column > 0) {
    while (column < width) {
      out << "| " << std::string(max_width + 1, ' ');
      ++column;
    }
    out << "|\n" << h_del << "\n";
  }
}

void DefaultTextFormatter::print(const std::string &text, std::ostream &out) const {
  out << text;
}

void Colorized::print(const std::string &text, std::ostream &out) const {
  switch (_color) {
  case Color::RED:
    out << "\u001b[31m";
    break;
  case Color::GREEN:
    out << "\u001b[32m";
    break;
  case Color::ORANGE:
    out << "\u001b[33m";
    break;
  case Color::MAGENTA:
    out << "\u001b[35m";
    break;
  case Color::CYAN:
    out << "\u001b[36m";
    break;
  default:
    break; // do nothing
  }
  out << text << "\u001b[0m";
}

using namespace std::literals::string_view_literals;
DefaultTextFormatter DEFAULT_TEXT{};
Colorized RED{logger::Colorized::Color::RED};
Colorized GREEN{logger::Colorized::Color::GREEN};
Colorized MAGENTA{logger::Colorized::Color::MAGENTA};
Colorized ORANGE{logger::Colorized::Color::ORANGE};
Colorized CYAN{logger::Colorized::Color::CYAN};
Colorized RESET{logger::Colorized::Color::RESET};
CompactContainerFormatter DEFAULT_CONTAINER{", "sv};
CompactContainerFormatter COMPACT{","sv};
Table TABLE{0};
} // namespace logger

std::atomic<std::uint8_t> Logger::_quiet = 0;

Logger::Logger() : Logger(std::cout) {}
Logger::Logger(std::ostream &out, std::string append)
    : _buffer(),
      _out(out),
      _append(std::move(append)) {}

void Logger::flush() {
  if (_quiet) {
    return;
  }

  if (!_flushed) {
    tbb::spin_mutex::scoped_lock lock(flush_mutex());
    _out << _buffer.str() << _append << std::flush;
  }

  _flushed = true;
}

tbb::spin_mutex &Logger::flush_mutex() {
  static tbb::spin_mutex mutex;
  return mutex;
}

void Logger::set_quiet_mode(const bool quiet) {
  _quiet = quiet;
}

bool Logger::is_quiet() {
  return _quiet;
}
} // namespace kaminpar
