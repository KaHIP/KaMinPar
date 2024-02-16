/*******************************************************************************
 * Sequential METIS parser.
 *
 * @file:   metis_parser.h
 * @author: Daniel Seemaier
 * @date:   26.10.2022
 ******************************************************************************/
#pragma once

#include <cctype>
#include <stdexcept>
#include <string>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "kaminpar-common/assert.h"
#include "kaminpar-common/logger.h"

namespace kaminpar::io {
class TokerException : public std::exception {
public:
  TokerException(std::string msg) : _msg(std::move(msg)) {}

  const char *what() const noexcept override {
    return _msg.c_str();
  }

private:
  std::string _msg;
};

template <bool throwing = false> class MappedFileToker {
public:
  explicit MappedFileToker(const std::string &filename) {
    _fd = open_file(filename);
    _position = 0;
    _length = file_size(_fd);
    _contents = static_cast<char *>(mmap(nullptr, _length, PROT_READ, MAP_PRIVATE, _fd, 0));
    if (_contents == MAP_FAILED) {
      close(_fd);
      if constexpr (throwing) {
        throw TokerException("mmap failed");
      } else {
        KASSERT(_contents != MAP_FAILED);
      }
    }
  }

  ~MappedFileToker() {
    munmap(_contents, _length);
    close(_fd);
  }

  void skip_spaces() {
    while (valid_position() && current() == ' ') {
      advance();
    }
  }

  void skip_line() {
    while (valid_position() && current() != '\n') {
      advance();
    }
    if (valid_position()) {
      advance();
    }
  }

  inline std::uint64_t scan_uint() {
    expect_uint_start();

    std::uint64_t number = 0;
    while (valid_position() && std::isdigit(current())) {
      const int digit = current() - '0';
      number = number * 10 + digit;
      advance();
    }
    skip_spaces();
    return number;
  }

  void skip_uint() {
    expect_uint_start();

    while (valid_position() && std::isdigit(current())) {
      advance();
    }
    skip_spaces();
  }

  void consume_string(const char *str) {
    std::size_t i = 0;
    while (str[i] != '\0') {
      if constexpr (throwing) {
        if (!valid_position() || str[i] != current()) {
          throw TokerException("unexpected symbol");
        }
      } else {
        KASSERT(
            valid_position() && str[i] == current(),
            "unexpected symbol: " << current() << ", but expected " << str[i]
        );
      }
      advance();
      ++i;
    }
  }

  void consume_char(const char ch) {
    if constexpr (throwing) {
      if (!valid_position() || current() != ch) {
        throw TokerException("unexpected char");
      }
    } else {
      KASSERT(
          valid_position() && current() == ch,
          "unexpected symbol: " << current() << ", but expected " << ch
      );
    }
    advance();
  }

  bool test_string(const char *str) {
    std::size_t pos = _position;
    bool match = true;
    std::size_t i = 0;

    while (str[i] != '\0') {
      if (!valid_position() || str[i] != current()) {
        match = false;
        break;
      }
      advance();
      ++i;
    }

    _position = pos;
    return match;
  }

  [[nodiscard]] bool valid_position() const {
    return _position < _length;
  }
  [[nodiscard]] char current() const {
    return _contents[_position];
  }
  void advance() {
    ++_position;
  }

  [[nodiscard]] std::size_t position() const {
    return _position;
  }
  [[nodiscard]] std::size_t length() const {
    return _length;
  }

private:
  static int open_file(const std::string &filename) {
    const int file = open(filename.c_str(), O_RDONLY);
    if constexpr (throwing) {
      if (file < 0) {
        throw TokerException("could not open the input file");
      }
    } else {
      KASSERT(file >= 0, "could not open the input file: " << filename);
    }
    return file;
  }

  static std::size_t file_size(const int fd) {
    struct stat file_info {};
    fstat(fd, &file_info);
    return static_cast<std::size_t>(file_info.st_size);
  }

  void expect_uint_start() {
    if constexpr (throwing) {
      if (!valid_position()) {
        throw TokerException("scan_uint() called on end of input stream");
      }
      if (!std::isdigit(current())) {
        throw TokerException("expected start of unsigned integer");
      }
    } else {
      KASSERT(valid_position() && std::isdigit(current()));
    }
  }

  int _fd;
  std::size_t _position;
  std::size_t _length;
  char *_contents;
};
} // namespace kaminpar::io

namespace kaminpar::io::metis {
struct Format {
  std::uint64_t number_of_nodes = 0;
  std::uint64_t number_of_edges = 0;
  bool has_node_weights = false;
  bool has_edge_weights = false;
};

template <bool throwing> inline Format parse_header(MappedFileToker<throwing> &toker) {
  toker.skip_spaces();
  while (toker.current() == '%') {
    toker.skip_line();
    toker.skip_spaces();
  }

  const std::uint64_t number_of_nodes = toker.scan_uint();
  const std::uint64_t number_of_edges = toker.scan_uint();
  const std::uint64_t format = (toker.current() != '\n') ? toker.scan_uint() : 0;
  toker.consume_char('\n');

  if (format != 0 && format != 1 && format != 10 && format != 11 && format && format != 100 &&
      format != 110 && format != 101 && format != 111) {
    LOG_WARNING << "invalid or unsupported graph format";
  }

  [[maybe_unused]] const bool has_node_sizes = format / 100; // == 1xx
  const bool has_node_weights = (format % 100) / 10;         // == x1x
  const bool has_edge_weights = format % 10;                 // == xx1

  if (has_node_sizes) {
    LOG_WARNING << "ignoring node sizes";
  }

  return {
      .number_of_nodes = number_of_nodes,
      .number_of_edges = number_of_edges,
      .has_node_weights = has_node_weights,
      .has_edge_weights = has_edge_weights,
  };
}

template <bool throwing> inline Format parse_header(const std::string &filename) {
  MappedFileToker<throwing> toker(filename);
  return parse_header(toker);
}

template <bool throwing, typename GraphFormatCB, typename NextNodeCB, typename NextEdgeCB>
void parse(
    MappedFileToker<throwing> &toker,
    GraphFormatCB &&format_cb,
    NextNodeCB &&next_node_cb,
    NextEdgeCB &&next_edge_cb
) {
  static_assert(std::is_invocable_v<GraphFormatCB, Format>);
  static_assert(std::is_invocable_v<NextNodeCB, std::uint64_t>);
  static_assert(std::is_invocable_v<NextEdgeCB, std::uint64_t, std::uint64_t>);

  constexpr bool stoppable = std::is_invocable_r_v<bool, NextNodeCB, std::uint64_t>;

  const Format format = parse_header(toker);
  const bool read_node_weights = format.has_node_weights;
  const bool read_edge_weights = format.has_edge_weights;
  format_cb(format);

  bool has_exited_preemptively = false;

  for (std::uint64_t u = 0; u < format.number_of_nodes; ++u) {
    toker.skip_spaces();
    while (toker.current() == '%') {
      toker.skip_line();
      toker.skip_spaces();
    }

    std::uint64_t node_weight = 1;
    if (format.has_node_weights) {
      if (read_node_weights) {
        node_weight = toker.scan_uint();
      } else {
        toker.scan_uint();
      }
    }
    if constexpr (stoppable) {
      if (!next_node_cb(node_weight)) {
        has_exited_preemptively = true;
        break;
      }
    } else {
      next_node_cb(node_weight);
    }

    while (std::isdigit(toker.current())) {
      const std::uint64_t v = toker.scan_uint() - 1;
      std::uint64_t edge_weight = 1;
      if (format.has_edge_weights) {
        if (read_edge_weights) {
          edge_weight = toker.scan_uint();
        } else {
          toker.scan_uint();
        }
      }
      next_edge_cb(edge_weight, v);
    }

    if (toker.valid_position()) {
      toker.consume_char('\n');
    }
  }

  if (!has_exited_preemptively) {
    while (toker.current() == '%') {
      toker.skip_line();
    }

    if (toker.valid_position()) {
      LOG_WARNING << "ignorning extra lines in input file";
    }
  }
}

template <bool throwing, typename GraphFormatCB, typename NextNodeCB, typename NextEdgeCB>
void parse(
    const std::string &filename,
    GraphFormatCB &&format_cb,
    NextNodeCB &&next_node_cb,
    NextEdgeCB &&next_edge_cb
) {
  MappedFileToker<throwing> toker(filename);
  parse(
      toker,
      std::forward<GraphFormatCB>(format_cb),
      std::forward<NextNodeCB>(next_node_cb),
      std::forward<NextEdgeCB>(next_edge_cb)
  );
}
} // namespace kaminpar::io::metis
