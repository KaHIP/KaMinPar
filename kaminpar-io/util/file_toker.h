/*******************************************************************************
 * Tokener that transforms a text file into tokens.
 *
 * @file:   file_toker.h
 * @author: Daniel Seemaier
 * @date:   26.10.2022
 ******************************************************************************/
#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <exception>
#include <limits>
#include <string>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "kaminpar-common/assert.h"

namespace kaminpar::io {

class TokerException : public std::exception {
public:
  TokerException(std::string msg) : _msg(std::move(msg)) {}

  [[nodiscard]] const char *what() const noexcept override {
    return _msg.c_str();
  }

private:
  std::string _msg;
};

class MappedFileToker {
public:
  explicit MappedFileToker(
      const std::string &filename,
      const std::size_t begin = 0,
      const std::size_t end = std::numeric_limits<std::size_t>::max()
  ) {
    _fd = open(filename.c_str(), O_RDONLY);
    if (_fd == -1) {
      throw TokerException("Cannot open input file");
    }

    struct stat file_info{};
    if (fstat(_fd, &file_info) == -1) {
      close(_fd);
      throw TokerException("Cannot get input file status");
    }

    _length = static_cast<std::size_t>(file_info.st_size);

    _contents = static_cast<char *>(mmap(nullptr, _length, PROT_READ, MAP_PRIVATE, _fd, 0));
    if (_contents == MAP_FAILED) {
      close(_fd);
      throw TokerException("Cannot map input file into memory");
    }

    _owns_mapping = true;
    set_bounds(begin, end);
  }

  MappedFileToker(const MappedFileToker &parent, const std::size_t begin, const std::size_t end)
      : _fd(-1),
        _length(parent._length),
        _contents(parent._contents),
        _owns_mapping(false) {
    set_bounds(begin, end);
  }

  MappedFileToker(const MappedFileToker &) = delete;
  MappedFileToker &operator=(const MappedFileToker &) = delete;

  ~MappedFileToker() {
    if (_owns_mapping) {
      munmap(_contents, _length);
      close(_fd);
    }
  }

  void reset() {
    _position = _begin;
  }

  void seek(const std::size_t position) {
    KASSERT(_begin <= position && position <= _end);
    _position = position;
  }

  [[nodiscard]] std::size_t advance_to_line_begin(std::size_t position) const {
    KASSERT(_begin <= position && position <= _end);
    if (position <= _begin) {
      return _begin;
    }
    if (position >= _end) {
      return _end;
    }

    while (position < _end && _contents[position - 1] != '\n') {
      ++position;
    }

    return position;
  }

  inline void skip_spaces() {
    while (valid_position() && current() == ' ') {
      advance();
    }
  }

  inline void skip_line() {
    while (valid_position() && current() != '\n') {
      advance();
    }

    if (valid_position()) {
      advance();
    }
  }

  inline std::uint64_t scan_uint() {
    KASSERT(valid_position() && std::isdigit(current()));

    std::uint64_t number = 0;
    while (valid_position() && std::isdigit(current())) {
      const int digit = current() - '0';
      number = number * 10 + digit;
      advance();
    }

    skip_spaces();
    return number;
  }

  inline void skip_uint() {
    KASSERT(valid_position() && std::isdigit(current()));

    while (valid_position() && std::isdigit(current())) {
      advance();
    }

    skip_spaces();
  }

  inline void consume_string(const char *str) {
    std::size_t i = 0;
    while (str[i] != '\0') {
      KASSERT(
          valid_position() && str[i] == current(),
          "unexpected symbol: " << current() << ", but expected " << str[i]
      );

      advance();
      ++i;
    }
  }

  inline void consume_char([[maybe_unused]] const char ch) {
    KASSERT(
        valid_position() && current() == ch,
        "unexpected symbol: " << current() << ", but expected " << ch
    );

    advance();
  }

  inline bool test_string(const char *str) {
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

  [[nodiscard]] inline bool valid_position() const {
    return _position < _end;
  }

  [[nodiscard]] inline char current() const {
    return _contents[_position];
  }

  inline void advance() {
    ++_position;
  }

  [[nodiscard]] inline std::size_t position() const {
    return _position;
  }

  [[nodiscard]] inline std::size_t length() const {
    return _length;
  }

  [[nodiscard]] inline const char *contents() const {
    return _contents;
  }

private:
  void set_bounds(const std::size_t begin, const std::size_t end) {
    const std::size_t bounded_end = std::min(end, _length);
    KASSERT(begin <= bounded_end && bounded_end <= _length);

    _begin = begin;
    _end = bounded_end;
    _position = _begin;
  }

  int _fd = -1;
  std::size_t _position = 0;
  std::size_t _length = 0;
  char *_contents = nullptr;
  std::size_t _begin = 0;
  std::size_t _end = 0;
  bool _owns_mapping = false;
};

} // namespace kaminpar::io
