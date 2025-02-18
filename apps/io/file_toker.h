/*******************************************************************************
 * Tokener that transforms a text file into tokens.
 *
 * @file:   file_toker.h
 * @author: Daniel Seemaier
 * @date:   26.10.2022
 ******************************************************************************/
#pragma once

#include <cctype>
#include <cstdint>
#include <exception>
#include <string>

#include <fcntl.h>
#include <kassert/kassert.hpp>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

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
  explicit MappedFileToker(const std::string &filename) {
    _fd = open(filename.c_str(), O_RDONLY);
    if (_fd == -1) {
      throw TokerException("Cannot open input file");
    }

    struct stat file_info{};
    if (fstat(_fd, &file_info) == -1) {
      close(_fd);
      throw TokerException("Cannot get input file status");
    }

    _position = 0;
    _length = static_cast<std::size_t>(file_info.st_size);

    _contents = static_cast<char *>(mmap(nullptr, _length, PROT_READ, MAP_PRIVATE, _fd, 0));
    if (_contents == MAP_FAILED) {
      close(_fd);
      throw TokerException("Cannot map input file into memory");
    }
  }

  ~MappedFileToker() {
    munmap(_contents, _length);
    close(_fd);
  }

  void reset() {
    _position = 0;
  }

  void seek(const std::size_t position) {
    _position = position;
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

  inline void consume_char(const char ch) {
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
    return _position < _length;
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

private:
  int _fd;
  std::size_t _position;
  std::size_t _length;
  char *_contents;
};

} // namespace kaminpar::io
