/*******************************************************************************
 * @file:   mmap_toker.h
 * @author: Daniel Seemaier
 * @date:   26.10.2022
 * @brief:  Tokenizer for input files read with mmap.
 ******************************************************************************/
#pragma once

#include <cctype>
#include <stdexcept>
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

  const char *what() const noexcept override { return _msg.c_str(); }

private:
  std::string _msg;
};

template <bool throwing = false> class MappedFileToker {
public:
  explicit MappedFileToker(const std::string &filename) {
    _fd = open_file(filename);
    _position = 0;
    _length = file_size(_fd);
    _contents = static_cast<char *>(
        mmap(nullptr, _length, PROT_READ, MAP_PRIVATE, _fd, 0));
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
        KASSERT(valid_position() && str[i] == current(),
                "unexpected symbol: " << current() << ", but expected "
                                      << str[i]);
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
      KASSERT(valid_position() && current() == ch,
              "unexpected symbol: " << current() << ", but expected " << ch);
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

  [[nodiscard]] bool valid_position() const { return _position < _length; }
  [[nodiscard]] char current() const { return _contents[_position]; }
  void advance() { ++_position; }

  [[nodiscard]] std::size_t position() const { return _position; }
  [[nodiscard]] std::size_t length() const { return _length; }

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
