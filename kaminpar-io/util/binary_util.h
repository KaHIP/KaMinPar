/*******************************************************************************
 * Reader and writer for binary files.
 *
 * @file:   bianry_util.h
 * @author: Daniel Salwasser
 * @date:   07.07.2024
 ******************************************************************************/
#pragma once

#include <cstddef>
#include <cstdint>
#include <exception>
#include <fstream>
#include <string>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::io {

class BinaryReaderException : public std::exception {
public:
  BinaryReaderException(std::string msg) : _msg(std::move(msg)) {}

  [[nodiscard]] const char *what() const noexcept override {
    return _msg.c_str();
  }

private:
  std::string _msg;
};

class BinaryReader {
public:
  BinaryReader(const std::string &filename) {
    _file = open(filename.c_str(), O_RDONLY);
    if (_file == -1) {
      throw BinaryReaderException("Cannot read the file that stores the graph");
    }

    struct stat file_info;
    if (fstat(_file, &file_info) == -1) {
      close(_file);
      throw BinaryReaderException("Cannot determine the size of the file that stores the graph");
    }

    _length = static_cast<std::size_t>(file_info.st_size);
    _data = static_cast<std::uint8_t *>(mmap(nullptr, _length, PROT_READ, MAP_PRIVATE, _file, 0));
    if (_data == MAP_FAILED) {
      close(_file);
      throw BinaryReaderException("Cannot map the file that stores the graph");
    }
  }

  ~BinaryReader() {
    munmap(_data, _length);
    close(_file);
  }

  template <typename T> [[nodiscard]] T read(const std::size_t position) const {
    return *reinterpret_cast<T *>(_data + position);
  }

  template <typename T> [[nodiscard]] const T *fetch(const std::size_t position) const {
    return reinterpret_cast<const T *>(_data + position);
  }

private:
  int _file;
  std::size_t _length;
  std::uint8_t *_data;
};

class BinaryWriter {
public:
  BinaryWriter(const std::string &filename) : _out(filename, std::ios::binary) {}

  void write(const char *data, const std::size_t size) {
    _out.write(data, size);
  }

  template <typename T> void write_int(const T value) {
    _out.write(reinterpret_cast<const char *>(&value), sizeof(T));
  }

  template <typename T> void write_raw_static_array(const StaticArray<T> &static_array) {
    const char *data = reinterpret_cast<const char *>(static_array.data());
    const std::size_t size = static_array.size() * sizeof(T);
    write(data, size);
  }

private:
  std::ofstream _out;
};

} // namespace kaminpar::io
