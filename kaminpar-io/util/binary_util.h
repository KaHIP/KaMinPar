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
#include <fstream>
#include <limits>
#include <string>
#include <type_traits>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "kaminpar-io/util/io_exception.h"

#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::io {

class BinaryReader {
public:
  BinaryReader(const std::string &filename) {
    _file = open(filename.c_str(), O_RDONLY);
    if (_file == -1) {
      throw IOException("Cannot read the file that stores the graph");
    }

    struct stat file_info;
    if (fstat(_file, &file_info) == -1) {
      close(_file);
      throw IOException("Cannot determine the size of the file that stores the graph");
    }

    if (file_info.st_size == 0) {
      close(_file);
      throw IOException("The file that stores the graph is empty");
    }

    _length = static_cast<std::size_t>(file_info.st_size);
    _data = static_cast<std::uint8_t *>(mmap(nullptr, _length, PROT_READ, MAP_PRIVATE, _file, 0));
    if (_data == MAP_FAILED) {
      close(_file);
      throw IOException("Cannot map the file that stores the graph");
    }
  }

  ~BinaryReader() {
    munmap(_data, _length);
    close(_file);
  }

  template <typename T> [[nodiscard]] T read(const std::size_t position) const {
    require_available(position, sizeof(T));
    return *reinterpret_cast<const T *>(_data + position);
  }

  template <typename T> [[nodiscard]] const T *fetch(const std::size_t position) const {
    if constexpr (!std::is_void_v<T>) {
      require_available(position, sizeof(T));
    } else {
      require_available(position, 0);
    }
    return reinterpret_cast<const T *>(_data + position);
  }

  void require_available(const std::size_t position, const std::size_t bytes) const {
    if (position > _length || bytes > _length - position) [[unlikely]] {
      throw IOException("Unexpected end of binary graph file");
    }
  }

  [[nodiscard]] std::size_t length() const {
    return _length;
  }

private:
  int _file;
  std::size_t _length;
  std::uint8_t *_data;
};

class BinaryWriter {
public:
  BinaryWriter(const std::string &filename) : _out(filename, std::ios::binary) {
    if (!_out) {
      throw IOException("Cannot open output file");
    }
  }

  void write(const char *data, const std::size_t size) {
    if (!_out.write(data, size)) {
      throw IOException("Cannot write output file");
    }
  }

  template <typename T> void write_int(const T value) {
    write(reinterpret_cast<const char *>(&value), sizeof(T));
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
