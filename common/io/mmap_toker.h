#pragma once

#include <cctype>
#include <stdexcept>
#include <string>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace kaminpar::io {
class MappedFileToker {
public:
    explicit MappedFileToker(const std::string& filename) {
        _fd       = open_file(filename);
        _position = 0;
        _length   = file_size(_fd);
        _contents = static_cast<char*>(mmap(nullptr, _length, PROT_READ, MAP_PRIVATE, _fd, 0));
        if (_contents == MAP_FAILED) {
            close(_fd);
            throw std::runtime_error("mmap failed");
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

    template <typename Int>
    inline Int scan_uint() {
        Int number = 0;
        while (valid_position() && std::isdigit(current())) {
            const int digit = current() - '0';
            number          = number * 10 + digit;
            advance();
        }
        skip_spaces();
        return number;
    }

    void skip_uint() {
        while (valid_position() && std::isdigit(current())) {
            advance();
        }
        skip_spaces();
    }

    void consume_string(const char* str) {
        std::size_t i = 0;
        while (str[i] != '\0') {
            if (!valid_position() || str[i] != current()) {
                throw std::runtime_error("unexpected char");
            }
            advance();
            ++i;
        }
    }

    void consume_char(const char ch) {
        if (!valid_position() || current() != ch) {
            throw std::runtime_error("unexpected char");
        }
        advance();
    }

    bool test_string(const char* str) {
        std::size_t pos   = _position;
        bool        match = true;
        std::size_t i     = 0;

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
    static int open_file(const std::string& filename) {
        const int file = open(filename.c_str(), O_RDONLY);
        if (file < 0) {
            throw std::runtime_error("cannot open input file");
        }
        return file;
    }

    static std::size_t file_size(const int fd) {
        struct stat file_info {};
        fstat(fd, &file_info);
        return static_cast<std::size_t>(file_info.st_size);
    }

    int         _fd;
    std::size_t _position;
    std::size_t _length;
    char*       _contents;
};
} // namespace kaminpar::io
