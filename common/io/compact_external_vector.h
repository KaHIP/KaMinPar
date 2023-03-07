/*******************************************************************************
 * @file:   compact_external_vector.h
 * @author: Daniel Seemaier
 * @date:   26.10.2022
 * @brief:  Tokenizer for input files read with mmap.
 ******************************************************************************/
#pragma once

#include <fstream>
#include <string>

#include "common/math.h"

namespace kaminpar {
template <typename UInt, template <typename> typename Container>
void write(const std::string& filename, const Container<UInt>& data, const UInt max, const bool write_header) {
    const std::uint64_t max64          = static_cast<std::uint64_t>(max);
    const std::uint64_t bits_per_entry = ceil_log2(max + 1);
    const std::uint64_t size           = data.size();
    const std::uint64_t compact_size   = size * bits_per_entry / 64 + 1;

    std::vector<std::uint64_t> compact_data(compact_size);

    std::uint64_t current_entry_pos = 0;
    std::uint64_t current_bit_pos   = 0;
    for (std::size_t i = 0; i < data.size(); ++i) {
        if (current_bit_pos + bits_per_entry > 64) {
            const std::uint64_t prev_bits = 64 - current_bit_pos;
            const std::uint64_t next_bits = bits_per_entry - prev_bits;
            compact_data[current_entry_pos] |= (data[i] >> next_bits) & ((1 << prev_bits) - 1);
            compact_data[current_entry_pos + 1] |= (data[i] & ((1 << next_bits) - 1)) << (64 - next_bits);
        } else {
            compact_data[current_entry_pos] |= (data[i] << (64 - current_bit_pos - bits_per_entry));
        }

        current_bit_pos += bits_per_entry;
        if (current_bit_pos >= 64) {
            current_bit_pos -= 64;
            current_entry_pos += 1;
        }
    }

    std::ofstream out(filename, std::ios_base::binary | std::ios_base::trunc);
    if (write_header) {
        out.write(reinterpret_cast<const char*>(&max64), sizeof(std::uint64_t));
    }
    out.write(reinterpret_cast<const char*>(&size), sizeof(std::uint64_t));
    out.write(reinterpret_cast<const char*>(compact_data.data()), sizeof(std::uint64_t) * compact_size);
}
} // namespace kaminpar
