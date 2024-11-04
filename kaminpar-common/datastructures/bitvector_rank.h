/*******************************************************************************
 * A bit vector and rank data structure.
 *
 * @file:   bitvector_rank.h
 * @author: Daniel Salwasser
 * @date:   15.07.2024
 ******************************************************************************/
#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/math.h"

namespace kaminpar {

template <std::size_t BlockWidth = 512, std::size_t BlockHeaderWidth = 14>
class RankCombinedBitVector {
  static_assert(BlockWidth % 2 == 0, "Block width has to be a power of two.");
  static_assert(BlockWidth > 64, "Block width has to greater than 64 bits.");
  static_assert(BlockHeaderWidth <= 64, "Block header has to be a at most 64 bits wide.");
  static_assert(
      (static_cast<std::size_t>(1) << BlockHeaderWidth) > BlockWidth,
      "Superblock width has to be greater than the block width."
  );

  using Word = std::uint64_t;
  static constexpr std::size_t kWordWidth = sizeof(Word) * 8;

  static constexpr std::size_t kBlockWidth = BlockWidth;
  static constexpr std::size_t kBlockHeaderWidth = BlockHeaderWidth;
  static constexpr std::size_t kBlockDataWidth = kBlockWidth - kBlockHeaderWidth;
  static constexpr std::size_t kHeaderDataWidth = kWordWidth - kBlockHeaderWidth;
  static constexpr std::size_t kNumWordsPerBlock = kBlockWidth / kWordWidth;

  static constexpr std::size_t kSuperblockWidth = static_cast<std::size_t>(1) << kBlockHeaderWidth;
  static constexpr std::size_t kNumBlocksPerSuperblock = kSuperblockWidth / kBlockWidth;
  static constexpr std::size_t kNumWordsPerSuperblock = kSuperblockWidth / kWordWidth;
  static constexpr std::size_t kSuperblockDataWidth =
      kSuperblockWidth - kNumBlocksPerSuperblock * kBlockHeaderWidth;

  [[nodiscard]] inline static Word block_popcount(const Word *const data) {
    Word popcount = std::popcount(*data >> kBlockHeaderWidth);

    for (std::size_t i = 1; i < kNumWordsPerBlock; ++i) {
      popcount += std::popcount(data[i]);
    }

    return popcount;
  }

  template <typename Int>
  [[nodiscard]] inline static constexpr Int
  setbits(const std::size_t num_set_bits, const std::size_t start = 0) {
    if (num_set_bits == 0) {
      return 0;
    }

    constexpr Int kOnes = std::numeric_limits<Int>::max();
    constexpr std::size_t kWidth = std::numeric_limits<Int>::digits;
    return (kOnes >> static_cast<Int>(kWidth - num_set_bits)) << start;
  }

public:
  /*!
   * Constructs an empty bit vector.
   */
  explicit RankCombinedBitVector()
      : _length(0),
        _num_blocks(0),
        _data(0),
        _num_superblocks(0),
        _superblock_data(0) {}

  /*!
   * Constructs an uninitialized bit vector.
   *
   * @param length The number of bits that this bit vector contains.
   */
  explicit RankCombinedBitVector(const std::size_t length)
      : _length(length),
        _num_blocks(math::div_ceil(length, kBlockDataWidth)),
        _data(_num_blocks * kNumWordsPerBlock),
        _num_superblocks(math::div_ceil(length, kSuperblockDataWidth)),
        _superblock_data(_num_superblocks) {
    if (_num_blocks > 0) {
      // Fill the last bits with zeros such that the behaivour is predictable,
      // since this bits are nether set explicitly when the length is not a
      // multiple of the block-data width.
      Word *last_block = _data.data() + (_num_blocks - 1) * kNumWordsPerBlock;
      std::fill_n(last_block, kNumWordsPerBlock, 0);
    }
  }

  /*!
   * Constructs an bit vector with all bits set to one or zero. Note that the rank structure is not
   * initialized.
   *
   * @param length The number of bits that this bit vector contains.
   * @param set Whether to set the bits initially.
   */
  explicit RankCombinedBitVector(const std::size_t length, const bool set)
      : RankCombinedBitVector(length) {
    std::fill_n(_data.data(), _data.size(), set ? std::numeric_limits<Word>::max() : 0);
  }

  RankCombinedBitVector(RankCombinedBitVector &&) noexcept = default;
  RankCombinedBitVector &operator=(RankCombinedBitVector &&) noexcept = default;

  RankCombinedBitVector(RankCombinedBitVector const &) = delete;
  RankCombinedBitVector &operator=(RankCombinedBitVector const &) = delete;

  /*!
   * Sets a bit within this bit vector to zero.
   *
   * @param pos The position of the bit that is to be set to zero.
   */
  inline void unset(const std::size_t pos) {
    const std::size_t num_block = pos / kBlockDataWidth;
    const std::size_t block_pos = pos % kBlockDataWidth + kBlockHeaderWidth;

    const std::size_t num_local_word = block_pos / kWordWidth;
    const std::size_t num_word = num_block * kNumWordsPerBlock + num_local_word;

    _data[num_word] &= ~(static_cast<Word>(1) << (block_pos % kWordWidth));
  }

  /*!
   * Sets a bit within this bit vector to one.
   *
   * @param pos The position of the bit that is to be set to one.
   */
  inline void set(const std::size_t pos) {
    const std::size_t num_block = pos / kBlockDataWidth;
    const std::size_t block_pos = pos % kBlockDataWidth + kBlockHeaderWidth;

    const std::size_t num_local_word = block_pos / kWordWidth;
    const std::size_t num_word = num_block * kNumWordsPerBlock + num_local_word;

    _data[num_word] |= static_cast<Word>(1) << (block_pos % kWordWidth);
  }

  /*!
   * Sets a bit within this bit vector depending on a boolean value.
   *
   * @param pos The position of the bit that is to be set to one.
   * @param value Whether to set the bit.
   */
  inline void set(const std::size_t pos, const bool value) {
    const std::size_t num_block = pos / kBlockDataWidth;
    const std::size_t block_pos = pos % kBlockDataWidth + kBlockHeaderWidth;

    const std::size_t num_local_word = block_pos / kWordWidth;
    const std::size_t num_word = num_block * kNumWordsPerBlock + num_local_word;

    // The following implementation is due to the following source:
    // https://graphics.stanford.edu/~seander/bithacks.html#ConditionalSetOrClearBitsWithoutBranching
    const Word mask = static_cast<Word>(1) << (block_pos % kWordWidth);
    _data[num_word] = (_data[num_word] & ~mask) | (-value & mask);
  }

  /*!
   * Returns whether a bit within this bit vector is set.
   *
   * @param pos The position of the bit that is to be queried.
   * @param value Whether the bit is set.
   */
  [[nodiscard]] inline bool is_set(const std::size_t pos) const {
    const std::size_t num_block = pos / kBlockDataWidth;
    const std::size_t block_pos = pos % kBlockDataWidth + kBlockHeaderWidth;

    const std::size_t num_local_word = block_pos / kWordWidth;
    const std::size_t num_word = num_block * kNumWordsPerBlock + num_local_word;

    const Word word = _data[num_word];
    const std::size_t word_pos = block_pos % kWordWidth;

    const bool is_set = ((word >> word_pos) & static_cast<Word>(1)) == 1;
    return is_set;
  }

  /*!
   * Updates this rank data structure such that updates to the bit vector since
   * the initialization or the last update are reflected.
   */
  void update() {
    const Word *const data = _data.data();
    const std::size_t num_words = _num_blocks * kNumWordsPerBlock;

    Word cur_rank = 0;
    Word cur_block_rank = 0;
    std::size_t cur_num_super_block = 0;
    for (std::size_t i = 0; i < num_words; i += kNumWordsPerBlock) {
      const bool is_superblock_word = (i % kNumWordsPerSuperblock) == 0;

      if (is_superblock_word) [[unlikely]] {
        cur_rank += cur_block_rank;
        _superblock_data[cur_num_super_block] = cur_rank;

        cur_num_super_block += 1;
        cur_block_rank = 0;
      }

      _data[i] = (_data[i] & setbits<Word>(kHeaderDataWidth, kBlockHeaderWidth)) | cur_block_rank;
      cur_block_rank += block_popcount(data + i);
    }
  }

  /**
   * Returns the number of bits equal to one up to a position.
   *
   * @param pos The position up to which bits are to be taken into account.
   * @return The number of bits equal to zero up to the position.
   */
  [[nodiscard]] inline Word rank(const std::size_t pos) const {
    const std::size_t num_block = pos / kBlockDataWidth;
    const std::size_t block_pos = pos % kBlockDataWidth + kBlockHeaderWidth;

    std::size_t num_word = block_pos / kWordWidth;
    const std::size_t word_pos = block_pos % kWordWidth;

    const std::size_t num_superblock = pos / kSuperblockDataWidth;
    Word rank = _superblock_data[num_superblock];

    const Word *const data = _data.data() + num_block * kNumWordsPerBlock;
    const Word first_word = *data;
    rank += first_word & setbits<Word>(kBlockHeaderWidth);

    if (num_word == 0) [[unlikely]] {
      const std::size_t shift = (kWordWidth + kBlockHeaderWidth) - word_pos;
      rank += std::popcount((first_word >> kBlockHeaderWidth) << shift) *
              (word_pos != kBlockHeaderWidth);
    } else {
      rank += std::popcount(first_word >> kBlockHeaderWidth);

      std::size_t i = 1;
      while (i < num_word) {
        rank += std::popcount(data[i++]);
      }

      const std::size_t shift = kWordWidth - word_pos;
      rank += std::popcount(data[i] << shift) * (word_pos != 0);
    }

    return rank;
  }

  /**
   * Returns the number of bits that this bit vector contains.
   *
   * @return The number of bits that this bit vector contains.
   */
  [[nodiscard]] inline std::size_t length() const {
    return _length;
  }

private:
  std::size_t _length;
  std::size_t _num_blocks;
  StaticArray<Word> _data;

  std::size_t _num_superblocks;
  StaticArray<Word> _superblock_data;
};

} // namespace kaminpar
