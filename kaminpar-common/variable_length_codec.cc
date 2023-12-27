/*******************************************************************************
 * Variable length encoding and decoding methods for integers.
 *
 * @file:   variable_length_codec.cc
 * @author: Daniel Salwasser
 * @date:   26.12.2023
 ******************************************************************************/
#include "kaminpar-common/variable_length_codec.h"

namespace kaminpar::debug {

static VariabeLengthStats stats = {0, 0, 0, 0, 0, 0};

void varint_stats_reset() {
  stats.varint_count = 0;
  stats.signed_varint_count = 0;
  stats.marked_varint_count = 0;

  stats.varint_bytes = 0;
  stats.signed_varint_bytes = 0;
  stats.marked_varint_bytes = 0;
}

VariabeLengthStats &varint_stats_global() {
  return stats;
}

} // namespace kaminpar::debug
