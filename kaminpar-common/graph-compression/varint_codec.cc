/*******************************************************************************
 * Encoding and decoding methods for VarInts.
 *
 * @file:   varint_codec.cc
 * @author: Daniel Salwasser
 * @date:   26.12.2023
 ******************************************************************************/
#include "kaminpar-common/graph-compression/varint_codec.h"

namespace kaminpar {

namespace debug {

static VarIntStats stats = {0, 0, 0, 0, 0, 0};

void varint_stats_reset() {
  stats.varint_count = 0;
  stats.signed_varint_count = 0;
  stats.marked_varint_count = 0;

  stats.varint_bytes = 0;
  stats.signed_varint_bytes = 0;
  stats.marked_varint_bytes = 0;
}

VarIntStats &varint_stats_global() {
  return stats;
}

} // namespace debug

} // namespace kaminpar
