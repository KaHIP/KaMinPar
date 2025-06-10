/*******************************************************************************
 * Public C library interface of KaMinPar.
 *
 * @file:   ckaminpar.cc
 * @author: Daniel Seemaier
 * @date:   09.12.2024
 ******************************************************************************/
#include "kaminpar-shm/ckaminpar.h"

#include <span>
#include <string>
#include <vector>

#include "kaminpar-shm/kaminpar.h"

struct kaminpar_context_t {
  kaminpar::shm::Context ctx;
};

struct kaminpar_t {
  kaminpar_t(const int num_threads, const kaminpar_context_t *ctx)
      : n(0),
        kaminpar(num_threads, ctx->ctx) {}

  kaminpar::shm::NodeID n;
  kaminpar::KaMinPar kaminpar;
};

kaminpar_context_t *kaminpar_create_context_by_preset_name(const char *name) {
  kaminpar_context_t *ctx = new kaminpar_context_t;
  ctx->ctx = kaminpar::shm::create_context_by_preset_name(name);
  return ctx;
}

kaminpar_context_t *kaminpar_create_default_context() {
  kaminpar_context_t *ctx = new kaminpar_context_t;
  ctx->ctx = kaminpar::shm::create_default_context();
  return ctx;
}

kaminpar_context_t *kaminpar_create_strong_context() {
  kaminpar_context_t *ctx = new kaminpar_context_t;
  ctx->ctx = kaminpar::shm::create_strong_context();
  return ctx;
}

kaminpar_context_t *kaminpar_create_terapart_context() {
  kaminpar_context_t *ctx = new kaminpar_context_t;
  ctx->ctx = kaminpar::shm::create_terapart_context();
  return ctx;
}

kaminpar_context_t *kaminpar_create_largek_context() {
  kaminpar_context_t *ctx = new kaminpar_context_t;
  ctx->ctx = kaminpar::shm::create_largek_context();
  return ctx;
}

kaminpar_context_t *kaminpar_create_vcycle_context(bool restrict_refinement) {
  kaminpar_context_t *ctx = new kaminpar_context_t;
  ctx->ctx = kaminpar::shm::create_vcycle_context(restrict_refinement);
  return ctx;
}

void kaminpar_context_free(kaminpar_context_t *ctx) {
  delete ctx;
}

kaminpar_t *kaminpar_create(int num_threads, kaminpar_context_t *ctx) {
  return new kaminpar_t(num_threads, ctx);
}

void kaminpar_free(kaminpar_t *kaminpar) {
  delete kaminpar;
}

void kaminpar_set_output_level(kaminpar_t *kaminpar, kaminpar_output_level_t output_level) {
  kaminpar::OutputLevel cpp_output_level;

  switch (output_level) {
  case KAMINPAR_OUTPUT_LEVEL_QUIET:
    cpp_output_level = kaminpar::OutputLevel::QUIET;
    break;
  case KAMINPAR_OUTPUT_LEVEL_PROGRESS:
    cpp_output_level = kaminpar::OutputLevel::PROGRESS;
    break;
  case KAMINPAR_OUTPUT_LEVEL_APPLICATION:
    cpp_output_level = kaminpar::OutputLevel::APPLICATION;
    break;
  case KAMINPAR_OUTPUT_LEVEL_EXPERIMENT:
    cpp_output_level = kaminpar::OutputLevel::EXPERIMENT;
    break;
  case KAMINPAR_OUTPUT_LEVEL_DEBUG:
    cpp_output_level = kaminpar::OutputLevel::DEBUG;
    break;

  default:
    cpp_output_level = kaminpar::OutputLevel::APPLICATION;
  }

  kaminpar->kaminpar.set_output_level(cpp_output_level);
}

void kaminpar_set_max_timer_depth(kaminpar_t *kaminpar, const int max_timer_depth) {
  kaminpar->kaminpar.set_max_timer_depth(max_timer_depth);
}

void kaminpar_copy_graph(
    kaminpar_t *kaminpar,
    const kaminpar_node_id_t n,
    const kaminpar_edge_id_t *xadj,
    const kaminpar_node_id_t *adjncy,
    const kaminpar_node_weight_t *vwgt,
    const kaminpar_edge_weight_t *adjwgt
) {
  std::span<const kaminpar::shm::NodeWeight> vwgt_span;
  if (vwgt != NULL) {
    vwgt_span = {vwgt, n};
  }

  std::span<const kaminpar::shm::EdgeWeight> adjwgt_span;
  if (adjwgt != NULL) {
    adjwgt_span = {adjwgt, xadj[n]};
  }

  kaminpar->n = n;
  kaminpar->kaminpar.copy_graph({xadj, n + 1}, {adjncy, xadj[n]}, vwgt_span, adjwgt_span);
}

void kaminpar_borrow_and_mutate_graph(
    kaminpar_t *kaminpar,
    kaminpar_node_id_t n,
    kaminpar_edge_id_t *xadj,
    kaminpar_node_id_t *adjncy,
    kaminpar_node_weight_t *vwgt,
    kaminpar_edge_weight_t *adjwgt
) {
  std::span<kaminpar::shm::NodeWeight> vwgt_span;
  if (vwgt != NULL) {
    vwgt_span = {vwgt, n};
  }

  std::span<kaminpar::shm::EdgeWeight> adjwgt_span;
  if (adjwgt != NULL) {
    adjwgt_span = {adjwgt, xadj[n]};
  }

  kaminpar->n = n;
  kaminpar->kaminpar.borrow_and_mutate_graph(
      {xadj, n + 1}, {adjncy, xadj[n]}, vwgt_span, adjwgt_span
  );
}

void kaminpar_enable_balanced_minimum_block_weights(kaminpar_t *kaminpar, const int enable) {
  kaminpar->kaminpar.enable_balanced_minimum_block_weights(enable);
}

kaminpar_edge_weight_t kaminpar_compute_partition(
    kaminpar_t *kaminpar, kaminpar_block_id_t k, kaminpar_block_id_t *partition
) {
  return kaminpar->kaminpar.compute_partition(k, {partition, kaminpar->n});
}

kaminpar_edge_weight_t kaminpar_compute_partition_with_epsilon(
    kaminpar_t *kaminpar, kaminpar_block_id_t k, double epsilon, kaminpar_block_id_t *partition
) {
  return kaminpar->kaminpar.compute_partition(k, epsilon, {partition, kaminpar->n});
}

kaminpar_edge_weight_t kaminpar_compute_partition_with_max_block_weight_factors(
    kaminpar_t *kaminpar,
    kaminpar_block_id_t k,
    const double *max_block_weight_factors,
    kaminpar_block_id_t *partition
) {
  std::vector<double> max_block_weight_factors_vec(
      max_block_weight_factors, max_block_weight_factors + k
  );
  return kaminpar->kaminpar.compute_partition(
      std::move(max_block_weight_factors_vec), {partition, kaminpar->n}
  );
}

kaminpar_edge_weight_t kaminpar_compute_partition_with_max_block_weights(
    kaminpar_t *kaminpar,
    kaminpar_block_id_t k,
    const kaminpar_block_weight_t *max_block_weights,
    kaminpar_block_id_t *partition
) {
  std::vector<kaminpar::shm::BlockWeight> max_block_weights_vec(
      max_block_weights, max_block_weights + k
  );
  return kaminpar->kaminpar.compute_partition(
      std::move(max_block_weights_vec), {partition, kaminpar->n}
  );
}
