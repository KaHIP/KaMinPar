/*******************************************************************************
 * Public C library interface of KaMinPar.
 *
 * @file:   ckaminpar.h
 * @author: Daniel Seemaier
 * @date:   13.03.2023
 ******************************************************************************/
#ifndef CKAMINPAR_H
#define CKAMINPAR_H

#include <stdbool.h>
#include <stdint.h>

#define CKAMINPAR_VERSION_MAJOR 3
#define CKAMINPAR_VERSION_MINOR 4
#define CKAMINPAR_VERSION_PATCH 1

typedef enum {
  KAMINPAR_OUTPUT_LEVEL_QUIET = 0,
  KAMINPAR_OUTPUT_LEVEL_PROGRESS = 1,
  KAMINPAR_OUTPUT_LEVEL_APPLICATION = 2,
  KAMINPAR_OUTPUT_LEVEL_EXPERIMENT = 3,
  KAMINPAR_OUTPUT_LEVEL_DEBUG = 4,
} kaminpar_output_level_t;

#ifdef KAMINPAR_64BIT_NODE_IDS
typedef uint64_t kaminpar_node_id_t;
#else  // KAMINPAR_64BIT_NODE_IDS
typedef uint32_t kaminpar_node_id_t;
#endif // KAMINPAR_64BIT_NODE_IDS

#ifdef KAMINPAR_64BIT_EDGE_IDS
typedef uint64_t kaminpar_edge_id_t;
#else  // KAMINPAR_64BIT_EDGE_IDS
typedef uint32_t kaminpar_edge_id_t;
#endif // KAMINPAR_64BIT_EDGE_IDS

#ifdef KAMINPAR_64BIT_WEIGHTS
typedef int64_t kaminpar_node_weight_t;
typedef int64_t kaminpar_edge_weight_t;
typedef uint64_t kaminpar_unsigned_edge_weight_t;
typedef uint64_t kaminpar_unsigned_node_weight_t;
#else  // KAMINPAR_64BIT_WEIGHTS
typedef int32_t kaminpar_node_weight_t;
typedef int32_t kaminpar_edge_weight_t;
typedef uint32_t kaminpar_unsigned_edge_weight_t;
typedef uint32_t kaminpar_unsigned_node_weight_t;
#endif // KAMINPAR_64BIT_WEIGHTS

typedef uint32_t kaminpar_block_id_t;
typedef kaminpar_node_weight_t kaminpar_block_weight_t;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef struct kaminpar_context_t kaminpar_context_t;
typedef struct kaminpar_t kaminpar_t;

kaminpar_context_t *kaminpar_create_context_by_preset_name(const char *name);
kaminpar_context_t *kaminpar_create_default_context();
kaminpar_context_t *kaminpar_create_strong_context();
kaminpar_context_t *kaminpar_create_terapart_context();
kaminpar_context_t *kaminpar_create_largek_context();
kaminpar_context_t *kaminpar_create_vcycle_context(bool restrict_refinement);
void kaminpar_context_free(kaminpar_context_t *ctx);

kaminpar_t *kaminpar_create(int num_threads, kaminpar_context_t *ctx);
void kaminpar_free(kaminpar_t *kaminpar);

void kaminpar_set_output_level(kaminpar_t *kaminpar, kaminpar_output_level_t output_level);
void kaminpar_set_max_timer_depth(kaminpar_t *kaminpar, int max_timer_depth);

void kaminpar_copy_graph(
    kaminpar_t *kaminpar,
    kaminpar_node_id_t n,
    const kaminpar_edge_id_t *xadj,
    const kaminpar_node_id_t *adjncy,
    const kaminpar_node_weight_t *vwgt,
    const kaminpar_edge_weight_t *adjwgt
);

void kaminpar_borrow_and_mutate_graph(
    kaminpar_t *kaminpar,
    kaminpar_node_id_t n,
    kaminpar_edge_id_t *xadj,
    kaminpar_node_id_t *adjncy,
    kaminpar_node_weight_t *vwgt,
    kaminpar_edge_weight_t *adjwgt
);

void kaminpar_set_k(kaminpar_t *kaminpar, kaminpar_block_id_t k);

void kaminpar_set_uniform_max_block_weights(kaminpar_t *kaminpar, double epsilon);
void kaminpar_set_absolute_max_block_weights(
    kaminpar_t *kaminpar, size_t k, const kaminpar_block_weight_t *absolute_max_block_weights
);
void kaminpar_set_relative_max_block_weights(
    kaminpar_t *kaminpar, size_t k, const double *relative_max_block_weights
);
void kaminpar_clear_max_block_weights(kaminpar_t *kaminpar);

void kaminpar_set_uniform_min_block_weights(kaminpar_t *kaminpar, double min_epsilon);
void kaminpar_set_absolute_min_block_weights(
    kaminpar_t *kaminpar, size_t k, const kaminpar_block_weight_t *absolute_min_block_weights
);
void kaminpar_set_relative_min_block_weights(
    kaminpar_t *kaminpar, size_t k, const double *relative_min_block_weights
);
void kaminpar_clear_min_block_weights(kaminpar_t *kaminpar);

kaminpar_edge_weight_t
kaminpar_compute_partition(kaminpar_t *kaminpar, kaminpar_block_id_t *partition);

kaminpar_edge_weight_t kaminpar_compute_partition_with_epsilon(
    kaminpar_t *kaminpar, kaminpar_block_id_t k, double epsilon, kaminpar_block_id_t *partition
);

kaminpar_edge_weight_t kaminpar_compute_partition_with_max_block_weight_factors(
    kaminpar_t *kaminpar,
    kaminpar_block_id_t k,
    const double *max_block_weight_factors,
    kaminpar_block_id_t *partition
);

kaminpar_edge_weight_t kaminpar_compute_partition_with_max_block_weights(
    kaminpar_t *kaminpar,
    kaminpar_block_id_t k,
    const kaminpar_block_weight_t *max_block_weights,
    kaminpar_block_id_t *partition
);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif
