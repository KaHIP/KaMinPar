#include <kaminpar-shm/ckaminpar.h>
#include <stdio.h>
#include <stdlib.h>

void print_node_id(const kaminpar_node_id_t id, const kaminpar_block_id_t *partition) {
  const char *colors[] = {"\033[0;31m", "\033[0;32m", "\033[0;33m", "\033[0;34m"};
  const char *reset = "\033[0m";

  if (partition != NULL) {
    printf("%s%02d%s", colors[partition[id]], id, reset);
  } else {
    printf("%02d", id);
  }
}

void render_graph(const unsigned int *partition) {
  printf("   ");
  print_node_id(0, partition);
  printf("        ");
  print_node_id(4, partition);
  printf("\n");

  printf("  /  \\      /  \\\n");

  print_node_id(3, partition);
  printf("    ");
  print_node_id(1, partition);
  printf("--");
  print_node_id(7, partition);
  printf("    ");
  print_node_id(5, partition);
  printf("\n");

  printf("  \\  /      \\  /\n");
  printf("   ");
  print_node_id(2, partition);
  printf("        ");
  print_node_id(6, partition);
  printf("\n");

  printf("   ||        ||\n");
  printf("   ");
  print_node_id(12, partition);
  printf("        ");
  print_node_id(8, partition);
  printf("\n");

  printf("  /  \\      /  \\\n");
  print_node_id(15, partition);
  printf("    ");
  print_node_id(13, partition);
  printf("--");
  print_node_id(11, partition);
  printf("    ");
  print_node_id(9, partition);
  printf("\n");

  printf("  \\  /      \\  /\n");
  printf("   ");
  print_node_id(14, partition);
  printf("        ");
  print_node_id(10, partition);
  printf("\n");
}

int main() {
  //    00        04
  //   /  \      /  \
  // 03    01--07    05
  //   \  /      \  /
  //    02        06
  //    ||        ||
  //    12        08
  //   /  \      /  \
  // 15    13--11    09
  //   \  /      \  /
  //    14        10
  const kaminpar_node_id_t n = 16;
  const kaminpar_edge_id_t xadj[] = {
      0, 2, 5, 8, 10, 12, 14, 17, 20, 23, 25, 27, 30, 33, 36, 38, 40
  };
  const kaminpar_node_id_t adjncy[] = {
      1, 3, 0,  2, 7,  1, 12, 3, 0,  2,  5, 7,  4,  6,  5,  7,  8,  1,  4,  6,
      6, 9, 11, 8, 10, 9, 11, 8, 10, 13, 2, 13, 15, 11, 12, 14, 13, 15, 12, 14,
  };

  kaminpar_context_t *ctx = kaminpar_create_default_context();
  kaminpar_t *shm = kaminpar_create(4, ctx);

  kaminpar_set_output_level(shm, KAMINPAR_OUTPUT_LEVEL_QUIET);
  kaminpar_copy_graph(shm, n, xadj, adjncy, NULL, NULL);

  kaminpar_block_id_t *partition = (kaminpar_block_id_t *)malloc(n * sizeof(kaminpar_block_id_t));
  kaminpar_edge_weight_t cut = 0;

  cut = kaminpar_compute_partition(shm, 2, partition);
  printf("Balanced 2-way partition: %lld edges cut\n", (long long)cut);
  render_graph(partition);
  printf("\n");

  cut = kaminpar_compute_partition(shm, 4, partition);
  printf("Balanced 4-way partition: %lld edges cut\n", (long long)cut);
  render_graph(partition);
  printf("\n");

  double max_block_weight_factors[] = {0.5, 0.25, 0.25};
  cut = kaminpar_compute_partition_with_max_block_weight_factors(
      shm, 3, max_block_weight_factors, partition
  );
  printf("Relative max block weights {0.5, 0.25, 0.25} + 0.01: %lld edges cut\n", (long long)cut);
  render_graph(partition);
  printf("\n");

  kaminpar_block_weight_t max_block_weights[] = {2, 3, 3, 10};
  cut = kaminpar_compute_partition_with_max_block_weights(shm, 4, max_block_weights, partition);
  printf("Absolute max block weights {2, 3, 3, 10}: %lld edges cut\n", (long long)cut);
  render_graph(partition);
  printf("\n");

  kaminpar_free(shm);
  kaminpar_context_free(ctx);
}
