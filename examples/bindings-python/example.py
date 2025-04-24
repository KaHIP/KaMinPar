import kaminpar

# Initialize the KaMinPar algorithm with one thread and the default settings.
instance = kaminpar.KaMinPar(num_threads=1, ctx=kaminpar.default_context())

# Load a graph stored in METIS format from disk, which can be optionally compressed during IO.
graph = kaminpar.load_graph(
    "misc/rgg2d.metis", kaminpar.GraphFileFormat.METIS, compress=False
)

# Partition the graph into four blocks using imbalance factor 0.03
partition = instance.compute_partition(graph, k=4, eps=0.03)
edge_cut = kaminpar.edge_cut(graph, partition)
print("Computed a partition with an edge cut of", edge_cut)
