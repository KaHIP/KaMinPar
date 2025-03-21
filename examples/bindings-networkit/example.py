import kaminpar_networkit as kaminpar
import networkit

# Use NetworKit to read a graph from disk stored in METIS format and output statistics for the graph.
graph = networkit.readGraph("misc/rgg2d.metis", networkit.Format.METIS)
networkit.overview(graph)

# Create a KaMinPar instance and compute a partition of the graph into four blocks using imbalance factor 3%
shm = kaminpar.KaMinPar(graph)
partition = shm.computePartitionWithEpsilon(4, 0.03)

edge_cut = networkit.community.EdgeCut().getQuality(partition, graph)
print("Computed a partition with an edge cut of", edge_cut)
