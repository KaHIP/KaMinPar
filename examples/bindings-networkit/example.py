import networkit
import kaminpar_networkit as kaminpar


graph = networkit.readGraph("misc/rgg2d.metis", networkit.Format.METIS)
networkit.overview(graph)

shm = kaminpar.KaMinPar(graph)
partition = shm.computePartitionWithFactors([0.53, 0.253, 0.253])

edge_cut = networkit.community.EdgeCut().getQuality(partition, graph)
print("Computed a partition with edge cut of ", edge_cut)
