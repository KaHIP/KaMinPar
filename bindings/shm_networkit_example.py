import networkit as nk
import kaminpar as ka

G = nk.readGraph("~/Graphs/144.metis", nk.Format.METIS)
nk.overview(G)

shm = ka.KaMinPar(6)
shm.copy_graph(G)
part = shm.compute_partition(4)

print("Edge cut: ", nk.community.EdgeCut().getQuality(part, G))
