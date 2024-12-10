import networkit as nk
import kaminpar as ka

G = nk.readGraph("~/Graphs/144.metis", nk.Format.METIS)
nk.overview(G)

shm = ka.KaMinPar(G)
part = shm.computePartitionWithFactors([0.53, 0.253, 0.253])

print("Edge cut: ", nk.community.EdgeCut().getQuality(part, G))
