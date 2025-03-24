import networkit as nk
import kaminpar_networkit as kaminpar
from networkit import vizbridges

graph = nk.generators.HyperbolicGenerator(100, k = 16, gamma = 2.7, T = 0).generate()
partition = kaminpar.KaMinPar(graph).computePartitionWithEpsilon(4, 0.03)

fig = nk.vizbridges.widgetFromGraph(graph, dimension = nk.vizbridges.Dimension.Three, nodePartition = partition, nodePalette = [(0, 0, 0), (255, 0, 0)])
fig.write_image("partitioned_hyperbolic.png")


