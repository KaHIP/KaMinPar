import kaminpar


def edge_cut(graph, partition):
    cut = 0

    for u in graph.nodes():
        for v, w in graph.neighbors(u):
            if partition[u] != partition[v]:
                cut += w

    return cut // 2


kaminpar.reseed(42)

graph = kaminpar.load_graph("misc/rgg2d.metis", kaminpar.GraphFileFormat.METIS)
instance = kaminpar.KaMinPar(num_threads=1, ctx=kaminpar.default_context())

partition = instance.compute_partition(graph, k=4)
print("Computed a partition with edge cut of", edge_cut(graph, partition))
