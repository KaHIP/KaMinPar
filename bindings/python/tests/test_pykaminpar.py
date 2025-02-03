from __future__ import annotations

import pytest
import kaminpar

graphs = [
    ("misc/rgg2d.metis", kaminpar.GraphFileFormat.METIS),
]
if not kaminpar.__64bit__:
    graphs.append(("misc/rgg2d-32bit.parhip", kaminpar.GraphFileFormat.PARHIP))


def test_version():
    assert kaminpar.__version__ == "3.1.0"


def test_seed():
    assert kaminpar.seed() == 0
    kaminpar.reseed(42)
    assert kaminpar.seed() == 42


def test_context():
    context_names = kaminpar.context_names()
    assert all(
        context_name in context_names
        for context_name in [
            "default",
            "fast",
            "strong",
            "terapart",
            "terapart-strong",
            "terapart-largek",
            "largek",
            "largek-fast",
            "largek-strong",
        ]
    )

    assert type(kaminpar.context_by_name("default")) is kaminpar.Context
    with pytest.raises(ValueError):
        kaminpar.context_by_name("invalid_context_name")

    assert type(kaminpar.default_context()) is kaminpar.Context
    assert type(kaminpar.fast_context()) is kaminpar.Context
    assert type(kaminpar.strong_context()) is kaminpar.Context

    assert type(kaminpar.terapart_context()) is kaminpar.Context
    assert type(kaminpar.terapart_strong_context()) is kaminpar.Context
    assert type(kaminpar.terapart_largek_context()) is kaminpar.Context

    assert type(kaminpar.largek_context()) is kaminpar.Context
    assert type(kaminpar.largek_fast_context()) is kaminpar.Context
    assert type(kaminpar.largek_strong_context()) is kaminpar.Context


def test_load_graph():
    with pytest.raises(ValueError):
        kaminpar.load_graph("invalid.metis", kaminpar.GraphFileFormat.METIS)

    with pytest.raises(ValueError):
        kaminpar.load_graph("invalid.parhip", kaminpar.GraphFileFormat.PARHIP)

    assert (
        type(kaminpar.load_graph("misc/rgg2d.metis", kaminpar.GraphFileFormat.METIS))
        is kaminpar.Graph
    )


@pytest.mark.parametrize("filename,file_format", graphs)
def test_graph_interface(filename, file_format):
    graph = kaminpar.load_graph(filename, file_format)

    assert graph.n() == 1024
    assert graph.m() == 8226

    assert all(u == i for i, u in enumerate(graph.nodes()))
    assert all(e == i for i, e in enumerate(graph.edges()))

    assert graph.is_node_weighted() is False
    assert graph.is_edge_weighted() is False

    assert all(graph.node_weight(u) == 1 for u in graph.nodes())

    assert sum(graph.degree(u) for u in range(graph.n())) == graph.m()

    assert all(len(graph.neighbors(u)) == graph.degree(u) for u in graph.nodes())


@pytest.mark.parametrize("filename,file_format", graphs)
def test_partitioning(filename, file_format):
    ctx = kaminpar.default_context()
    instance = kaminpar.KaMinPar(num_threads=1, ctx=ctx)

    graph = kaminpar.load_graph(filename, file_format)
    partition = instance.compute_partition(graph=graph, k=4, eps=0.03)

    assert len(partition) == graph.n()
    assert all(0 <= block < 4 for block in partition)
