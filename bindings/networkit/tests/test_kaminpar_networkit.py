import networkit as nk
import pytest

import kaminpar_networkit as ka


@pytest.mark.parametrize(
    "filename,file_format",
    [
        ("misc/rgg2d.metis", nk.Format.METIS),
    ],
)
def test_partitioning(filename, file_format):
    G = nk.readGraph("misc/rgg2d.metis", nk.Format.METIS)
    nk.overview(G)

    shm = ka.KaMinPar(G)
    part = shm.computePartitionWithFactors([0.53, 0.253, 0.253])

    nk.community.EdgeCut().getQuality(part, G)
