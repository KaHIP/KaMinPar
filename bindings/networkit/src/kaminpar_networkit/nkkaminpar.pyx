# cython: language_level=3
# distutils: language=c++

from libcpp.vector cimport vector
from libc.stdint cimport uint64_t, int32_t

import networkit


cdef extern from "kaminpar-shm/kaminpar.h":
    cdef cppclass _KaMinPar "kaminpar::KaMinPar":
        pass


cdef extern from "kaminpar_networkit.h":
    cdef cppclass _KaMinParNetworKit "kaminpar::KaMinParNetworKit"(_KaMinPar):
        _KaMinParNetworKit() except +
        void copyCSRGraph(vector[uint64_t], vector[uint64_t], vector[int32_t]) except +
        vector[uint64_t] computePartition(unsigned int) except +
        vector[uint64_t] computePartitionWithEpsilon(unsigned int, double) except +
        vector[uint64_t] computePartitionWithFactors(vector[double]) except +
        vector[uint64_t] computePartitionWithWeights(vector[int]) except +


cdef _vector_to_partition(vector[uint64_t] vec):
    return networkit.Partition(data=vec)


cdef _extract_csr(object G):
    """Extract CSR arrays from a networkit.Graph via its adjacency matrix."""
    from networkit.algebraic import adjacencyMatrix
    A = adjacencyMatrix(G, matrixType="sparse")

    xadj = A.indptr.tolist()
    adjncy = A.indices.tolist()

    if G.isWeighted():
        adjwgt = [int(w) for w in A.data]
    else:
        adjwgt = []

    return xadj, adjncy, adjwgt


cdef class KaMinPar:
    cdef _KaMinParNetworKit *thisptr

    def __cinit__(self, object G):
        self.thisptr = new _KaMinParNetworKit()
        xadj, adjncy, adjwgt = _extract_csr(G)
        self.thisptr.copyCSRGraph(xadj, adjncy, adjwgt)

    def __dealloc__(self):
        del self.thisptr

    def copyGraph(self, object G):
        xadj, adjncy, adjwgt = _extract_csr(G)
        self.thisptr.copyCSRGraph(xadj, adjncy, adjwgt)

    def computePartition(self, unsigned int k):
        return _vector_to_partition(self.thisptr.computePartition(k))

    def computePartitionWithEpsilon(self, unsigned int k, double epsilon):
        return _vector_to_partition(self.thisptr.computePartitionWithEpsilon(k, epsilon))

    def computePartitionWithFactors(self, list maxBlockWeightFactors):
        cdef vector[double] v
        for i in range(len(maxBlockWeightFactors)):
            v.push_back(maxBlockWeightFactors[i])
        return _vector_to_partition(self.thisptr.computePartitionWithFactors(v))

    def computePartitionWithWeights(self, list maxBlockWeights):
        cdef vector[int] v
        for i in range(len(maxBlockWeights)):
            v.push_back(maxBlockWeights[i])
        return _vector_to_partition(self.thisptr.computePartitionWithWeights(v))
