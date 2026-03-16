# cython: language_level=3
# distutils: language=c++

from libcpp.vector cimport vector
from libc.stdint cimport uint64_t

from networkit.graph cimport _Graph, Graph

import networkit

cdef extern from "kaminpar-shm/kaminpar.h":
    cdef cppclass _KaMinPar "kaminpar::KaMinPar":
        pass


cdef extern from "kaminpar_networkit.h":
    cdef cppclass _KaMinParNetworKit "kaminpar::KaMinParNetworKit"(_KaMinPar):
        _KaMinParNetworKit(const _Graph &)
        void copyGraph(const _Graph &) except +
        vector[uint64_t] computePartition(unsigned int) except +
        vector[uint64_t] computePartitionWithEpsilon(unsigned int, double) except +
        vector[uint64_t] computePartitionWithFactors(vector[double]) except +
        vector[uint64_t] computePartitionWithWeights(vector[int]) except +


cdef _vector_to_partition(vector[uint64_t] vec):
    return networkit.Partition(data=vec)


cdef class KaMinPar:
    cdef _KaMinParNetworKit *thisptr

    def __cinit__(self, Graph G):
        self.thisptr = new _KaMinParNetworKit(G._this)

    def __dealloc__(self):
        del self.thisptr

    def copyGraph(self, Graph G):
        self.thisptr.copyGraph(G._this)

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
