# cython: language_level=3
# distutils: language=c++

from libcpp.vector cimport vector

from networkit.graph cimport _Graph, Graph
from networkit.structures cimport _Partition, Partition

cdef extern from "kaminpar-shm/kaminpar.h":
    cdef cppclass _KaMinPar "kaminpar::KaMinPar":
        pass


cdef extern from "kaminpar_networkit.h":
    cdef cppclass _KaMinParNetworKit "kaminpar::KaMinParNetworKit"(_KaMinPar):
        _KaMinParNetworKit(const _Graph &)
        void copyGraph(const _Graph &) except +
        _Partition computePartition(unsigned int) except +
        _Partition computePartitionWithEpsilon(unsigned int, double) except +
        _Partition computePartitionWithFactors(vector[double]) except +
        _Partition computePartitionWithWeights(vector[int]) except +


cdef class KaMinPar:
    cdef _KaMinParNetworKit *thisptr

    def __cinit__(self, Graph G):
        self.thisptr = new _KaMinParNetworKit(G._this)

    def __dealloc__(self):
        del self.thisptr

    def copyGraph(self, Graph G):
        self.thisptr.copyGraph(G._this)

    def computePartition(self, unsigned int k):
        return Partition().setThis(self.thisptr.computePartition(k))

    def computePartitionWithEpsilon(self, unsigned int k, double epsilon):
        return Partition().setThis(self.thisptr.computePartitionWithEpsilon(k, epsilon))

    def computePartitionWithFactors(self, list maxBlockWeightFactors):
        cdef vector[double] v
        for i in range(len(maxBlockWeightFactors)):
            v.push_back(maxBlockWeightFactors[i])
        return Partition().setThis(self.thisptr.computePartitionWithFactors(v))

    def computePartitionWithWeights(self, list maxBlockWeights):
        cdef vector[int] v
        for i in range(len(maxBlockWeights)):
            v.push_back(maxBlockWeights[i])
        return Partition().setThis(self.thisptr.computePartitionWithWeights(v))

