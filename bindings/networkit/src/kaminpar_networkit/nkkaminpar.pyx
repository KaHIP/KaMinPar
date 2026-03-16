# cython: language_level=3
# distutils: language=c++

from libcpp cimport bool as cbool
from libcpp.vector cimport vector
from libc.stdint cimport uint64_t

import networkit

# Declare NetworKit::Graph directly to avoid cimporting from networkit,
# which would transitively pull in Partition and cause binary incompatibility
# when the build-time and run-time struct layouts differ.
cdef extern from "<networkit/graph/Graph.hpp>" namespace "NetworKit":
    cdef cppclass _Graph "NetworKit::Graph":
        _Graph() except +
        _Graph(unsigned long, cbool, cbool) except +


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


cdef _Graph* _get_graph_ptr(object G):
    """Extract the C++ Graph pointer from a networkit.Graph Python object."""
    return <_Graph*><size_t>G._this


cdef class KaMinPar:
    cdef _KaMinParNetworKit *thisptr

    def __cinit__(self, object G):
        self.thisptr = new _KaMinParNetworKit(_get_graph_ptr(G)[0])

    def __dealloc__(self):
        del self.thisptr

    def copyGraph(self, object G):
        self.thisptr.copyGraph(_get_graph_ptr(G)[0])

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
