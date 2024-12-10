from libcpp.vector cimport vector

from cython.operator import dereference, preincrement

from networkit.graph cimport _Graph, Graph
from networkit.structures cimport _Partition, Partition

cdef extern from "kaminpar-shm/kaminpar.h":
    cdef cppclass _KaMinPar "kaminpar::KaMinPar":
        pass

cdef extern from "bindings/kaminpar_networkit.h":
    cdef cppclass _KaMinParNetworKit "kaminpar::KaMinParNetworKit"(_KaMinPar):
        _KaMinParNetworKit(int)
        void copy_graph(_Graph&) except +
        _Partition compute_partition(unsigned int) except +
        _Partition compute_partition(unsigned int, double) except +
        _Partition compute_partition(vector[double]) except +
        _Partition compute_partition(vector[int]) except +

cdef class KaMinPar:
    cdef _KaMinParNetworKit *thisptr

    def __cinit__(self, int seed):
        self.thisptr = new _KaMinParNetworKit(seed)

    def __dealloc__(self):
        del self.thisptr

    def copy_graph(self, Graph G):
        self.thisptr.copy_graph(G._this)

    def compute_partition(self, unsigned int k):
        return Partition().setThis(self.thisptr.compute_partition(k))

    def compute_partition_with_epsilon(self, unsigned int k, double epsilon):
        return Partition().setThis(self.thisptr.compute_partition(k, epsilon))

    def compute_partition_with_factors(self, list max_block_weight_factors):
        cdef vector[double] v
        for i in range(len(max_block_weight_factors)):
            v.push_back(max_block_weight_factors[i])
        return Partition().setThis(self.thisptr.compute_partition(v))

    def compute_partition_with_weights(self, list max_block_sizes):
        cdef vector[int] v
        for i in range(len(max_block_sizes)):
            v.push_back(max_block_sizes[i])
        return Partition().setThis(self.thisptr.compute_partition(v))
