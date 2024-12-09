from cython.operator import dereference, preincrement

from networkit.graph cimport _Graph, Graph
from networkit.structures cimport _Partition, Partition

cdef extern from "kaminpar-shm/kaminpar.h":
    cdef cppclass _KaMinPar "kaminpar::KaMinPar":
        pass

cdef extern from "bindings/kaminpar_networkit.h":
    cdef cppclass _KaMinParNetworKit "kaminpar::KaMinParNetworKit"(_KaMinPar):
        _KaMinParNetworKit(int)
        void copy_graph(_Graph&)
        _Partition compute_partition(unsigned int) except +

cdef class KaMinParNetworKit:
    cdef _KaMinParNetworKit *thisptr

    def __cinit__(self, int seed):
        self.thisptr = new _KaMinParNetworKit(seed)

    def __dealloc__(self):
        del self.thisptr

    def copy_graph(self, Graph G):
        self.thisptr.copy_graph(G._this)

    def compute_partition(self, unsigned int k):
        return Partition().setThis(self.thisptr.compute_partition(k))
