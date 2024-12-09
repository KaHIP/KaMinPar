from setuptools import setup, Extension
from Cython.Build import cythonize

ext = Extension(
        "kaminpar",
        sources=["kaminpar.pyx"],
        extra_compile_args=["-std=c++20"],
        language="c++",
        include_dirs=["../"],
        library_dirs=["../build/kaminpar-shm/", "../build/bindings/"],
        libraries=["kaminpar_shm", "kaminpar_networkit", "networkit", "tbb", "tbbmalloc"],
        packages=['networkit'],
)

setup(
        name="kaminpar",
        ext_modules=cythonize(ext),
)
