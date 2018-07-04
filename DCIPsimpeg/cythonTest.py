from cython.parallel import prange
import numpy as np
import JDataObject as Jdata

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

cdef int num_threads

openmp.omp_set_dynamic(1)
with nogil, parallel():
    num_threads = openmp.omp_get_num_threads()
################################################################
# define the file required for import
# fileName = "/Users/juan/Documents/testData/FieldSchool_2017new.DAT"

# patch = Jdata.loadDias(fileName)

# for i in prange(len(patch.readings), nogil=True):
#     patch.readings[n].Vdp[0]