# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from os import environ, path as osp
from setuptools import setup
from distutils.extension import Extension

import numpy as np

from Cython.Build import cythonize
from Cython.Distutils import build_ext


def find_in_path(name, paths):
    """Find a file in a search path"""
    # Adapted from
    # http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for path in paths.split(osp.sep):
        path_bin = osp.join(path, name)
        if osp.exists(path_bin):
            return osp.abspath(path_bin)
    return None


def locate_cuda(cudaconfig):
    """Locate the CUDA environment on the system

    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.

    Starts by looking for the CUDAHOME env variable. If not found, everything
    is based on finding 'nvcc' in the PATH.
    """

    # first check if the CUDAHOME env variable is in use
    default_path = osp.join(osp.sep, 'usr', 'local', 'cuda', 'bin')
    if 'CUDAHOME' in environ:
        home = environ['CUDAHOME']
        nvcc = osp.join(home, 'bin', 'nvcc')
    elif osp.isdir(default_path):
        # otherwise, search the PATH for NVCC
        nvcc = find_in_path('nvcc', environ['PATH'] + osp.sep + default_path)
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                                   'located in your $PATH. Either add it to '
                                   'your path, or set $CUDAHOME')
        home = osp.dirname(osp.dirname(nvcc))
    else:
        home = ''
        nvcc = ''

    if cudaconfig['home']:
        cudaconfig.update({
            'home': home,
            'nvcc': nvcc,
            'include': osp.join(home, 'include'),
            'lib64': osp.join(home, 'lib64')
        })
        for k, v in cudaconfig.items():
            if not osp.exists(v):
                raise EnvironmentError(f"The CUDA {k} path could not be located"
                                       f" in {v}")

    return cudaconfig


CUDA = {
    'home': '',
    'nvcc': '',
    'include': '',
    'lib64': ''
}

locate_cuda(CUDA)

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()


def customize_compiler_for_nvcc(self):
    """inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.

    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on."""

    # tell the compiler it can process .cu
    if CUDA['home']:
        self.src_extensions.append('.cu')

    # save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    _compile_default = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if osp.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            self.set_executable('compiler_so', CUDA['nvcc'])
            # use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']

        _compile_default(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile


# run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        if CUDA['home']:
            print(f"Building with CUDA: {CUDA['home']}")
        else:
            print(f"Building for CPU only")
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)


print(f"DEBUG: CUDA = {CUDA}")

ext_modules = [
    Extension(
        "utils.cython_bbox",
        ["utils/bbox.pyx"],
        extra_compile_args={'gcc': ["-Wno-cpp", "-Wno-unused-function"]},
        include_dirs=[numpy_include]
    ),
    Extension(
        "nms.cpu_nms",
        ["nms/cpu_nms.pyx"],
        extra_compile_args={'gcc': ["-Wno-cpp", "-Wno-unused-function"]},
        include_dirs=[numpy_include]
    ) if not CUDA['home'] else
    Extension(
        'nms.gpu_nms',
        ['nms/nms_kernel.cu', 'nms/gpu_nms.pyx'],
        library_dirs=[CUDA['lib64']],
        libraries=['cudart'],
        language='c++',
        runtime_library_dirs=[CUDA['lib64']],
        # this syntax is specific to this build system
        # we're only going to use certain compiler args with nvcc and not with
        # gcc the implementation of this trick is in customize_compiler() below
        extra_compile_args={
            'gcc': ["-Wno-unused-function"],
            'nvcc': [
                '-arch=sm_35',
                '--ptxas-options=-v',
                '-c',
                '--compiler-options',
                "'-fPIC'"
            ]
        },
        include_dirs=[numpy_include, CUDA['include']]
    ),
    Extension(
        'pycocotools._mask',
        sources=['pycocotools/maskApi.c', 'pycocotools/_mask.pyx'],
        include_dirs=[numpy_include, 'pycocotools'],
        extra_compile_args={
            'gcc': ['-Wno-cpp', '-Wno-unused-function', '-std=c99']
        },
    ),
]

# print(f"DEBUG: ext modules = {ext_modules}")

setup(
    name='fast_rcnn',
    ext_modules=ext_modules,
    # inject our custom trigger
    cmdclass={'build_ext': custom_build_ext},
)
