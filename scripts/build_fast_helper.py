"""Build the native fast-search helper for this platform and architecture."""
import os
from pathlib import Path
from setuptools._distutils.ccompiler import new_compiler
from setuptools._distutils.sysconfig import customize_compiler

NAME='eternity-fast'


def build(root):
    target=root/'build'/'native-helper'
    target.mkdir(parents=True,exist_ok=True)
    source=root/'scripts'/'fast_dfs.cpp'
    compiler=new_compiler(force=True)
    customize_compiler(compiler)
    msvc=compiler.compiler_type=='msvc'
    # Portable code generation only: release archives run on older processors too.
    flags=['/std:c++17','/O2','/EHsc'] if msvc else ['-std=c++17','-O3','-pthread']
    objects=compiler.compile([str(source)],output_dir=str(target/'objects'),extra_postargs=flags)
    compiler.link_executable(objects,NAME,output_dir=str(target),target_lang='c++',
                             extra_postargs=[] if msvc else ['-pthread'])
    helper=target/(NAME+'.exe' if os.name=='nt' else NAME)
    if not helper.is_file():
        raise RuntimeError('Compiler produced no native fast-search helper')
    return helper


if __name__=='__main__':
    print(build(Path(__file__).resolve().parents[1]))
