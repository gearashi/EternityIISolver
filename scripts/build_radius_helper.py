"""Build the native bounded-repair helper for this platform and architecture."""
import os
from pathlib import Path
from setuptools._distutils.ccompiler import new_compiler
from setuptools._distutils.sysconfig import customize_compiler


def build(root):
    target=root/'build'/'native-helper'
    target.mkdir(parents=True,exist_ok=True)
    source=root/'scripts'/'repair_dual_radius.cpp'
    compiler=new_compiler(force=True)
    customize_compiler(compiler)
    flags=['/std:c++17','/O2','/EHsc'] if compiler.compiler_type=='msvc' else ['-std=c++17','-O2']
    objects=compiler.compile([str(source)],output_dir=str(target/'objects'),extra_postargs=flags)
    compiler.link_executable(objects,'eternity-radius',output_dir=str(target),target_lang='c++')
    helper=target/('eternity-radius.exe' if os.name=='nt' else 'eternity-radius')
    if not helper.is_file():
        raise RuntimeError('Compiler produced no native radius helper')
    return helper


if __name__=='__main__':
    print(build(Path(__file__).resolve().parents[1]))
