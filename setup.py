"""Copy the curated public resources into the wheel's resource package."""
from pathlib import Path
import shutil
import sys
from setuptools import setup
from setuptools.command.build_py import build_py

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from build_support import resource_files

class BuildWithResources(build_py):
    def run(self):
        super().run()
        for relative, source in resource_files(ROOT):
            destination = Path(self.build_lib) / "eternity_resources" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)

    def get_outputs(self, include_bytecode=1):
        outputs = super().get_outputs(include_bytecode)
        outputs.extend(str(Path(self.build_lib) / "eternity_resources" / relative)
                       for relative, _ in resource_files(ROOT))
        return list(dict.fromkeys(outputs))

setup(cmdclass={"build_py": BuildWithResources})
