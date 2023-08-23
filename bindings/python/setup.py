from skbuild import setup

setup(
    name="manifold3d",
    version="2.2.0",
    description=" Geometry library for topological robustness",
    author="Emmett Lalish",
    packages=["manifold3d"],
    cmake_install_dir="manifold3d",
    cmake_source_dir="../../",
    zip_safe=True,
    cmake_args=["-DMANIFOLD_PAR=TBB"],
)
