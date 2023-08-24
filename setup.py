from skbuild import setup

setup(
    name="manifold3d",
    version="2.2.0",
    description=" Geometry library for topological robustness",
    author="Emmett Lalish",
    packages=["manifold3d"],
    package_dir={"": "bindings/python"},
    cmake_install_dir="bindings/python/manifold3d",
    zip_safe=True,
    cmake_args=["-DMANIFOLD_PAR=TBB", "-DMANIFOLD_TEST=OFF"],
)
