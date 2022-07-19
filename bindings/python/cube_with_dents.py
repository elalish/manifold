from pymanifold import Manifold
from functools import reduce

# https://gist.github.com/ochafik/2db96400e3c1f73558fcede990b8a355#file-cube-with-half-spheres-dents-scad

def run(n=5, overlap=True):
    a = Manifold.cube(n, n, 0.5).translate(-0.5, -0.5, -0.5)

    spheres = [Manifold.sphere(0.45 if overlap else 0.55, 50).translate(i, j, 0)
               for i in range(n) for j in range(n)]
    spheres = reduce(lambda a, b: a + b, spheres)

    return a - spheres

