"""
 Copyright 2023 The Manifold Authors.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

import numpy as np
from manifold3d import Manifold, Mesh

# Generalized Winding Numbers Implementation from 
# https://github.com/marmakoide/inside-3d-mesh/blob/master/is_inside_mesh.py#L28

def is_inside_turbo(triangles, X):
    # Compute euclidean norm along axis 1
    def anorm2(X):
        return np.sqrt(np.sum(X ** 2, axis = 1))

    # Compute 3x3 determinant along axis 1
    def adet(X, Y, Z):
        ret  = np.multiply(np.multiply(X[:,0], Y[:,1]), Z[:,2])
        ret += np.multiply(np.multiply(Y[:,0], Z[:,1]), X[:,2])
        ret += np.multiply(np.multiply(Z[:,0], X[:,1]), Y[:,2])
        ret -= np.multiply(np.multiply(Z[:,0], Y[:,1]), X[:,2])
        ret -= np.multiply(np.multiply(Y[:,0], X[:,1]), Z[:,2])
        ret -= np.multiply(np.multiply(X[:,0], Z[:,1]), Y[:,2])
        return ret

    # One generalized winding number per input vertex
    ret = np.zeros(X.shape[0], dtype = X.dtype)

    # Accumulate generalized winding number for each triangle
    for U, V, W in triangles:
        A, B, C = U - X, V - X, W - X
        omega = adet(A, B, C)
        a, b, c = anorm2(A), anorm2(B), anorm2(C)
        k  = a * b * c 
        k += c * np.sum(np.multiply(A, B), axis = 1)
        k += a * np.sum(np.multiply(B, C), axis = 1)
        k += b * np.sum(np.multiply(C, A), axis = 1)
        ret += np.arctan2(omega, k)
    return ret # Job done

def mesh_winding_number_batched(queries):
    isosurface = 1.8
    tris = np.array([[[ 0.5  ,  0.5,  0.0], # Top Right Tri
                      [ 0.125,  0.0,  0.5], 
                      [ 0.5  , -0.5,  0.0]],
                     [[-0.125,  0.0,  0.5], # Top Left Tri
                      [-0.5  ,  0.5,  0.0],
                      [-0.5  , -0.5,  0.0]],
                     [[ 0.0  ,  0.5, -0.5], # Bottom Tri
                      [ 0.5  , -0.5, -0.5], 
                      [-0.5  , -0.5, -0.5]]])
    return is_inside_turbo(tris, queries) - isosurface

def run():
    levelset_mesh = Mesh.levelset_batch(mesh_winding_number_batched, 
                                        [-0.5, -0.5, -0.5, 0.5, 0.5, 0.5], 0.05, 0.0, False)
    model = Manifold.from_mesh(levelset_mesh)
    return model
