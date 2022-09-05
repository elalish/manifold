"""
 Copyright 2022 The Manifold Authors.

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

import pathlib
import sys
import importlib
import meshio
from time import time

if __name__ == "__main__":
    current_file = pathlib.Path(__file__)
    current_dir = current_file.parent
    files = [f.parts[-1][:-3]
             for f in current_dir.glob('*.py') if f != current_file]

    export_models = len(sys.argv) == 2 and sys.argv[-1] == '-e'

    for f in files:
        module = importlib.import_module(f)
        t0 = time()
        model = module.run()
        if export_models:
            mesh = model.to_mesh()
            cells = [("triangle", mesh.tri_verts)]
            meshio.write_points_cells(f'{f}.ply', mesh.vert_pos, cells)
            print(f'Exported model to {f}.ply')
        t1 = time()
        print(f'Took {(t1-t0)*1000:.1f}ms for {f}')
