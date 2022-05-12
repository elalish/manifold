import pathlib
import sys
import importlib
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
            model.export(f'{f}.glb')
            print(f'Exported model to {f}.glb')
        t1 = time()
        print(f'Took {(t1-t0)*1000:.1f}ms for {f}')
