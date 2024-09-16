1. Go through Github issues to verify bugs have been fixed and closed.
1. Verify [ManifoldCAD.org](https://manifoldcad.org) - check a few examples, run them, download a GLB and a 3MF.
1. Verify our three.js [example](https://manifoldcad.org/three) is functional.
1. Verify our model-viewer [example](https://manifoldcad.org/model-viewer) is functional - select Union and Intersection.
1. Verify [make-manifold](https://manifoldcad.org/make-manifold) is functional. Try dropping [DragonAttenuation.glb](https://github.com/KhronosGroup/glTF-Sample-Assets/blob/main/Models/DragonAttenuation/glTF-Binary/DragonAttenuation.glb) in and verify you can select "View Manifold GLB" and that the dragon is still present while the backdrop is removed. Download the GLB.
1. Make a new branch called the version, e.g. v2.3.0.
1. Use VSCode to search and replace the old version with the new - so far in test-cmake.sh, flake.nix, pyproject.toml, and package.json.
1. Also update CMakeLists.txt version by searching for "set(MANIFOLD_VERSION_".
1. Commit, push, open a PR, verify tests pass, merge.
1. On Github, draft a new release, make a new tag with the version number, add release notes, and publish.
1. Check the Actions and verify that both PyPI and npm publishing actions ran successfully.
1. Verify the npm [package](https://www.npmjs.com/package/manifold-3d?activeTab=code) looks good - unpacked size should be close to 1MB.
1. Verify PyPI [package](https://pypi.org/project/manifold3d/#files) looks good - a bunch of built distributions ranging from ~600kB to ~1.1MB.
1. If there's a problem with release deployment, the release workflows can be triggered separately, manually for any branch, under the Actions tab.