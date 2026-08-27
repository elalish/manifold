1. Go through Github issues to verify bugs have been fixed and closed.
1. Verify [ManifoldCAD.org](https://manifoldcad.org) - check a few examples, run them, download a GLB and a 3MF.
1. Verify our three.js [example](https://manifoldcad.org/three) is functional.
1. Verify our model-viewer [example](https://manifoldcad.org/model-viewer) is functional - select Union and Intersection.
1. Verify [make-manifold](https://manifoldcad.org/make-manifold) is functional. Try dropping [DragonAttenuation.glb](https://github.com/KhronosGroup/glTF-Sample-Assets/blob/main/Models/DragonAttenuation/glTF-Binary/DragonAttenuation.glb) in and verify you can select "View Manifold GLB" and that the dragon is still present while the backdrop is removed. Download the GLB.
1. Make a new branch called the version, e.g. v2.3.0.
1. Use VSCode to search and replace the old version with the new - so far in test-cmake.sh, flake.nix, pyproject.toml, and package.json.
1. Also update CMakeLists.txt version by searching for "set(MANIFOLD_VERSION_".
1. in `bindings/wasm`, run `npm run install:all` to update the package-lock files.
1. Commit, push, open a PR, verify tests pass, manually trigger build_wheels CI, merge.
1. On Github, draft a new release, make a new tag with the version number, add release notes, and publish.
1. Check the Actions and verify that both PyPI and npm publishing actions ran successfully.
1. Verify the npm [package](https://www.npmjs.com/package/manifold-3d?activeTab=code) looks good - unpacked size should be close to 1MB.
1. Verify PyPI [package](https://pypi.org/project/manifold3d/#files) looks good - a bunch of built distributions ranging from ~600kB to ~1.1MB.
1. If there's a problem with release deployment, the release workflows can be triggered separately, manually for any branch, under the Actions tab. See [Deploying manually](#deploying-manually) for the site deploy in particular.

## Where the site is published

`manifoldcad.org` is served from the `gh-pages` branch, which holds several
builds side by side:

| Path | Updated | Contents |
| --- | --- | --- |
| `/` (root) | each published release | the current release, served at manifoldcad.org |
| `alpha/` | every master commit that passes CI | unreleased build, for testing |
| `vX.Y.Z/` | once, when that release is published | a frozen snapshot of that version |

Each is a complete copy of the site: the ManifoldCAD editor, the examples,
the benchmark dashboard, and both the C++ and TypeScript docs. A release
publishes twice, once to its own `vX.Y.Z/` directory and once to the root,
so manifoldcad.org always serves the current release directly.

Subdirectory deploys erase their target before writing, so each holds exactly
one build. The root deploy cannot, because clearing it would take the
subdirectories with it, so there it overwrites in place instead.

## Deploying manually

The Deploy documentation workflow can be run by hand from the Actions tab for
a quick fix or a docs-only update, without cutting a release. It takes two
inputs:

- **ref** - the branch or tag to build from, e.g. `v3.4.0`
- **destination_dir** - the directory on `gh-pages` to publish into, e.g. `v3.4.0`

The target directory is erased and rebuilt, so the result is exactly what the
chosen ref produces. Nothing outside that directory is affected.

Keep `ref` and `destination_dir` describing the same version. Publishing
master into `v3.4.0/` would leave that directory claiming to be a release it
is not.

Typical uses:

- **Fix a published version.** Branch from its tag, commit the fix, then
  deploy that branch into the version's own directory. Useful for correcting
  documentation without a patch release.
- **Backfill an old release.** Deploy its tag into a matching directory so the
  version has a browsable copy of the site.
- **Preview a branch.** Deploy it somewhere disposable rather than over a
  directory that matters.

Builds from before the versioned layout existed assume they are served from
the site root and will load the wrong assets from a subdirectory. Deploying
such a tag needs the relative-path fix applied on top first.
