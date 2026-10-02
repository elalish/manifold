import { vfs } from "./vfs.js";

interface GitHubTreeEntry {
  path: string;
  type: string;
}

const MAX_PARALLEL_DOWNLOADS = 10;

async function findGitHubRepository(repoName: string) {
  const res = await fetch(
    `https://api.github.com/search/repositories?q=${encodeURIComponent(repoName)}+language:OpenSCAD&sort=stars`,
    { headers: { 'Accept': 'application/vnd.github.v3+json' } }
  );
  if (!res.ok) throw new Error(`GitHub API search failed: ${res.status}`);
  const data = await res.json();

  if (!data.items?.length) {
    throw new Error(`Repository not found on GitHub: ${repoName}`);
  }

  const best = data.items.find(
    (item: {name: string}) => item.name.toLowerCase() === repoName.toLowerCase()
  );
  if (!best) {
    throw new Error(`No GitHub repository named ${repoName} was found`);
  }
  return {
    owner: best.owner.login,
    repo: best.name,
    branch: best.default_branch,
  };
}


export async function fetchAndSaveLibrary(libName: string): Promise<boolean> {
  if (!/^[A-Za-z0-9_.-]+$/.test(libName) || libName === '.' || libName === '..') {
    throw new Error(`Invalid library name: ${libName}`);
  }
  const libRoot = `/openscad_libs/${libName}`;
  if (vfs.existsSync(libRoot)) return true;
  try {
    const { owner, repo, branch } = await findGitHubRepository(libName);
    console.log(`Found GitHub repository: ${owner}/${repo} (${branch})`);
    // The archive redirects to codeload.github.com, which blocks browser requests - Use the tree API and raw file host instead
    const treeUrl = `https://api.github.com/repos/${encodeURIComponent(owner)}/${encodeURIComponent(repo)}/git/trees/${encodeURIComponent(branch)}?recursive=1`;
    const treeResponse = await fetch(treeUrl);
    if (!treeResponse.ok) {
      throw new Error(`Failed to list ${owner}/${repo}: ${treeResponse.status}`);
    }
    const tree: {tree?: GitHubTreeEntry[]; truncated?: boolean} =
      await treeResponse.json();
    if (tree.truncated) {
      throw new Error(`GitHub returned an incomplete file list for ${owner}/${repo}`);
    }
    const files = tree.tree?.filter(entry => entry.type === 'blob') ?? [];
    if (files.length === 0) {
      throw new Error(`GitHub repository ${owner}/${repo} has no files`);
    }

    for (const file of files) {
      if (!file.path || file.path.includes('\\') || file.path.split('/').some(
            part => !part || part === '.' || part === '..')) {
        throw new Error(`Unsafe GitHub repository path: ${file.path}`);
      }
    }

    let nextFile = 0;
    let failure: Error|undefined;
    const download = async () => {
      while (!failure && nextFile < files.length) {
        const file = files[nextFile++]!;
        try {
          const rawPath = file.path.split('/').map(encodeURIComponent).join('/');
          const rawUrl = `https://raw.githubusercontent.com/${encodeURIComponent(owner)}/${encodeURIComponent(repo)}/${encodeURIComponent(branch)}/${rawPath}`;
          const response = await fetch(rawUrl);
          if (!response.ok) {
            throw new Error(`Failed to download ${file.path}: ${response.status}`);
          }
          const vfsPath = `${libRoot}/${file.path}`;
          vfs.mkdirSync(vfsPath.slice(0, vfsPath.lastIndexOf('/')),
                        {recursive: true});
          vfs.writeFileSync(vfsPath, new Uint8Array(await response.arrayBuffer()));
        } catch (error) {
          failure = error instanceof Error ? error : new Error(String(error));
        }
      }
    };
    await Promise.all(Array.from({length: Math.min(MAX_PARALLEL_DOWNLOADS, files.length)},
                                 () => download()));
    if (failure) throw failure;

    console.log(`Saved ${files.length} files from ${repo} to ${libRoot}`);
    return true;
  } catch (err) {
    // An incomplete download must not block a later retry.
    vfs.rmSync(libRoot, {recursive: true, force: true});
    throw err;
  }
}
