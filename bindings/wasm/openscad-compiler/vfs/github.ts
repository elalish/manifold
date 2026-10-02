import { unzipSync } from "fflate";
import { vfs } from "./vfs.js";

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
  const libRoot = `/openscad_libs/${libName}`;
  if (vfs.existsSync(libRoot)) return true;
  try {
    // libname is the repo name
    const { owner, repo, branch } = await findGitHubRepository(libName);

    console.log(`Found GitHub repository: ${owner}/${repo} (${branch})`);

    // download repo zip
    const url = `https://api.github.com/repos/${owner}/${repo}/zipball/${encodeURIComponent(branch)}`;

    const response = await fetch(url);

    if (!response.ok) {
      throw new Error(`Failed to download repository: ${response.status}`);
    }

    const archive = new Uint8Array(await response.arrayBuffer());

    // extract the zip
    const files = unzipSync(archive);


    // GitHub's zipball uses a generated top-level directory, which does not
    // necessarily match either the repository name or its branch.
    const rootPrefix = Object.keys(files)[0]?.replace(/\\/g, '/').split('/')[0];
    if (!rootPrefix) throw new Error('GitHub archive is empty');

    let savedFiles = 0;
    for (const [filePath, data] of Object.entries(files)) {
      const normalizedPath = filePath.replace(/\\/g, '/');
      // ignore directories
      if (normalizedPath.endsWith("/")) {
        continue;
      }

      if (!normalizedPath.startsWith(`${rootPrefix}/`)) {
        throw new Error(`Unexpected GitHub archive path: ${filePath}`);
      }
      const relativePath = normalizedPath.slice(rootPrefix.length + 1);

      if (!relativePath) {
        continue;
      }
      if (relativePath.split('/').some(
              part => !part || part === '.' || part === '..')) {
        throw new Error(`Unsafe GitHub archive path: ${filePath}`);
      }

      const vfsPath = `${libRoot}/${relativePath}`;

      // ensure parent directories exist
      const dirPath = vfsPath.split("/").slice(0, -1).join("/");
      if (dirPath) {
        vfs.mkdirSync(dirPath, {recursive: true});
      }

      vfs.writeFileSync(vfsPath, data);
      savedFiles++;
    }
    if (savedFiles === 0) throw new Error('GitHub archive has no files');

    console.log(
      `Saved ${savedFiles} files from ${repo} to ${libRoot}`
    );
    return true;
  } catch (err) {
    // An incomplete download must not block a later retry.
    vfs.rmSync(libRoot, {recursive: true, force: true});
    console.error(
      "Failed to fetch and save library",
      err
    );
    return false;
  }
}
