import { WebIO } from "@gltf-transform/core";
import { expect, suite, test } from "vitest";
import assert from "node:assert";
import Module from "./built/manifold";
import { readMesh, setupIO } from "./gltf-io";
import { evaluateCADToModel } from "./worker";
// @ts-ignore
import { examples } from "./public/examples.js";

const io = setupIO(new WebIO());

const wasm = await Module();
wasm.setup();

async function runExample(name: string) {
  const code = examples.functionBodies.get(name);
  const result = await evaluateCADToModel(code);
  assert.ok(result?.glbURL);
  const docIn = await io.read(result.glbURL);
  const nodes = docIn.getRoot().listNodes();
  for (const node of nodes) {
    const docMesh = node.getMesh();
    if (!docMesh) {
      continue;
    }
    const { mesh } = readMesh(docMesh)!;
    const manifold = new wasm.Manifold(mesh as any);
    const prop = manifold.getProperties();
    const genus = manifold.genus();
    return { prop, genus };
  }
  return;
}

suite("Examples", () => {
  test("Intro", async () => {
    const result = await runExample("Intro");
    expect(result).toMatchInlineSnapshot(`
      {
        "genus": 5,
        "prop": {
          "surfaceArea": 62046.0234375,
          "volume": 203163.453125,
        },
      }
    `);
  });
});
