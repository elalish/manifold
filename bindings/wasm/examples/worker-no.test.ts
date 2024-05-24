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
    return { ...prop, genus };
  }
  assert.ok(false);
}

suite("Examples", () => {
  test.only('Intro', async () => {
    const result = await runExample('Intro');
    expect(result.genus).to.equal(5, 'Genus');
    expect(result.volume).to.be.closeTo(203164, 1, 'Volume');
    expect(result.surfaceArea).to.be.closeTo(62046, 1, 'Surface Area');
  });

  test('Tetrahedron Puzzle', async () => {
    const result = await runExample('Tetrahedron Puzzle');
    expect(result.genus).to.equal(0, 'Genus');
    expect(result.volume).to.be.closeTo(7240, 1, 'Volume');
    expect(result.surfaceArea).to.be.closeTo(3235, 1, 'Surface Area');
  });

  test('Rounded Frame', async () => {
    const result = await runExample('Rounded Frame');
    expect(result.genus).to.equal(5, 'Genus');
    expect(result.volume).to.be.closeTo(270807, 10, 'Volume');
    expect(result.surfaceArea).to.be.closeTo(74599, 1, 'Surface Area');
  });

  test('Heart', async () => {
    const result = await runExample('Heart');
    expect(result.genus).to.equal(0, 'Genus');
    expect(result.volume).to.be.closeTo(3.342, 0.001, 'Volume');
    expect(result.surfaceArea).to.be.closeTo(11.51, 0.01, 'Surface Area');
  });

  test('Scallop', async () => {
    const result = await runExample('Scallop');
    expect(result.genus).to.equal(0, 'Genus');
    expect(result.volume).to.be.closeTo(41400, 100, 'Volume');
    expect(result.surfaceArea).to.be.closeTo(7770, 10, 'Surface Area');
  });

  test('Torus Knot', async () => {
    const result = await runExample('Torus Knot');
    expect(result.genus).to.equal(1, 'Genus');
    expect(result.volume).to.be.closeTo(20786, 1, 'Volume');
    expect(result.surfaceArea).to.be.closeTo(11177, 1, 'Surface Area');
  });

  test('Menger Sponge', async () => {
    const result = await runExample('Menger Sponge');
    expect(result.genus).to.equal(729, 'Genus');
    expect(result.volume).to.be.closeTo(203222, 10, 'Volume');
    expect(result.surfaceArea).to.be.closeTo(130475, 10, 'Surface Area');
  });

  test('Stretchy Bracelet', async () => {
    const result = await runExample('Stretchy Bracelet');
    expect(result.genus).to.equal(1, 'Genus');
    expect(result.volume).to.be.closeTo(3992, 1, 'Volume');
    expect(result.surfaceArea).to.be.closeTo(22267, 1, 'Surface Area');
  });

  test('Gyroid Module', async () => {
    const result = await runExample('Gyroid Module');
    expect(result.genus).to.equal(15, 'Genus');
    expect(result.volume).to.be.closeTo(4167, 1, 'Volume');
    expect(result.surfaceArea).to.be.closeTo(5642, 1, 'Surface Area');
  });
});
