import {expect, test} from 'vitest';

import type {Component3MF} from './export-3mf.ts';
import {toposort} from './export-3mf.ts';

test(
    'emits a shared component once even when a parent instantiates it repeatedly',
    () => {
      const child: Component3MF = {id: 'child', children: []};
      const parent: Component3MF = {
        id: 'parent',
        children: [{objectID: 'child'}, {objectID: 'child'}],
      };
      const input = [parent, child];
      expect(toposort(input)).toEqual([child, parent]);
      expect(input).toEqual([parent, child]);
      expect(parent.children).toHaveLength(2);
    });

test('keeps every component in a large shared dependency graph', () => {
  const input: Component3MF[] = Array.from(
      {length: 1000}, (_, i) => ({
                        id: String(i),
                        children: [i + 1, i + 2, i + 7]
                                      .filter(j => j < 1000)
                                      .map(j => ({objectID: String(j)})),
                      }));
  const sorted = toposort(input);
  expect(sorted).toHaveLength(input.length);
  const index = new Map(sorted.map((node, i) => [node.id, i]));
  for (const parent of input) {
    for (const child of parent.children)
      expect(index.get(child.objectID)!).toBeLessThan(index.get(parent.id)!);
  }
});

test('reports cycles instead of silently omitting components', () => {
  const input: Component3MF[] = [
    {id: 'first', children: [{objectID: 'second'}]},
    {id: 'second', children: [{objectID: 'first'}]},
  ];
  expect(() => toposort(input)).toThrow(/cycle/i);
});
