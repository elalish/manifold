import {beforeEach, expect, test, vi} from 'vitest';

const factory = vi.hoisted(() => vi.fn());
vi.mock('manifold-3d', () => ({default: factory}));

beforeEach(() => {
  vi.resetModules();
  factory.mockReset();
});

test(
    'shares one pending module initialization among concurrent callers',
    async () => {
      const created: Array<{setup: ReturnType<typeof vi.fn>}> = [];
      factory.mockImplementation(async () => {
        const module = {setup: vi.fn()};
        created.push(module);
        return module;
      });
      const {getManifoldModule, getManifoldModuleSync} =
          await import('./wasm.ts');
      const modules = await Promise.all([
        getManifoldModule(),
        getManifoldModule(),
        getManifoldModule(),
      ]);
      expect(factory).toHaveBeenCalledTimes(1);
      expect(created[0].setup).toHaveBeenCalledTimes(1);
      for (const module of modules) expect(module).toBe(created[0]);
      expect(getManifoldModuleSync()).toBe(created[0]);
      expect(await getManifoldModule()).toBe(created[0]);
      expect(factory).toHaveBeenCalledTimes(1);
    });

test('retries initialization after a shared failure', async () => {
  const failure = new Error('WASM could not be loaded');
  factory.mockRejectedValueOnce(failure);
  const module = {setup: vi.fn()};
  factory.mockResolvedValue(module);
  const {getManifoldModule, getManifoldModuleSync} = await import('./wasm.ts');
  const attempts = await Promise.allSettled([
    getManifoldModule(),
    getManifoldModule(),
  ]);
  expect(attempts.map(result => result.status)).toEqual([
    'rejected', 'rejected'
  ]);
  expect(factory).toHaveBeenCalledTimes(1);
  expect(getManifoldModuleSync()).toBeNull();
  expect(await getManifoldModule()).toBe(module);
  expect(factory).toHaveBeenCalledTimes(2);
  expect(module.setup).toHaveBeenCalledTimes(1);
});
