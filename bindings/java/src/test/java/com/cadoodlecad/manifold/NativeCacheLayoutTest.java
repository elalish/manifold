// Copyright 2026 The Manifold Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.cadoodlecad.manifold;

import static org.junit.jupiter.api.Assertions.*;

import java.nio.file.Files;
import java.nio.file.Path;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Where the native libraries get extracted to. A caller-supplied cache must not
 * be reused verbatim across jar upgrades, or the old binaries are loaded with
 * the new descriptors.
 */
class NativeCacheLayoutTest {

	@Test
	void callerSuppliedRootGetsAVersionSubdirectory(@TempDir Path root) throws Throwable {
		Path dir = ManifoldBindings.nativeCacheDir(root.toFile()).toPath();
		assertEquals(root.resolve(ManifoldBindings.cacheVersion()), dir);
		assertNotEquals(root, dir, "the caller's directory must not be the extraction target");
	}

	@Test
	void noRootGetsAFreshTemporaryDirectory() throws Throwable {
		Path first = ManifoldBindings.nativeCacheDir(null).toPath();
		Path second = ManifoldBindings.nativeCacheDir(null).toPath();
		assertTrue(Files.isDirectory(first));
		assertNotEquals(first, second, "each caller gets its own directory");
	}

	@Test
	void versionFallsBackWhenTheJarIsUnversioned() {
		String version = ManifoldBindings.cacheVersion();
		assertFalse(version.isBlank(), "the cache path needs a stable component");
	}
}
