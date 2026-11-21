package com.example;

import static org.junit.jupiter.api.Assertions.*;

import java.lang.foreign.MemorySegment;

import org.junit.jupiter.api.Test;

import com.cadoodlecad.manifold.ManifoldBindings;

class TestNativeLibLoading {

	@Test
	void test() throws Throwable {
		ManifoldBindings manifold = new ManifoldBindings();
		MemorySegment seg = manifold.cube(10, 10, 10, false);
		
	}

}
