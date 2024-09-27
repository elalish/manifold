set(MANIFOLD_PUBLIC_HDR
	manifold/common.h
	manifold/iters.h
	manifold/manifold.h
	manifold/optional_assert.h
	manifold/parallel.h
	manifold/tri_dist.h
	manifold/vec_view.h
	)

if (MANIFOLD_CROSS_SECTION)
	list(APPEND MANIFOLD_PUBLIC_HDRS manifold/cross_section.h)
endif()

if (MANIFOLD_EXPORT)
	list(APPEND MANIFOLD_PUBLIC_HDRS manifold/meshIO.h)
endif()

if (MANIFOLD_POLYGON)
	list(APPEND MANIFOLD_PUBLIC_HDRS manifold/polygon.h)
endif()

install(FILES ${MANIFOLD_PUBLIC_HDRS}
	DESTINATION ${INCLUDE_DIR}/manifold
	)

