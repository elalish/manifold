set(MANIFOLD_PUBLIC_HDR
	manifold/common.h
	manifold/iters.h
	manifold/manifold.h
	manifold/optional_assert.h
	manifold/parallel.h
	manifold/tri_dist.h
	manifold/vec_view.h
	)

if (POLYGON_ENABLED)
	list(APPEND MANIFOLD_PUBLIC_HDRS manifold/polygon.h)
endif()

if (MESHIO_ENABLED)
	list(APPEND MANIFOLD_PUBLIC_HDRS manifold/meshIO.h)
endif()

if (CROSSSECTION_ENABLED)
	list(APPEND MANIFOLD_PUBLIC_HDRS manifold/cross_section.h)
endif()

install(FILES ${MANIFOLD_PUBLIC_HDRS}
  	DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/manifold
	)

