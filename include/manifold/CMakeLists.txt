set(MANIFOLD_PUBLIC_HDRS
	common.h
	iters.h
	manifold.h
	optional_assert.h
	parallel.h
	polygon.h
	vec_view.h
	)

set(MANIFOLD_C_BINDING_HDRS
	manifoldc.h
	types.h
	)
list(APPEND MANIFOLD_PUBLIC_HDRS ${MANIFOLD_C_BINDING_HDRS})

if (MANIFOLD_CROSS_SECTION)
	list(APPEND MANIFOLD_PUBLIC_HDRS cross_section.h)
endif()

if (MANIFOLD_EXPORT)
	list(APPEND MANIFOLD_PUBLIC_HDRS meshIO.h)
endif()

install(FILES ${MANIFOLD_PUBLIC_HDRS}
	DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/manifold
	)

