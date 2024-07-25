#ifndef QUICKHULL_HPP_
#define QUICKHULL_HPP_
#include <deque>
#include <array>
#include <limits>
#include <unordered_map>
#include <fstream>
#include <memory>
#include <cmath>
#include <iostream>
#include <cassert>
#include <cinttypes>
#include <vector>

// Pool.hpp


namespace quickhull {
	
	template<typename T>
	class Pool {
		std::vector<std::unique_ptr<T>> m_data;
	public:
		void clear() {
			m_data.clear();
		}
		
		void reclaim(std::unique_ptr<T>& ptr) {
			m_data.push_back(std::move(ptr));
		}
		
		std::unique_ptr<T> get() {
			if (m_data.size()==0) {
				return std::unique_ptr<T>(new T());
			}
			auto it = m_data.end()-1;
			std::unique_ptr<T> r = std::move(*it);
			m_data.erase(it);
			return r;
		}
		
	};
	

// Vector3.hpp


	template <typename T>
	class Vector3
	{
	public:
		Vector3() = default;
		
		Vector3(T x, T y, T z) : x(x), y(y), z(z) {
			
		}
		
		T x,y,z;
		
		T dotProduct(const Vector3& other) const {
			return x*other.x+y*other.y+z*other.z;
		}
		
		void normalize() {
			const T len = getLength();
			x/=len;
			y/=len;
			z/=len;
		}
		
		Vector3 getNormalized() const {
			const T len = getLength();
			return Vector3(x/len,y/len,z/len);
		}
		
		T getLength() const {
			return std::sqrt(x*x+y*y+z*z);
		}
		
		Vector3 operator-(const Vector3& other) const {
			return Vector3(x-other.x,y-other.y,z-other.z);
		}
		
		Vector3 operator+(const Vector3& other) const {
			return Vector3(x+other.x,y+other.y,z+other.z);
		}
		
		Vector3& operator+=(const Vector3& other) {
			x+=other.x;
			y+=other.y;
			z+=other.z;
			return *this;
		}
		Vector3& operator-=(const Vector3& other) {
			x-=other.x;
			y-=other.y;
			z-=other.z;
			return *this;
		}
		Vector3& operator*=(T c) {
			x*=c;
			y*=c;
			z*=c;
			return *this;
		}
		
		Vector3& operator/=(T c) {
			x/=c;
			y/=c;
			z/=c;
			return *this;
		}
		
		Vector3 operator-() const {
			return Vector3(-x,-y,-z);
		}

		template<typename S>
		Vector3 operator*(S c) const {
			return Vector3(x*c,y*c,z*c);
		}
		
		template<typename S>
		Vector3 operator/(S c) const {
			return Vector3(x/c,y/c,z/c);
		}
		
		T getLengthSquared() const {
			return x*x + y*y + z*z;
		}
		
		bool operator!=(const Vector3& o) const {
			return x != o.x || y != o.y || z != o.z;
		}
		
		// Projection onto another vector
		Vector3 projection(const Vector3& o) const {
			T C = dotProduct(o)/o.getLengthSquared();
			return o*C;
		}
		
		Vector3 crossProduct (const Vector3& rhs ) {
			T a = y * rhs.z - z * rhs.y ;
			T b = z * rhs.x - x * rhs.z ;
			T c = x * rhs.y - y * rhs.x ;
			Vector3 product( a , b , c ) ;
			return product ;
		}
		
		T getDistanceTo(const Vector3& other) const {
			Vector3 diff = *this - other;
			return diff.getLength();
		}
		
		T getSquaredDistanceTo(const Vector3& other) const {
			const T dx = x-other.x;
			const T dy = y-other.y;
			const T dz = z-other.z;
			return dx*dx+dy*dy+dz*dz;
		}
		
	};
	
	// Overload also << operator for easy printing of debug data
	template <typename T>
	inline std::ostream& operator<<(std::ostream& os, const Vector3<T>& vec) {
		os << "(" << vec.x << "," << vec.y << "," << vec.z << ")";
		return os;
	}
	
	template <typename T>
	inline Vector3<T> operator*(T c, const Vector3<T>& v) {
		return Vector3<T>(v.x*c,v.y*c,v.z*c);
	}
	
// Plane.hpp 

	template<typename T>
	class Plane {
	public:
		Vector3<T> m_N;
		
		// Signed distance (if normal is of length 1) to the plane from origin
		T m_D;
		
		// Normal length squared
		T m_sqrNLength;

		bool isPointOnPositiveSide(const Vector3<T>& Q) const {
			T d = m_N.dotProduct(Q)+m_D;
			if (d>=0) return true;
			return false;
		}

		Plane() = default;

		// Construct a plane using normal N and any point P on the plane
		Plane(const Vector3<T>& N, const Vector3<T>& P) : m_N(N), m_D(-N.dotProduct(P)), m_sqrNLength(m_N.x*m_N.x+m_N.y*m_N.y+m_N.z*m_N.z) {
			
		}
	};

// Ray.hpp 

	template <typename T>
	struct Ray {
		const Vector3<T> m_S;
		const Vector3<T> m_V;
		const T m_VInvLengthSquared;
		
		Ray(const Vector3<T>& S,const Vector3<T>& V) : m_S(S), m_V(V), m_VInvLengthSquared(1/m_V.getLengthSquared()) {
		}
	};
	

// VertexDataSource

	
	template<typename T>
	class VertexDataSource {
		const Vector3<T>* m_ptr;
		size_t m_count;
	
	public:
		VertexDataSource(const Vector3<T>* ptr, size_t count) : m_ptr(ptr), m_count(count) {
			
		}
		
		VertexDataSource(const std::vector<Vector3<T>>& vec) : m_ptr(&vec[0]), m_count(vec.size()) {
			
		}
		
		VertexDataSource() : m_ptr(nullptr), m_count(0) {
			
		}
		
		VertexDataSource& operator=(const VertexDataSource& other) = default;
		
		size_t size() const {
			return m_count;
		}
		
		const Vector3<T>& operator[](size_t index) const {
			return m_ptr[index];
		}
		
		const Vector3<T>* begin() const {
			return m_ptr;
		}
		
		const Vector3<T>* end() const {
			return m_ptr + m_count;
		}
	};
	


// Mesh.hpp 


	template <typename T>
	class MeshBuilder {
	public:
		struct HalfEdge {
			size_t m_endVertex;
			size_t m_opp;
			size_t m_face;
			size_t m_next;
			
			void disable() {
				m_endVertex = std::numeric_limits<size_t>::max();
			}
			
			bool isDisabled() const {
				return m_endVertex == std::numeric_limits<size_t>::max();
			}
		};

		struct Face {
			size_t m_he;
			Plane<T> m_P{};
			T m_mostDistantPointDist;
			size_t m_mostDistantPoint;
			size_t m_visibilityCheckedOnIteration;
			std::uint8_t m_isVisibleFaceOnCurrentIteration : 1;
			std::uint8_t m_inFaceStack : 1;
			std::uint8_t m_horizonEdgesOnCurrentIteration : 3; // Bit for each half edge assigned to this face, each being 0 or 1 depending on whether the edge belongs to horizon edge
			std::unique_ptr<std::vector<size_t>> m_pointsOnPositiveSide;

			Face() : m_he(std::numeric_limits<size_t>::max()),
					 m_mostDistantPointDist(0),
					 m_mostDistantPoint(0),
					 m_visibilityCheckedOnIteration(0),
					 m_isVisibleFaceOnCurrentIteration(0),
					 m_inFaceStack(0),
					 m_horizonEdgesOnCurrentIteration(0)
			{

			}
			
			void disable() {
				m_he = std::numeric_limits<size_t>::max();
			}

			bool isDisabled() const {
				return m_he == std::numeric_limits<size_t>::max();
			}
		};

		// Mesh data
		std::vector<Face> m_faces;
		std::vector<HalfEdge> m_halfEdges;
		
		// When the mesh is modified and faces and half edges are removed from it, we do not actually remove them from the container vectors.
		// Insted, they are marked as disabled which means that the indices can be reused when we need to add new faces and half edges to the mesh.
		// We store the free indices in the following vectors.
		std::vector<size_t> m_disabledFaces,m_disabledHalfEdges;
		
		size_t addFace() {
			if (m_disabledFaces.size()) {
				size_t index = m_disabledFaces.back();
				auto& f = m_faces[index];
				assert(f.isDisabled());
				assert(!f.m_pointsOnPositiveSide);
				f.m_mostDistantPointDist = 0;
				m_disabledFaces.pop_back();
				return index;
			}
			m_faces.emplace_back();
			return m_faces.size()-1;
		}

		size_t addHalfEdge()	{
			if (m_disabledHalfEdges.size()) {
				const size_t index = m_disabledHalfEdges.back();
				m_disabledHalfEdges.pop_back();
				return index;
			}
			m_halfEdges.emplace_back();
			return m_halfEdges.size()-1;
		}

		// Mark a face as disabled and return a pointer to the points that were on the positive of it.
		std::unique_ptr<std::vector<size_t>> disableFace(size_t faceIndex) {
			auto& f = m_faces[faceIndex];
			f.disable();
			m_disabledFaces.push_back(faceIndex);
			return std::move(f.m_pointsOnPositiveSide);
		}

		void disableHalfEdge(size_t heIndex) {
			auto& he = m_halfEdges[heIndex];
			he.disable();
			m_disabledHalfEdges.push_back(heIndex);
		}

		MeshBuilder() = default;
		
		// Create a mesh with initial tetrahedron ABCD. Dot product of AB with the normal of triangle ABC should be negative.
		void setup(size_t a, size_t b, size_t c, size_t d) {
			m_faces.clear();
			m_halfEdges.clear();
			m_disabledFaces.clear();
			m_disabledHalfEdges.clear();
			
			m_faces.reserve(4);
			m_halfEdges.reserve(12);
			
			// Create halfedges
			HalfEdge AB;
			AB.m_endVertex = b;
			AB.m_opp = 6;
			AB.m_face = 0;
			AB.m_next = 1;
			m_halfEdges.push_back(AB);

			HalfEdge BC;
			BC.m_endVertex = c;
			BC.m_opp = 9;
			BC.m_face = 0;
			BC.m_next = 2;
			m_halfEdges.push_back(BC);

			HalfEdge CA;
			CA.m_endVertex = a;
			CA.m_opp = 3;
			CA.m_face = 0;
			CA.m_next = 0;
			m_halfEdges.push_back(CA);

			HalfEdge AC;
			AC.m_endVertex = c;
			AC.m_opp = 2;
			AC.m_face = 1;
			AC.m_next = 4;
			m_halfEdges.push_back(AC);

			HalfEdge CD;
			CD.m_endVertex = d;
			CD.m_opp = 11;
			CD.m_face = 1;
			CD.m_next = 5;
			m_halfEdges.push_back(CD);

			HalfEdge DA;
			DA.m_endVertex = a;
			DA.m_opp = 7;
			DA.m_face = 1;
			DA.m_next = 3;
			m_halfEdges.push_back(DA);

			HalfEdge BA;
			BA.m_endVertex = a;
			BA.m_opp = 0;
			BA.m_face = 2;
			BA.m_next = 7;
			m_halfEdges.push_back(BA);

			HalfEdge AD;
			AD.m_endVertex = d;
			AD.m_opp = 5;
			AD.m_face = 2;
			AD.m_next = 8;
			m_halfEdges.push_back(AD);

			HalfEdge DB;
			DB.m_endVertex = b;
			DB.m_opp = 10;
			DB.m_face = 2;
			DB.m_next = 6;
			m_halfEdges.push_back(DB);

			HalfEdge CB;
			CB.m_endVertex = b;
			CB.m_opp = 1;
			CB.m_face = 3;
			CB.m_next = 10;
			m_halfEdges.push_back(CB);

			HalfEdge BD;
			BD.m_endVertex = d;
			BD.m_opp = 8;
			BD.m_face = 3;
			BD.m_next = 11;
			m_halfEdges.push_back(BD);

			HalfEdge DC;
			DC.m_endVertex = c;
			DC.m_opp = 4;
			DC.m_face = 3;
			DC.m_next = 9;
			m_halfEdges.push_back(DC);

			// Create faces
			Face ABC;
			ABC.m_he = 0;
			m_faces.push_back(std::move(ABC));

			Face ACD;
			ACD.m_he = 3;
			m_faces.push_back(std::move(ACD));

			Face BAD;
			BAD.m_he = 6;
			m_faces.push_back(std::move(BAD));

			Face CBD;
			CBD.m_he = 9;
			m_faces.push_back(std::move(CBD));
		}

		std::array<size_t,3> getVertexIndicesOfFace(const Face& f) const {
			std::array<size_t,3> v;
			const HalfEdge* he = &m_halfEdges[f.m_he];
			v[0] = he->m_endVertex;
			he = &m_halfEdges[he->m_next];
			v[1] = he->m_endVertex;
			he = &m_halfEdges[he->m_next];
			v[2] = he->m_endVertex;
			return v;
		}

		std::array<size_t,2> getVertexIndicesOfHalfEdge(const HalfEdge& he) const {
			return {m_halfEdges[he.m_opp].m_endVertex,he.m_endVertex};
		}

		std::array<size_t,3> getHalfEdgeIndicesOfFace(const Face& f) const {
			return {f.m_he,m_halfEdges[f.m_he].m_next,m_halfEdges[m_halfEdges[f.m_he].m_next].m_next};
		}
	};
	




// ConvexHull.hpp

	template<typename T>
	class ConvexHull {
		std::unique_ptr<std::vector<Vector3<T>>> m_optimizedVertexBuffer;
		VertexDataSource<T> m_vertices;
		std::vector<size_t> m_indices;
	public:
		ConvexHull() {}
		
		// Copy constructor
		ConvexHull(const ConvexHull& o) {
			m_indices = o.m_indices;
			if (o.m_optimizedVertexBuffer) {
				m_optimizedVertexBuffer.reset(new std::vector<Vector3<T>>(*o.m_optimizedVertexBuffer));
				m_vertices = VertexDataSource<T>(*m_optimizedVertexBuffer);
			}
			else {
				m_vertices = o.m_vertices;
			}
		}
		
		ConvexHull& operator=(const ConvexHull& o) {
			if (&o == this) {
				return *this;
			}
			m_indices = o.m_indices;
			if (o.m_optimizedVertexBuffer) {
				m_optimizedVertexBuffer.reset(new std::vector<Vector3<T>>(*o.m_optimizedVertexBuffer));
				m_vertices = VertexDataSource<T>(*m_optimizedVertexBuffer);
			}
			else {
				m_vertices = o.m_vertices;
			}
			return *this;
		}
		
		ConvexHull(ConvexHull&& o) {
			m_indices = std::move(o.m_indices);
			if (o.m_optimizedVertexBuffer) {
				m_optimizedVertexBuffer = std::move(o.m_optimizedVertexBuffer);
				o.m_vertices = VertexDataSource<T>();
				m_vertices = VertexDataSource<T>(*m_optimizedVertexBuffer);
			}
			else {
				m_vertices = o.m_vertices;
			}
		}
		
		ConvexHull& operator=(ConvexHull&& o) {
			if (&o == this) {
				return *this;
			}
			m_indices = std::move(o.m_indices);
			if (o.m_optimizedVertexBuffer) {
				m_optimizedVertexBuffer = std::move(o.m_optimizedVertexBuffer);
				o.m_vertices = VertexDataSource<T>();
				m_vertices = VertexDataSource<T>(*m_optimizedVertexBuffer);
			}
			else {
				m_vertices = o.m_vertices;
			}
			return *this;
		}
		
		// Construct vertex and index buffers from half edge mesh and pointcloud
		ConvexHull(const MeshBuilder<T>& mesh, const VertexDataSource<T>& pointCloud, bool CCW, bool useOriginalIndices) {
			if (!useOriginalIndices) {
				m_optimizedVertexBuffer.reset(new std::vector<Vector3<T>>());
			}
			
			std::vector<bool> faceProcessed(mesh.m_faces.size(),false);
			std::vector<size_t> faceStack;
			std::unordered_map<size_t,size_t> vertexIndexMapping; // Map vertex indices from original point cloud to the new mesh vertex indices
			for (size_t i = 0;i<mesh.m_faces.size();i++) {
				if (!mesh.m_faces[i].isDisabled()) {
					faceStack.push_back(i);
					break;
				}
			}
			if (faceStack.size()==0) {
				return;
			}

			const size_t iCCW = CCW ? 1 : 0;
			const size_t finalMeshFaceCount = mesh.m_faces.size() - mesh.m_disabledFaces.size();
			m_indices.reserve(finalMeshFaceCount*3);

			while (faceStack.size()) {
				auto it = faceStack.end()-1;
				size_t top = *it;
				assert(!mesh.m_faces[top].isDisabled());
				faceStack.erase(it);
				if (faceProcessed[top]) {
					continue;
				}
				else {
					faceProcessed[top]=true;
					auto halfEdges = mesh.getHalfEdgeIndicesOfFace(mesh.m_faces[top]);
					size_t adjacent[] = {mesh.m_halfEdges[mesh.m_halfEdges[halfEdges[0]].m_opp].m_face,mesh.m_halfEdges[mesh.m_halfEdges[halfEdges[1]].m_opp].m_face,mesh.m_halfEdges[mesh.m_halfEdges[halfEdges[2]].m_opp].m_face};
					for (auto a : adjacent) {
						if (!faceProcessed[a] && !mesh.m_faces[a].isDisabled()) {
							faceStack.push_back(a);
						}
					}
					auto vertices = mesh.getVertexIndicesOfFace(mesh.m_faces[top]);
					if (!useOriginalIndices) {
						for (auto& v : vertices) {
							auto itV = vertexIndexMapping.find(v);
							if (itV == vertexIndexMapping.end()) {
								m_optimizedVertexBuffer->push_back(pointCloud[v]);
								vertexIndexMapping[v] = m_optimizedVertexBuffer->size()-1;
								v = m_optimizedVertexBuffer->size()-1;
							}
							else {
								v = itV->second;
							}
						}
					}
					m_indices.push_back(vertices[0]);
					m_indices.push_back(vertices[1 + iCCW]);
					m_indices.push_back(vertices[2 - iCCW]);
				}
			}
			
			if (!useOriginalIndices) {
				m_vertices = VertexDataSource<T>(*m_optimizedVertexBuffer);
			}
			else {
				m_vertices = pointCloud;
			}
		}

		std::vector<size_t>& getIndexBuffer() {
			return m_indices;
		}

		const std::vector<size_t>& getIndexBuffer() const {
			return m_indices;
		}

		VertexDataSource<T>& getVertexBuffer() {
			return m_vertices;
		}
		
		const VertexDataSource<T>& getVertexBuffer() const {
			return m_vertices;
		}
		
		// Export the mesh to a Waveform OBJ file
		void writeWaveformOBJ(const std::string& filename, const std::string& objectName = "quickhull") const
		{
			std::ofstream objFile;
			objFile.open (filename);
			objFile << "o " << objectName << "\n";
			for (const auto& v : getVertexBuffer()) {
				objFile << "v " << v.x << " " << v.y << " " << v.z << "\n";
			}
			const auto& indBuf = getIndexBuffer();
			size_t triangleCount = indBuf.size()/3;
			for (size_t i=0;i<triangleCount;i++) {
				objFile << "f " << indBuf[i*3]+1 << " " << indBuf[i*3+1]+1 << " " << indBuf[i*3+2]+1 << "\n";
			}
			objFile.close();
		}

	};



// HalfEdgeMesh.hpp
	
	template<typename FloatType, typename IndexType>
	class HalfEdgeMesh {
	public:
		
		struct HalfEdge {
			IndexType m_endVertex;
			IndexType m_opp;
			IndexType m_face;
			IndexType m_next;
		};
		
		struct Face {
			IndexType m_halfEdgeIndex; // Index of one of the half edges of this face
		};
		
		std::vector<Vector3<FloatType>> m_vertices;
		std::vector<Face> m_faces;
		std::vector<HalfEdge> m_halfEdges;
		
		HalfEdgeMesh(const MeshBuilder<FloatType>& builderObject, const VertexDataSource<FloatType>& vertexData )
		{
			std::unordered_map<IndexType,IndexType> faceMapping;
			std::unordered_map<IndexType,IndexType> halfEdgeMapping;
			std::unordered_map<IndexType, IndexType> vertexMapping;
			
			size_t i=0;
			for (const auto& face : builderObject.m_faces) {
				if (!face.isDisabled()) {
					m_faces.push_back({static_cast<IndexType>(face.m_he)});
					faceMapping[i] = m_faces.size()-1;
					
					const auto heIndices = builderObject.getHalfEdgeIndicesOfFace(face);
					for (const auto heIndex : heIndices) {
						const IndexType vertexIndex = builderObject.m_halfEdges[heIndex].m_endVertex;
						if (vertexMapping.count(vertexIndex)==0) {
							m_vertices.push_back(vertexData[vertexIndex]);
							vertexMapping[vertexIndex] = m_vertices.size()-1;
						}
					}
				}
				i++;
			}
			
			i=0;
			for (const auto& halfEdge : builderObject.m_halfEdges) {
				if (!halfEdge.isDisabled()) {
					m_halfEdges.push_back({static_cast<IndexType>(halfEdge.m_endVertex),static_cast<IndexType>(halfEdge.m_opp),static_cast<IndexType>(halfEdge.m_face),static_cast<IndexType>(halfEdge.m_next)});
					halfEdgeMapping[i] = m_halfEdges.size()-1;
				}
				i++;
			}
			
			for (auto& face : m_faces) {
				assert(halfEdgeMapping.count(face.m_halfEdgeIndex) == 1);
				face.m_halfEdgeIndex = halfEdgeMapping[face.m_halfEdgeIndex];
			}
			
			for (auto& he : m_halfEdges) {
				he.m_face = faceMapping[he.m_face];
				he.m_opp = halfEdgeMapping[he.m_opp];
				he.m_next = halfEdgeMapping[he.m_next];
				he.m_endVertex = vertexMapping[he.m_endVertex];
			}
		}
		
	};



// MathUtils.hpp

	namespace mathutils {
		
		template <typename T>
		inline T getSquaredDistanceBetweenPointAndRay(const Vector3<T>& p, const Ray<T>& r) {
			const Vector3<T> s = p-r.m_S;
			T t = s.dotProduct(r.m_V);
			return s.getLengthSquared() - t*t*r.m_VInvLengthSquared;
		}
		
		// Note that the unit of distance returned is relative to plane's normal's length (divide by N.getNormalized() if needed to get the "real" distance).
		template <typename T>
		inline T getSignedDistanceToPlane(const Vector3<T>& v, const Plane<T>& p) {
			return p.m_N.dotProduct(v) + p.m_D;
		}
		
		template <typename T>
		inline Vector3<T> getTriangleNormal(const Vector3<T>& a,const Vector3<T>& b,const Vector3<T>& c) {
			// We want to get (a-c).crossProduct(b-c) without constructing temp vectors
			T x = a.x - c.x;
			T y = a.y - c.y;
			T z = a.z - c.z;
			T rhsx = b.x - c.x;
			T rhsy = b.y - c.y;
			T rhsz = b.z - c.z;
			T px = y * rhsz - z * rhsy ;
			T py = z * rhsx - x * rhsz ;
			T pz = x * rhsy - y * rhsx ;
			return Vector3<T>(px,py,pz);
		}
		
		
	}
	

// QuickHull.hpp

/*
 * Implementation of the 3D QuickHull algorithm by Antti Kuukka
 *
 * No copyrights. What follows is 100% Public Domain.
 *
 *
 *
 * INPUT:  a list of points in 3D space (for example, vertices of a 3D mesh)
 *
 * OUTPUT: a ConvexHull object which provides vertex and index buffers of the generated convex hull as a triangle mesh.
 *
 *
 *
 * The implementation is thread-safe if each thread is using its own QuickHull object.
 *
 *
 * SUMMARY OF THE ALGORITHM:
 *         - Create initial simplex (tetrahedron) using extreme points. We have four faces now and they form a convex mesh M.
 *         - For each point, assign them to the first face for which they are on the positive side of (so each point is assigned to at most
 *           one face). Points inside the initial tetrahedron are left behind now and no longer affect the calculations.
 *         - Add all faces that have points assigned to them to Face Stack.
 *         - Iterate until Face Stack is empty:
 *              - Pop topmost face F from the stack
 *              - From the points assigned to F, pick the point P that is farthest away from the plane defined by F.
 *              - Find all faces of M that have P on their positive side. Let us call these the "visible faces".
 *              - Because of the way M is constructed, these faces are connected. Solve their horizon edge loop.
 *				- "Extrude to P": Create new faces by connecting P with the points belonging to the horizon edge. Add the new faces to M and remove the visible
 *                faces from M.
 *              - Each point that was assigned to visible faces is now assigned to at most one of the newly created faces.
 *              - Those new faces that have points assigned to them are added to the top of Face Stack.
 *          - M is now the convex hull.
 *
 * TO DO:
 *  - Implement a proper 2D QuickHull and use that to solve the degenerate 2D case (when all the points lie on the same plane in 3D space).
 * */
	
	struct DiagnosticsData {
		size_t m_failedHorizonEdges; // How many times QuickHull failed to solve the horizon edge. Failures lead to degenerated convex hulls.
		
		DiagnosticsData() : m_failedHorizonEdges(0) { }
	};

	template<typename FloatType>
	FloatType defaultEps();

	template<typename FloatType>
	class QuickHull {
		using vec3 = Vector3<FloatType>;

		FloatType m_epsilon, m_epsilonSquared, m_scale;
		bool m_planar;
		std::vector<vec3> m_planarPointCloudTemp;
		VertexDataSource<FloatType> m_vertexData;
		MeshBuilder<FloatType> m_mesh;
		std::array<size_t,6> m_extremeValues;
		DiagnosticsData m_diagnostics;

		// Temporary variables used during iteration process
		std::vector<size_t> m_newFaceIndices;
		std::vector<size_t> m_newHalfEdgeIndices;
		std::vector< std::unique_ptr<std::vector<size_t>> > m_disabledFacePointVectors;
		std::vector<size_t> m_visibleFaces;
		std::vector<size_t> m_horizonEdges;
		struct FaceData {
			size_t m_faceIndex;
			size_t m_enteredFromHalfEdge; // If the face turns out not to be visible, this half edge will be marked as horizon edge
			FaceData(size_t fi, size_t he) : m_faceIndex(fi),m_enteredFromHalfEdge(he) {}
		};
		std::vector<FaceData> m_possiblyVisibleFaces;
		std::deque<size_t> m_faceList;

		// Create a half edge mesh representing the base tetrahedron from which the QuickHull iteration proceeds. m_extremeValues must be properly set up when this is called.
		void setupInitialTetrahedron();

		// Given a list of half edges, try to rearrange them so that they form a loop. Return true on success.
		bool reorderHorizonEdges(std::vector<size_t>& horizonEdges);
		
		// Find indices of extreme values (max x, min x, max y, min y, max z, min z) for the given point cloud
		std::array<size_t,6> getExtremeValues();
		
		// Compute scale of the vertex data.
		FloatType getScale(const std::array<size_t,6>& extremeValues);
		
		// Each face contains a unique pointer to a vector of indices. However, many - often most - faces do not have any points on the positive
		// side of them especially at the the end of the iteration. When a face is removed from the mesh, its associated point vector, if such
		// exists, is moved to the index vector pool, and when we need to add new faces with points on the positive side to the mesh,
		// we reuse these vectors. This reduces the amount of std::vectors we have to deal with, and impact on performance is remarkable.
		Pool<std::vector<size_t>> m_indexVectorPool;
		inline std::unique_ptr<std::vector<size_t>> getIndexVectorFromPool();
		inline void reclaimToIndexVectorPool(std::unique_ptr<std::vector<size_t>>& ptr);
		
		// Associates a point with a face if the point resides on the positive side of the plane. Returns true if the points was on the positive side.
		inline bool addPointToFace(typename MeshBuilder<FloatType>::Face& f, size_t pointIndex);
		
		// This will update m_mesh from which we create the ConvexHull object that getConvexHull function returns
		void createConvexHalfEdgeMesh();
		
		// Constructs the convex hull into a MeshBuilder object which can be converted to a ConvexHull or Mesh object
		void buildMesh(const VertexDataSource<FloatType>& pointCloud, bool CCW, bool useOriginalIndices, FloatType eps);
		
		// The public getConvexHull functions will setup a VertexDataSource object and call this
		ConvexHull<FloatType> getConvexHull(const VertexDataSource<FloatType>& pointCloud, bool CCW, bool useOriginalIndices, FloatType eps);
	public:
		// Computes convex hull for a given point cloud.
		// Params:
		//   pointCloud: a vector of of 3D points
		//   CCW: whether the output mesh triangles should have CCW orientation
		//   useOriginalIndices: should the output mesh use same vertex indices as the original point cloud. If this is false,
		//      then we generate a new vertex buffer which contains only the vertices that are part of the convex hull.
		//   eps: minimum distance to a plane to consider a point being on positive of it (for a point cloud with scale 1)
		ConvexHull<FloatType> getConvexHull(const std::vector<Vector3<FloatType>>& pointCloud,
											bool CCW,
											bool useOriginalIndices,
											FloatType eps = defaultEps<FloatType>());
		
		// Computes convex hull for a given point cloud.
		// Params:
		//   vertexData: pointer to the first 3D point of the point cloud
		//   vertexCount: number of vertices in the point cloud
		//   CCW: whether the output mesh triangles should have CCW orientation
		//   useOriginalIndices: should the output mesh use same vertex indices as the original point cloud. If this is false,
		//      then we generate a new vertex buffer which contains only the vertices that are part of the convex hull.
		//   eps: minimum distance to a plane to consider a point being on positive side of it (for a point cloud with scale 1)
		ConvexHull<FloatType> getConvexHull(const Vector3<FloatType>* vertexData,
											size_t vertexCount,
											bool CCW,
											bool useOriginalIndices,
											FloatType eps = defaultEps<FloatType>());
		
		// Computes convex hull for a given point cloud. This function assumes that the vertex data resides in memory
		// in the following format: x_0,y_0,z_0,x_1,y_1,z_1,...
		// Params:
		//   vertexData: pointer to the X component of the first point of the point cloud.
		//   vertexCount: number of vertices in the point cloud
		//   CCW: whether the output mesh triangles should have CCW orientation
		//   useOriginalIndices: should the output mesh use same vertex indices as the original point cloud. If this is false,
		//      then we generate a new vertex buffer which contains only the vertices that are part of the convex hull.
		//   eps: minimum distance to a plane to consider a point being on positive side of it (for a point cloud with scale 1)
		ConvexHull<FloatType> getConvexHull(const FloatType* vertexData,
											size_t vertexCount,
											bool CCW,
											bool useOriginalIndices,
											FloatType eps = defaultEps<FloatType>());
		
		// Computes convex hull for a given point cloud. This function assumes that the vertex data resides in memory
		// in the following format: x_0,y_0,z_0,x_1,y_1,z_1,...
		// Params:
		//   vertexData: pointer to the X component of the first point of the point cloud.
		//   vertexCount: number of vertices in the point cloud
		//   CCW: whether the output mesh triangles should have CCW orientation
		//   eps: minimum distance to a plane to consider a point being on positive side of it (for a point cloud with scale 1)
		// Returns:
		//   Convex hull of the point cloud as a mesh object with half edge structure.
		HalfEdgeMesh<FloatType, size_t> getConvexHullAsMesh(const FloatType* vertexData,
															size_t vertexCount,
															bool CCW,
															FloatType eps = defaultEps<FloatType>());
		
		// Get diagnostics about last generated convex hull
		const DiagnosticsData& getDiagnostics() {
			return m_diagnostics;
		}
	};
	
	/*
	 * Inline function definitions
	 */
	
	template<typename T>
	std::unique_ptr<std::vector<size_t>> QuickHull<T>::getIndexVectorFromPool() {
		auto r = m_indexVectorPool.get();
		r->clear();
		return r;
	}
	
	template<typename T>
	void QuickHull<T>::reclaimToIndexVectorPool(std::unique_ptr<std::vector<size_t>>& ptr) {
		const size_t oldSize = ptr->size();
		if ((oldSize+1)*128 < ptr->capacity()) {
			// Reduce memory usage! Huge vectors are needed at the beginning of iteration when faces have many points on their positive side. Later on, smaller vectors will suffice.
			ptr.reset(nullptr);
			return;
		}
		m_indexVectorPool.reclaim(ptr);
	}

	template<typename T>
	bool QuickHull<T>::addPointToFace(typename MeshBuilder<T>::Face& f, size_t pointIndex) {
		const T D = mathutils::getSignedDistanceToPlane(m_vertexData[ pointIndex ],f.m_P);
		if (D>0 && D*D > m_epsilonSquared*f.m_P.m_sqrNLength) {
			if (!f.m_pointsOnPositiveSide) {
				f.m_pointsOnPositiveSide = std::move(getIndexVectorFromPool());
			}
			f.m_pointsOnPositiveSide->push_back( pointIndex );
			if (D > f.m_mostDistantPointDist) {
				f.m_mostDistantPointDist = D;
				f.m_mostDistantPoint = pointIndex;
			}
			return true;
		}
		return false;
	}

}


#endif /* QUICKHULL_HPP_ */
