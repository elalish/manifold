#include <emscripten/bind.h>

using namespace emscripten;

#include <manifold.h>

using namespace manifold;

void logMesh(Mesh& mesh) {
    printf("# %d vertices:\n", (int)mesh.vertPos.size());
    for(int i = 0; i < mesh.vertPos.size(); i++) {
        printf("v %f %f %f\n",
            mesh.vertPos[i].x,
            mesh.vertPos[i].y,
            mesh.vertPos[i].z
        );
    }
    printf("# %d faces:\n", (int)mesh.triVerts.size());
    for(int i = 0; i < mesh.triVerts.size(); i++) {
        printf("f %d %d %d\n",
            mesh.triVerts[i].x + 1,
            mesh.triVerts[i].y + 1,
            mesh.triVerts[i].z + 1
        );
    }
}

struct Mesh makeEmptyMesh() {
  struct Mesh empty; return empty;
}

Manifold makeManifoldFromMesh(Mesh& mesh) {
    //logMesh(mesh);

    try {
        Manifold manifold(mesh);
        return manifold;
    }
    catch(const std::runtime_error& e) {
        puts(e.what());

        Manifold empty;
        return empty;
    }
}

Mesh makeMeshFromManifold(Manifold& manifold) {
    try {
        Mesh mesh = manifold.GetMesh();
        //logMesh(mesh);

        return mesh;
    }
    catch(const std::runtime_error& e) {
        puts(e.what());

        Mesh empty;
        return empty;
    }
}

void subtract(Manifold& a, Manifold& b) {
    a -= b;
}

EMSCRIPTEN_BINDINGS(whatever) {

    value_object<glm::ivec3>("ivec3")
        .field("0", &glm::ivec3::x)
        .field("1", &glm::ivec3::y)
        .field("2", &glm::ivec3::z)
        ;

    register_vector<glm::ivec3>("Vector_ivec3");

    value_object<glm::vec3>("vec3")
        .field("x", &glm::vec3::x)
        .field("y", &glm::vec3::y)
        .field("z", &glm::vec3::z)
        ;

    register_vector<glm::vec3>("Vector_vec3");

    value_object<glm::vec4>("vec4")
        .field("x", &glm::vec4::x)
        .field("y", &glm::vec4::y)
        .field("z", &glm::vec4::z)
        .field("w", &glm::vec4::w)
        ;

    register_vector<glm::vec4>("Vector_vec4");

    value_object<Mesh>("Mesh")
        .field("vertPos", &Mesh::vertPos)
        .field("triVerts", &Mesh::triVerts)
        .field("vertNormal", &Mesh::vertNormal)
        .field("halfedgeTangent", &Mesh::halfedgeTangent)
        ;

    function("makeEmptyMesh", &makeEmptyMesh);

    class_<Manifold>("Manifold")
        .constructor<Mesh>()
        .function("subtract", &subtract)
        .function("GetMesh", &Manifold::GetMesh)
        ;

    function("makeManifoldFromMesh", &makeManifoldFromMesh);
    function("makeMeshFromManifold", &makeMeshFromManifold);
}
