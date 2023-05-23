package manifold3d;

import org.junit.Test;
import manifold3d.Manifold;
import manifold3d.pub.DoubleMesh;
import manifold3d.glm.DoubleVec3;
import manifold3d.glm.DoubleVec3Vector;
import manifold3d.manifold.MeshIO;
import manifold3d.manifold.ExportOptions;

public class ManifoldTest {

    public ManifoldTest() {}

    @Test
    public void testManifold() {
        DoubleMesh mesh = new DoubleMesh();
        Manifold manifold = new Manifold(mesh);

        Manifold sphere = Manifold.Sphere(10.0f, 140);
        Manifold cube = Manifold.Cube(new DoubleVec3(15.0f, 15.0f, 15.0f), true);
        Manifold cylinder = Manifold.Cylinder(3, 30.0f, 30.0f, 0, false);

        Manifold diff = cube.subtract(sphere);
        Manifold intersection = cube.intersect(sphere);
        Manifold union = cube.add(sphere);

        DoubleMesh diffMesh = diff.GetMesh();
        DoubleMesh intersectMesh = intersection.GetMesh();
        DoubleMesh unionMesh = union.GetMesh();
        ExportOptions opts = new ExportOptions();
        opts.faceted(false);

        MeshIO.ExportMesh("CubeMinusSphere.stl", diffMesh, opts);
        MeshIO.ExportMesh("CubeIntersectSphere.glb", intersectMesh, opts);
        MeshIO.ExportMesh("CubeUnionSphere.obj", unionMesh, opts);

        Manifold hull = cylinder.ConvexHull(cube.TranslateZ(200.0)
                                            .TranslateX(50))
            .subtract(Manifold.Cylinder(100, 4f, 4f, 0, false).TranslateZ(-1f));
        DoubleMesh hullMesh = hull.GetMesh();

        DoubleVec3Vector vertPos = hullMesh.vertPos();
        //MeshIO.ExportMesh("HullCubes.stl", hullMesh, opts);
    }
}
