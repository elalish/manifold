package manifold3d;

import org.junit.Test;
import org.junit.Assert;
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
        Manifold cube = Manifold.Cube(new DoubleVec3(15.0f, 15.0f, 15.0f), false);
        Manifold cylinder = Manifold.Cylinder(3, 30.0f, 30.0f, 0, false).translateX(20).translateY(20).translateZ(-3.0);

        Manifold diff = cube.subtract(sphere);
        Manifold intersection = cube.intersect(sphere);
        Manifold union = cube.add(sphere);

        DoubleMesh diffMesh = diff.getMesh();
        DoubleMesh intersectMesh = intersection.getMesh();
        DoubleMesh unionMesh = union.getMesh();
        ExportOptions opts = new ExportOptions();
        opts.faceted(false);

        MeshIO.ExportMesh("CubeMinusSphere.stl", diffMesh, opts);
        MeshIO.ExportMesh("CubeIntersectSphere.glb", intersectMesh, opts);
        MeshIO.ExportMesh("CubeUnionSphere.obj", unionMesh, opts);

        Manifold hull = cylinder.convexHull(cube.translateZ(100.0));
        DoubleMesh hullMesh = hull.getMesh();

        //Assert.assertEquals(hull.status(), 0);
        //Assert.assertNotEquals(hull.getProperties().volume(), 0.0, 0.001);

        System.out.println("volume: " + hull.getProperties().volume());
        System.out.println("Status: " + hull.status());
        System.out.flush();
        assert hull.getProperties().volume() > 0.0;

        DoubleVec3Vector vertPos = hullMesh.vertPos();
        //MeshIO.ExportMesh("HullCubes.stl", hullMesh, opts);
    }
}
