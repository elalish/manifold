package manifold3d;

import org.junit.Test;
import manifold3d.Manifold;
import manifold3d.pub.DoubleMesh;
import manifold3d.glm.DoubleVec3;
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

        Manifold diff = cube.subtract(sphere);

        DoubleMesh diffMesh = diff.GetMesh();
        ExportOptions opts = new ExportOptions();
        opts.faceted(true);

        MeshIO.ExportMesh("CubeMinusSphere.stl", diffMesh, opts);
    }
}
