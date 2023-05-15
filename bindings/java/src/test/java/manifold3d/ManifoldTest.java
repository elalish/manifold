package manifold3d;

import org.junit.Test;
import manifold3d.Manifold;
import manifold3d.pub.DoubleMesh;

public class ManifoldTest {

    public ManifoldTest() {}

    @Test
    public void testManifold() {
        DoubleMesh mesh = new DoubleMesh();
        Manifold manifold = new Manifold(mesh);
    }
}
