package manifold3d;

import org.junit.Test;
import manifold3d.pub.DoubleMesh;
import manifold3d.glm.DoubleVec3;
import manifold3d.glm.DoubleVec3Vector;

public class DoubleMeshTest {

    public DoubleMeshTest() {
    }

    @Test
    public void testDoubleMesh() {
        DoubleMesh mesh = new DoubleMesh();

        DoubleVec3Vector v = new DoubleVec3Vector();
        v.pushBack(new DoubleVec3(3.0, 4.0, 5.0));
        v.pushBack(new DoubleVec3(6.0, 7.0, 8.0));

        assert v.size() == 2;
        assert v.get(0).get(0) == 3.0;

        mesh.vertPos(v);

        DoubleVec3Vector vp = mesh.vertPos();
        assert vp.get(1).get(1) == 7.0;
    }
}
