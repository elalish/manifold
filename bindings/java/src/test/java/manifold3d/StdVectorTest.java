package manifold3d;

import org.junit.Test;
import manifold3d.FloatVector;
import manifold3d.UIntVector;

public class StdVectorTest {

    public StdVectorTest() {
    }

    @Test
    public void testFloatVector() {
        FloatVector vec = new FloatVector();
        vec.pushBack(1.0f);
        assert vec.get(0) == 1.0f;
    }

    @Test
    public void testUIntVector() {
        UIntVector vec = new UIntVector();
        vec.pushBack(1);
        assert vec.get(0) == 1;

        assert vec.size() == 1;
        vec.resize(20);
        assert vec.size() == 20;
    }
}
