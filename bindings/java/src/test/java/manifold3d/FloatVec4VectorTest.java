package manifold3d;

import org.junit.Test;
import manifold3d.FloatVec4;
import manifold3d.FloatVec4Vector;

public class FloatVec4VectorTest {

    public FloatVec4VectorTest() {
    }

    @Test
    public void testMain() {
        FloatVec4 vec4 = new FloatVec4(1.0f, 2.0f, 3.0f, 4.0f);
        FloatVec4Vector vec4array = new FloatVec4Vector();

        vec4array.pushBack(vec4);
        vec4array.pushBack(vec4);

        assert vec4array.get(1).get(1) == 2.0f;
    }

}
