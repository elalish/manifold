package manifold3d;

import org.junit.Test;
import manifold3d.glm.FloatVec4;
import manifold3d.glm.FloatVec4Vector;

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

        assert vec4array.size() == 2.0;

        vec4array.set(1, new FloatVec4(5.0f, 6.0f, 7.0f, 8.0f));

        assert vec4array.get(1).get(1) == 6.0;
    }

}
