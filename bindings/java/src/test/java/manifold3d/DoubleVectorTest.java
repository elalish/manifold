package manifold3d;

import org.junit.Test;
import manifold3d.DoubleVec3;
import manifold3d.DoubleVec4;

public class DoubleVectorTest {

    public DoubleVectorTest() {
    }

    @Test
    public void testMain() {
        DoubleVec3 vec3 = new DoubleVec3(1.0f, 2.0f, 3.0f);
        DoubleVec4 vec4 = new DoubleVec4(1.0f, 2.0f, 3.0f, 4.0f);

        assert vec3.get(0) == vec3.x();
        assert vec3.get(2) != vec4.get(3);

        assert vec3 == vec3;
    }

}
