package manifold3d;

import org.junit.Test;
import manifold3d.Glm.FloatVec3;
import manifold3d.Glm.FloatVec4;

public class FloatVectorTest {

    public FloatVectorTest() {
    }

    @Test
    public void testMain() {
        FloatVec3 vec3 = new FloatVec3(1.0f, 2.0f, 3.0f);
        FloatVec4 vec4 = new FloatVec4(1.0f, 2.0f, 3.0f, 4.0f);

        assert vec3.get(0) == vec3.x();
        assert vec3.get(2) != vec4.get(3);

        assert vec3 == vec3;
    }

}
