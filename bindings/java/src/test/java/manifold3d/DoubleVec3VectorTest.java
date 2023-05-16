package manifold3d;

import org.junit.Test;
import org.junit.Assert;
import manifold3d.glm.DoubleVec3;
import manifold3d.glm.DoubleVec3Vector;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;

public class DoubleVec3VectorTest {

    public DoubleVec3VectorTest() {
    }

    @Test
    public void testMain() {

        double[] data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        ByteBuffer buff = ByteBuffer.allocateDirect(6 * Double.BYTES).order(ByteOrder.nativeOrder());
        DoubleBuffer doubleBuffer = buff.asDoubleBuffer();
        doubleBuffer.put(data);
        doubleBuffer.flip();

        DoubleVec3Vector vec = DoubleVec3Vector.FromBuffer(doubleBuffer);

        Assert.assertEquals(vec.get(1).get(1), 5.0, 0.0001);
    }

}
