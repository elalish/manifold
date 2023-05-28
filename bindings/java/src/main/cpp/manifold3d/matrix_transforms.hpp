#include <glm/glm.hpp>

namespace MatrixTransforms {

glm::vec3 rodrigues_rotation(const glm::vec3& v, const glm::vec3& k, float a) {
    return v * glm::cos(a) + glm::cross(k, v) * glm::sin(a) + k * glm::dot(k, v) * (1 - glm::cos(a));
}

glm::mat4x3 Yaw(const glm::mat4x3& m, float a) {
    glm::vec3 d[] = {
        rodrigues_rotation(m[0], m[1], a),
        m[1],
        rodrigues_rotation(m[2], m[1], a)
    };
    return glm::mat4x3(d[0], d[1], d[2], m[3]);
}

glm::mat4x3 Pitch(const glm::mat4x3& m, float a) {
    glm::vec3 d[] = {
        m[0],
        rodrigues_rotation(m[1], m[0], a),
        rodrigues_rotation(m[2], m[0], a)
    };
    return glm::mat4x3(d[0], d[1], d[2], m[3]);
}

glm::mat4x3 Roll(const glm::mat4x3& m, float a) {
    glm::vec3 d[] = {
        rodrigues_rotation(m[0], m[2], a),
        rodrigues_rotation(m[1], m[2], a),
        m[2]
    };
    return glm::mat4x3(d[0], d[1], d[2], m[3]);
}

glm::mat4x3 Rotate(const glm::mat4x3& m, const glm::vec3& axis, float a) {
    glm::vec3 d[] = {
        rodrigues_rotation(m[0], axis, a),
        rodrigues_rotation(m[1], axis, a),
        rodrigues_rotation(m[2], axis, a)
    };
    return glm::mat4x3(d[0], d[1], d[2], m[3]);
}

glm::mat4x3 Rotate(const glm::mat4x3& m, const glm::vec3& angles) {
    glm::mat4x3 res = m;
    if (angles[0] != 0) {
        res = Pitch(res, angles[0]);
    }
    if (angles[1] != 0) {
        res = Yaw(res, angles[1]);
    }
    if (angles[2] != 0) {
        res = Roll(res, angles[2]);
    }
    return res;
}

glm::mat4x3 SetRotation(const glm::mat4x3& m, const glm::mat3x3& rotation) {
    glm::mat4x3 result = m;
    for (int i = 0; i < 3; ++i) {
        result[i] = rotation[i];
    }
    return result;
}

glm::mat4x3 Translate(const glm::mat4x3& m, const glm::vec3& offset) {
    glm::mat4x3 result = m;

    if (offset[0] != 0.0) {
        result[3] += m[0]*offset[0];
    }
    if (offset[1] != 0.0) {
        result[3] += m[1]*offset[1];
    }
    if (offset[3] != 0.0) {
        result[3] += m[2]*offset[2];
    }

    return result;
}

glm::mat4x3 SetTranslation(const glm::mat4x3& m, const glm::vec3& translation) {
    glm::mat4x3 result = m;
    result[3] = translation;
    return result;
}

glm::mat4x3 Transform(const glm::mat4x3& a, const glm::mat4x3& b) {
    glm::mat4x3 result;

    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 4; ++col) {
            result[row][col] = a[row][0] * b[0][col] + a[row][1] * b[1][col] + a[row][2] * b[2][col];
            if (col == 3) {
                result[row][col] += a[row][3];
            }
        }
    }

    for (int i = 0; i < 3; ++i) {
        glm::vec3 colVector = glm::vec3(result[i]);
        result[i] = glm::normalize(colVector);
    }

    return result;
}

}
