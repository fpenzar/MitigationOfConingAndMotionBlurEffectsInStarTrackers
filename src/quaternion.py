from __future__ import annotations

import numpy as np


class Quaternion:

    def __init__(self, q1, q2, q3, q4=0):
        self.quat = np.array([q1, q2, q3, q4], dtype=float)

    @property
    def vec(self):
        return self.quat[:3]
    
    @property
    def w(self):
        return self.quat[3]


    def normalize(self):
        """
        FLOP complexity: 6 mul, 2 add, 1 div, 1 sqrt
        """
        norm = self.norm()
        if norm == 0:
            raise ValueError("Cannot normalize a zero quaternion")
        self.quat = self.quat / norm
    
    def norm(self):
        return np.linalg.norm(self.quat)    

    def __mul__(self, other):
        """
        Reference: Quaternion to Euler angles conversion - A direct, general and computationally efficient method

        FLOP complexity: 16 mul, 16 add
        """
        if isinstance(other, np.ndarray):
            return self * Quaternion(*(other))

        if isinstance(other, float) or isinstance(other, int):
            return Quaternion(*(other * self.quat))
        
        if not isinstance(other, Quaternion):
            return NotImplemented

        p_13, p_4 = self.quat[:3], self.quat[3]
        q_13, q_4 = other.quat[:3], other.quat[3]

        vec = q_4 * p_13 + p_4 * q_13 + np.cross(p_13, q_13)
        w   = p_4 * q_4 - np.dot(p_13, q_13)

        return Quaternion(*vec, w)
    

    def __rmul__(self, other):
        if isinstance(other, np.ndarray):
            return Quaternion(*(other)) * self
        if isinstance(other, Quaternion):
            return other * self
        return NotImplemented
    

    def conjugate(self):
        return Quaternion(*(-self.quat[0:3]), self.quat[3])
    

    def __add__(self, other):
        if isinstance(other, Quaternion):
            return Quaternion(*(self.quat + other.quat))
        else:
            return NotImplemented
    

    def __str__(self):
        return str(self.quat)
    
    def __repr__(self):
        return self.__str__()
    
    def rotation_angle(self, degrees: bool = False):
        qw = np.clip(self.w / self.norm(), -1.0, 1.0)
        theta = 2.0 * np.arccos(qw)
        if degrees:
            theta = np.degrees(theta)
        return theta
    
    @classmethod
    def I(cls):
        return Quaternion(0, 0, 0, 1)
    
    @classmethod
    def rotate(cls, v: np.ndarray, *q):
        """
        Chain the rotations. The rotations are intrisic, where the first one that occured
        is the first provided argument after the vector.
        """
        rotation_q = Quaternion.I()
        for qn in q:
            rotation_q = Quaternion.chain_rotations(rotation_q, qn)
        return (rotation_q * v * rotation_q.conjugate()).vec
    
    @classmethod
    def chain_rotations(cls, q1: Quaternion, q2: Quaternion):
        """
        First q1 rotation occured, then q2.
        The rotations were measured in the body frame (intrinsic measurements).

        This is why the order needs to be flipped to get the extrinsic rotation 
        which produces the total rotation (i.e. it is not q2 * q1 * v * q1.conjugate * q2.conjugate
        but rather q1 * q2 * v * q2.conjugate * q1.conjugate).
        """
        q1.normalize()
        q2.normalize()
        rotation_q = q1 * q2
        rotation_q.normalize()
        return rotation_q
    
    def to_dcm(self) -> np.ndarray:
        """
        Quaternion -> Direction Cosine Matrix (active rotation).
        Returns R such that v_out = R @ v_in.

        Reference: Quaternion to Euler angles conversion - A direct, general and computationally efficient method
        """
        qx, qy, qz, qw = self.quat / np.linalg.norm(self.quat)
        xx, yy, zz, rr = qx*qx, qy*qy, qz*qz, qw*qw
        xy, xz, yz = qx*qy, qx*qz, qy*qz
        rx, ry, rz = qw*qx, qw*qy, qw*qz

        R = np.array([
            [rr + xx - yy - zz,     2*(xy - rz),         2*(xz + ry)],
            [2*(xy + rz),         rr - xx + yy - zz,     2*(yz - rx)],
            [2*(xz - ry),         2*(yz + rx),         rr - xx - yy + zz]
        ], dtype=float)
        return R
    

    def to_euler(self, degrees: bool = False):
        """
        Quaternion -> Euler angles (ZYX / yaw-pitch-roll).
        Returns (roll, pitch, yaw) = (phi, theta, psi).
        (apply yaw first, roll last)

        Reference: A tutorial on SE(3) transformation parameterizations and on-manifold optimization
        """
        qx, qy, qz, qw = self.quat / np.linalg.norm(self.quat)

        # roll (x-axis)
        sinr_cosp = 2*(qw*qx + qy*qz)
        cosr_cosp = 1 - 2*(qx*qx + qy*qy)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # pitch (y-axis)
        sinp = 2*(qw*qy - qz*qx)
        sinp = np.clip(sinp, -1.0, 1.0)  # clamp for numerical safety
        pitch = np.arcsin(sinp)

        # yaw (z-axis)
        siny_cosp = 2*(qw*qz + qx*qy)
        cosy_cosp = 1 - 2*(qy*qy + qz*qz)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        if degrees:
            return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)
        return roll, pitch, yaw
    

    @classmethod
    def from_dcm(cls, R: np.ndarray):
        """
        Direction Cosine Matrix -> Quaternion (vector-first, scalar-last).
        Implements Shepperd's eq. sets (15)-(18) + Algorithm 1 selection

        The result is further conjugated, since Motekew is desrcibing rotations as
        q.conjugate() * v * q instead of q * v * q.conjugate()
        -> the order of quaternion operations is read from left to right in his example

        Reference: Motekew, "Quaternion to DCM and Back Again", §3, eqs (5)-(18), Alg. 1.
        """
        R = np.asarray(R, dtype=float)
        c11, c12, c13 = R[0,0], R[0,1], R[0,2]
        c21, c22, c23 = R[1,0], R[1,1], R[1,2]
        c31, c32, c33 = R[2,0], R[2,1], R[2,2]
        tr = c11 + c22 + c33

        def ssqrt(x):
            return np.sqrt(x if x > 0.0 else 0.0)

        if (tr > c11) and (tr > c22) and (tr > c33):
            qs = ssqrt((1.0 + tr) / 4.0)
            qi = (c23 - c32) / (4.0 * qs)
            qj = (c31 - c13) / (4.0 * qs)
            qk = (c12 - c21) / (4.0 * qs)
        elif (c11 > c22) and (c11 > c33):
            qi = ssqrt((1.0 + c11 - c22 - c33) / 4.0)
            qs = (c23 - c32) / (4.0 * qi)
            qj = (c12 + c21) / (4.0 * qi)
            qk = (c31 + c13) / (4.0 * qi)
        elif (c22 > c33):
            qj = ssqrt((1.0 - c11 + c22 - c33) / 4.0)
            qs = (c31 - c13) / (4.0 * qj)
            qi = (c12 + c21) / (4.0 * qj)
            qk = (c23 + c32) / (4.0 * qj)
        else:
            qk = ssqrt((1.0 - c11 - c22 + c33) / 4.0)
            qs = (c12 - c21) / (4.0 * qk)
            qi = (c31 + c13) / (4.0 * qk)
            qj = (c23 + c32) / (4.0 * qk)

        # Paper uses scalar-first (qs, qi, qj, qk). This class is vector-first, scalar-last
        # Conjugate (reverse the rotation), since this reference employs the opposite order of quaternion
        # multiplicatoin
        q = cls(qi, qj, qk, qs).conjugate()
        q.normalize()
        return q


    @classmethod
    def from_euler(cls, roll: float, pitch: float, yaw: float, degrees: bool = False):
        """
        Euler angles (ZYX / yaw-pitch-roll) -> Quaternion.
        Input: roll (phi), pitch (theta), yaw (psi).
        (yaw first, roll last)

        Reference: NASA - Euler Angles, Quaternions and Transformation Matrices
        """
        if degrees:
            roll, pitch, yaw = np.radians([roll, pitch, yaw])

        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)

        # ZYX = Rz(yaw) * Ry(pitch) * Rx(roll)
        qw = cr*cp*cy + sr*sp*sy
        qx = sr*cp*cy - cr*sp*sy
        qy = cr*sp*cy + sr*cp*sy
        qz = cr*cp*sy - sr*sp*cy

        q = cls(qx, qy, qz, qw)
        q.normalize()
        return q

    @classmethod
    def from_rotation_vector(cls, rotation_vector):
        # FLOP complexity: 10 mul, 2 add, 1 div, 1 sqrt, 1 sin, 1 cos
        theta = np.linalg.norm(rotation_vector)

        if theta < 1e-12:
            dq = Quaternion(*(0.5 * rotation_vector), 1.0)
        else:
            inv_theta = 1 / theta
            axis = rotation_vector * inv_theta
            half = 0.5 * theta
            s = np.sin(half)
            c = np.cos(half)
            dq = Quaternion(*(axis * s), c)
        
        return dq

    @classmethod
    def error(cls, q_est: Quaternion, q_true: Quaternion):
        """
        Reference: Fundamentals of Spececraft Attitude Determination and Control, Appendix 5
        Taken as the formula for estimation theory
        """
        q_err = Quaternion.chain_rotations(q_est.conjugate(), q_true)
        if q_err.w < 0.0:
            q_err = Quaternion(*(-q_err.vec), -q_err.w)
        q_err.normalize()
        return q_err
    
    @classmethod
    def error_angle(cls, q_est: Quaternion, q_true: Quaternion, degrees=False):
        q_err = Quaternion.error(q_est, q_true)
        return np.abs(q_err.rotation_angle(degrees))


def rad(a):
    return np.radians(a)


def almost(a, b, atol=1e-12):
    return np.allclose(a, b, atol=atol)


def test_identity_rotation():
    v = np.array([1., 2., 3.])
    qI = Quaternion.I()
    v_rot = Quaternion.rotate(v, qI)
    assert almost(v_rot, v)


def test_from_euler():
    angle = 45
    q_yaw = Quaternion.from_euler(0, 0, rad(angle))
    q_target = Quaternion(0, 0, np.sin(rad(angle/2)), np.cos(rad(angle/2)))
    assert almost(q_yaw.quat, q_target.quat)

    q_pitch = Quaternion.from_euler(0, rad(angle), 0)
    q_target = Quaternion(0, np.sin(rad(angle/2)), 0, np.cos(rad(angle/2)))
    assert almost(q_pitch.quat, q_target.quat)

    q_roll = Quaternion.from_euler(rad(angle), 0, 0)
    q_target = Quaternion(np.sin(rad(angle/2)), 0, 0, np.cos(rad(angle/2)))
    assert almost(q_roll.quat, q_target.quat)


def test_to_euler():
    angle = 45
    q = Quaternion(0, 0, np.sin(rad(angle/2)), np.cos(rad(angle/2)))
    euler_q = np.array(q.to_euler(True))
    euler_target = np.array([0, 0, 45])
    assert almost(euler_q, euler_target)

    q = Quaternion(0, np.sin(rad(angle/2)), 0, np.cos(rad(angle/2)))
    euler_q = np.array(q.to_euler(True))
    euler_target = np.array([0, 45, 0])
    assert almost(euler_q, euler_target)

    q = Quaternion(np.sin(rad(angle/2)), 0, 0, np.cos(rad(angle/2)))
    euler_q = np.array(q.to_euler(True))
    euler_target = np.array([45, 0, 0])
    assert almost(euler_q, euler_target)


def test_euler_and_roundtrip():
    angles = np.array([0, 0, 45])
    q = Quaternion.from_euler(*map(rad, angles))
    euler_q = np.array(q.to_euler(True))
    assert almost(angles, euler_q)

    angles = np.array([0, 45, 0])
    q = Quaternion.from_euler(*map(rad, angles))
    euler_q = np.array(q.to_euler(True))
    assert almost(angles, euler_q)

    angles = np.array([45, 0, 0])
    q = Quaternion.from_euler(*map(rad, angles))
    euler_q = np.array(q.to_euler(True))
    assert almost(angles, euler_q)

    angles = np.array([-14, 56, 146])
    q = Quaternion.from_euler(*map(rad, angles))
    euler_q = np.array(q.to_euler(True))
    assert almost(angles, euler_q)


def test_to_dcm():
    angles_deg = (90, 0, 90)
    vx = np.array([1, 0, 0])
    vy = np.array([0, 1, 0])
    vz = np.array([0, 0, 1])

    q = Quaternion.from_euler(*map(rad, angles_deg))
    R = q.to_dcm()

    vx_t = R @ vx
    vy_t = R @ vy
    vz_t = R @ vz

    assert almost(vx_t, np.array([0, 1, 0]))
    assert almost(vy_t, np.array([0, 0, 1]))
    assert almost(vz_t, np.array([1, 0, 0]))


def test_dcm():
    # Euler <-> Quaternion roundtrip (random angles)
    angles = (-2.2, 0.4, -1)

    q = Quaternion.from_euler(*angles)
    angles_rt = q.to_euler()

    # Compare orientations
    q_rt = Quaternion.from_euler(*angles_rt)
    assert (almost(q.quat, q_rt.quat) or almost(q.quat, -q_rt.quat))

    # DCM <-> Quaternion roundtrip
    R = q.to_dcm()
    q2 = Quaternion.from_dcm(R)
    assert (almost(q2.quat, q.quat) or almost(q2.quat, -q.quat))


def test_pure_rotations():
    # Pure yaw (+90° about z): x → y, y → -x, z → z ---
    q_yaw90 = Quaternion.from_euler(0, 0, rad(90))
    ex = np.array([1., 0., 0.])
    ey = np.array([0., 1., 0.])
    ez = np.array([0., 0., 1.])

    assert almost(Quaternion.rotate(ex, q_yaw90), ey)
    assert almost(Quaternion.rotate(ey, q_yaw90), -ex)
    assert almost(Quaternion.rotate(ez, q_yaw90), ez)

    # Pure pitch (+90° about y): z → x, x → -z, y → y ---
    q_pitch90 = Quaternion.from_euler(0, rad(90), 0)
    assert almost(Quaternion.rotate(ex, q_pitch90), -ez)
    assert almost(Quaternion.rotate(ez, q_pitch90), ex)
    assert almost(Quaternion.rotate(ey, q_pitch90), ey)

    # Pure roll (+90° about x): y → z, z → -y, x → x ---
    q_roll90 = Quaternion.from_euler(rad(90), 0, 0)
    assert almost(Quaternion.rotate(ey, q_roll90), ez)
    assert almost(Quaternion.rotate(ez, q_roll90), -ey)
    assert almost(Quaternion.rotate(ex, q_roll90), ex)


def test_simple_rotation_composition():
    # Compose simple angles: yaw 45° twice = yaw 90° ---
    ex = np.array([1., 0., 0.])
    q_yaw90 = Quaternion.from_euler(0, 0, rad(90))
    q_yaw45 = Quaternion.from_euler(0, 0, rad(45))
    v1 = Quaternion.rotate(ex, q_yaw45, q_yaw45)  # two sequential rotations
    v2 = Quaternion.rotate(ex, q_yaw90)
    assert almost(v1, v2)


def test_rotation_decomposition():
    angles_deg = (90, 0, 90)
    q = Quaternion.from_euler(*map(rad, angles_deg))
    R = q.to_dcm()

    v = np.array([1, 0, 0])
    v_quat = Quaternion.rotate(v, q)

    v_comp  = Quaternion.rotate(                 # explicit Z, then Y, then X
        v,
        Quaternion.from_euler(0, 0, rad(angles_deg[2])),
        Quaternion.from_euler(0, rad(angles_deg[1]), 0),
        Quaternion.from_euler(rad(angles_deg[0]), 0, 0),
    )

    q_seq = Quaternion.chain_rotations(Quaternion.from_euler(0, 0, rad(angles_deg[2])), 
                                       Quaternion.from_euler(0, rad(angles_deg[1]), 0))
    q_seq = Quaternion.chain_rotations(q_seq, Quaternion.from_euler(rad(angles_deg[0]), 0, 0))
    v_seq = Quaternion.rotate(v, q_seq)

    v_mat = R @ v

    assert almost(v_quat, v_mat, atol=1e-12)
    assert almost(v_seq, v_mat, atol=1e-12)
    assert almost(v_comp, v_mat, atol=1e-12)

    # Second part
    angles_deg = (30, 20, -40)
    q = Quaternion.from_euler(*map(rad, angles_deg))
    R = q.to_dcm()

    v = np.array([0.7, -0.2, 0.5])
    v_quat = Quaternion.rotate(v, q)           # single combined quaternion
    v_comp  = Quaternion.rotate(                 # explicit Z, then Y, then X
        v,
        Quaternion.from_euler(0, 0, rad(angles_deg[2])),
        Quaternion.from_euler(0, rad(angles_deg[1]), 0),
        Quaternion.from_euler(rad(angles_deg[0]), 0, 0),
    )
    q_seq = Quaternion.chain_rotations(Quaternion.from_euler(0, 0, rad(angles_deg[2])), 
                                       Quaternion.from_euler(0, rad(angles_deg[1]), 0))
    q_seq = Quaternion.chain_rotations(q_seq, Quaternion.from_euler(rad(angles_deg[0]), 0, 0))
    v_seq = Quaternion.rotate(v, q_seq)
    v_mat = R @ v

    assert almost(v_quat, v_mat, atol=1e-12)
    assert almost(v_seq, v_mat, atol=1e-12)
    assert almost(v_comp, v_mat, atol=1e-12)

def test_round_trip_with_random_angles():
    rng = np.random.default_rng(0)
    for _ in range(10):
        roll, pitch, yaw = (rng.uniform(-np.pi, np.pi),
                            rng.uniform(-np.pi/2, np.pi/2),  # keep pitch in principal range
                            rng.uniform(-np.pi, np.pi))
        q = Quaternion.from_euler(roll, pitch, yaw)
        v = rng.normal(size=3)
        vq = Quaternion.rotate(v, q)
        vm = (q.to_dcm() @ v)
        assert almost(vq, vm, atol=1e-12)

def test_successive_rotations():
    v = np.array([0.7, -0.2, 0.5])

    angles_deg_1 = (30, 20, -40)
    q1 = Quaternion.from_euler(*map(rad, angles_deg_1))
    R_1 = q1.to_dcm()

    angles_deg_2 = (15, -18, -115)
    q2 = Quaternion.from_euler(*map(rad, angles_deg_2))
    R_2 = q2.to_dcm()

    angles_deg_3 = (-300, 212, 175)
    q3 = Quaternion.from_euler(*map(rad, angles_deg_3))
    R_3 = q3.to_dcm()

    q_seq = Quaternion.chain_rotations(q1, q2)
    q_seq = Quaternion.chain_rotations(q_seq, q3)
    v_seq = Quaternion.rotate(v, q_seq)

    v_mat = R_1 @ R_2 @ R_3 @ v

    v_comp  = Quaternion.rotate(v, q1, q2, q3,)

    assert almost(v_seq, v_mat, atol=1e-12)
    assert almost(v_comp, v_mat, atol=1e-12)


if __name__ == "__main__":

    test_identity_rotation()
    test_from_euler()
    test_to_euler()
    test_euler_and_roundtrip()
    test_to_dcm()
    test_dcm()

    test_pure_rotations()
    test_simple_rotation_composition()
    test_rotation_decomposition()
    test_round_trip_with_random_angles()
    test_successive_rotations()

    print("All quaternion tests passed ✅")
