"""IMU pre-integration using the mid-point rule (mirrors integration_base.h)."""
import numpy as np
from ..utility import skew, quat_mul, quat_to_rot, small_angle_quat, normalize_quat


class IntegrationBase:
    def __init__(self, acc_0: np.ndarray, gyr_0: np.ndarray,
                 linearized_ba: np.ndarray, linearized_bg: np.ndarray,
                 acc_n: float, gyr_n: float, acc_w: float, gyr_w: float):
        self.acc_0 = acc_0.copy()
        self.gyr_0 = gyr_0.copy()
        self.linearized_ba = linearized_ba.copy()
        self.linearized_bg = linearized_bg.copy()
        self.ACC_N = acc_n
        self.GYR_N = gyr_n
        self.ACC_W = acc_w
        self.GYR_W = gyr_w

        self.delta_p = np.zeros(3)
        self.delta_q = np.array([1.0, 0.0, 0.0, 0.0])  # [w,x,y,z]
        self.delta_v = np.zeros(3)

        self.jacobian = np.eye(15)   # d(residual)/d(bias)
        self.covariance = np.zeros((15, 15))

        self.sum_dt = 0.0
        self.dt_buf: list[float] = []
        self.acc_buf: list[np.ndarray] = []
        self.gyr_buf: list[np.ndarray] = []

        # Noise matrix (18x18: measurement noise for acc/gyr on both ends + bias walk)
        an2 = acc_n**2
        gn2 = gyr_n**2
        aw2 = acc_w**2
        gw2 = gyr_w**2
        self._Q = np.zeros((18, 18))
        self._Q[0:3, 0:3]   = an2 * np.eye(3)
        self._Q[3:6, 3:6]   = gn2 * np.eye(3)
        self._Q[6:9, 6:9]   = an2 * np.eye(3)
        self._Q[9:12, 9:12] = gn2 * np.eye(3)
        self._Q[12:15,12:15]= aw2 * np.eye(3)
        self._Q[15:18,15:18]= gw2 * np.eye(3)

    def push_back(self, dt: float, acc: np.ndarray, gyr: np.ndarray) -> None:
        self.dt_buf.append(dt)
        self.acc_buf.append(acc.copy())
        self.gyr_buf.append(gyr.copy())
        self._propagate(dt, acc, gyr)

    def repropagate(self, ba: np.ndarray, bg: np.ndarray) -> None:
        self.linearized_ba = ba.copy()
        self.linearized_bg = bg.copy()
        self.delta_p = np.zeros(3)
        self.delta_q = np.array([1.0, 0.0, 0.0, 0.0])
        self.delta_v = np.zeros(3)
        self.jacobian = np.eye(15)
        self.covariance = np.zeros((15, 15))
        self.sum_dt = 0.0
        acc_0 = self.acc_buf[0] if self.acc_buf else self.acc_0
        gyr_0 = self.gyr_buf[0] if self.gyr_buf else self.gyr_0
        self.acc_0 = acc_0.copy()
        self.gyr_0 = gyr_0.copy()
        for dt, acc, gyr in zip(self.dt_buf, self.acc_buf, self.gyr_buf):
            self._propagate(dt, acc, gyr)

    def _propagate(self, dt: float, acc_1: np.ndarray, gyr_1: np.ndarray) -> None:
        ba = self.linearized_ba
        bg = self.linearized_bg

        acc_0 = self.acc_0 - ba
        gyr_0 = self.gyr_0 - bg
        acc_1c = acc_1 - ba
        gyr_1c = gyr_1 - bg

        # Mid-point integration
        gyr_mid = 0.5 * (gyr_0 + gyr_1c)
        dq = small_angle_quat(gyr_mid * dt)
        R0 = quat_to_rot(self.delta_q)
        new_q = normalize_quat(quat_mul(self.delta_q, dq))
        R1 = quat_to_rot(new_q)
        acc_mid = 0.5 * (R0 @ acc_0 + R1 @ acc_1c)

        new_p = self.delta_p + self.delta_v * dt + 0.5 * acc_mid * dt**2
        new_v = self.delta_v + acc_mid * dt

        # Jacobian propagation (15-state: dp,dq,dv,dba,dbg)
        F = np.zeros((15, 15))
        F[0:3, 0:3] = np.eye(3)
        F[0:3, 3:6] = -0.25 * (R0 @ skew(acc_0) + R1 @ skew(acc_1c)) * dt**2
        F[0:3, 6:9] = np.eye(3) * dt
        F[0:3, 9:12]= -0.25 * (R0 + R1) * dt**2
        F[0:3,12:15]= 0.25 * (-R0 @ skew(acc_0) + R1 @ skew(acc_1c)) * dt**3
        F[3:6, 3:6] = so3_exp_mat(-gyr_mid * dt)
        F[3:6,12:15]= -np.eye(3) * dt
        F[6:9, 3:6] = -0.5 * (R0 @ skew(acc_0) + R1 @ skew(acc_1c)) * dt
        F[6:9, 6:9] = np.eye(3)
        F[6:9, 9:12]= -0.5 * (R0 + R1) * dt
        F[6:9,12:15]= 0.5 * (-R0 @ skew(acc_0) + R1 @ skew(acc_1c)) * dt**2
        F[9:12, 9:12] = np.eye(3)
        F[12:15,12:15]= np.eye(3)

        V = np.zeros((15, 18))
        V[0:3, 0:3]  = 0.25 * R0 * dt**2
        V[0:3, 3:6]  = 0.25 * (-R1 @ skew(acc_1c)) * dt**3
        V[0:3, 6:9]  = 0.25 * R1 * dt**2
        V[0:3, 9:12] = V[0:3, 3:6]
        V[3:6, 3:6]  = 0.5 * np.eye(3) * dt
        V[3:6, 9:12] = 0.5 * np.eye(3) * dt
        V[6:9, 0:3]  = 0.5 * R0 * dt
        V[6:9, 3:6]  = 0.5 * (-R1 @ skew(acc_1c)) * dt**2
        V[6:9, 6:9]  = 0.5 * R1 * dt
        V[6:9, 9:12] = V[6:9, 3:6]
        V[12:15,12:15]= np.eye(3) * dt
        V[9:12, 15:18]= np.eye(3) * dt  # gyr_w on bg

        self.jacobian = F @ self.jacobian
        self.covariance = F @ self.covariance @ F.T + V @ self._Q @ V.T

        self.delta_p = new_p
        self.delta_q = new_q
        self.delta_v = new_v
        self.sum_dt += dt
        self.acc_0 = acc_1.copy()
        self.gyr_0 = gyr_1.copy()

    def evaluate(self, Pi: np.ndarray, Qi: np.ndarray, Vi: np.ndarray,
                 Bai: np.ndarray, Bgi: np.ndarray,
                 Pj: np.ndarray, Qj: np.ndarray, Vj: np.ndarray,
                 Baj: np.ndarray, Bgj: np.ndarray,
                 gravity: np.ndarray) -> np.ndarray:
        """Compute 15-dimensional IMU residual."""
        from ..utility import quat_inv, quat_to_rot, quat_mul
        dba = Bai - self.linearized_ba
        dbg = Bgj - self.linearized_bg
        J = self.jacobian

        dp = self.delta_p + J[0:3, 9:12] @ dba + J[0:3, 12:15] @ dbg
        dq_vec = self.delta_q.copy()
        dq_vec[1:4] += 0.5 * J[3:6, 12:15] @ dbg
        dq_vec = normalize_quat(dq_vec)
        dv = self.delta_v + J[6:9, 9:12] @ dba + J[6:9, 12:15] @ dbg

        Ri = quat_to_rot(Qi)
        Rj = quat_to_rot(Qj)

        res_p = Ri.T @ (Pj - Pi - Vi*self.sum_dt + 0.5*gravity*self.sum_dt**2) - dp
        q_ij = quat_mul(quat_inv(Qi), Qj)
        dq_inv = quat_inv(dq_vec)
        res_q = 2.0 * quat_mul(dq_inv, q_ij)[1:4]
        res_v = Ri.T @ (Vj - Vi + gravity*self.sum_dt) - dv
        res_ba = Baj - Bai
        res_bg = Bgj - Bgi
        return np.concatenate([res_p, res_q, res_v, res_ba, res_bg])


def so3_exp_mat(phi: np.ndarray) -> np.ndarray:
    """SO(3) exponential map returning a 3x3 matrix."""
    theta = np.linalg.norm(phi)
    if theta < 1e-10:
        return np.eye(3) + skew(phi)
    K = skew(phi / theta)
    return np.eye(3) + np.sin(theta)*K + (1 - np.cos(theta))*(K @ K)
