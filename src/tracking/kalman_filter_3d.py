"""
Filtro de Kalman 3D para rastreamento suave de posição e velocidade no espaço real.
"""
import numpy as np
from typing import Tuple, Optional

class KalmanFilter3D:
    def __init__(self, init_pos: Tuple[float, float, float], dt: float = 1.0 / 30.0):
        self.dt = dt
        # Estado: [X, Y, Z, Vx, Vy, Vz]
        self.x = np.array([init_pos[0], init_pos[1], init_pos[2], 0.0, 0.0, 0.0], dtype=np.float32)

        # Matriz de medição H (medimos apenas X, Y, Z)
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        ], dtype=np.float32)

        # Covariância de erro de estado P
        self.P = np.eye(6, dtype=np.float32) * 0.05
        self.P[3:, 3:] *= 0.5

        # Ruído de processo Q calibrado:
        # q_pos menor para estabilidade; q_vel moderado para acompanhar aceleração natural de crianças/atores
        q_pos = 0.002
        q_vel = 0.08
        self.Q = np.diag([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel]).astype(np.float32)

        # Ruído de medição R calibrado:
        # R=0.0025 equivale a desvio padrão sigma de ~5cm (realista para keypoints anatômicos 3D do Kinect v2)
        # Elimina o atraso severo ao andar que ocorria com R=0.06 (sigma 25cm)
        self.R = np.eye(3, dtype=np.float32) * 0.0025

    def _get_transition_matrix(self, dt: float) -> np.ndarray:
        return np.array([
            [1.0, 0.0, 0.0, dt,  0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, dt,  0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, dt ],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)

    def predict(self, dt: Optional[float] = None) -> np.ndarray:
        """Prediz o próximo estado com base na velocidade estimada e dt real."""
        step_dt = dt if (dt is not None and 0.001 <= dt <= 0.5) else self.dt
        F = self._get_transition_matrix(step_dt)
        self.x = np.dot(F, self.x)
        self.P = np.dot(np.dot(F, self.P), F.T) + self.Q
        return self.x[:3]

    def update(self, measurement: Tuple[float, float, float]):
        """Corrige o estado com uma nova medição observada."""
        z = np.array(measurement, dtype=np.float32)
        if np.isnan(z).any():
            return # Não atualiza medição com NaNs

        y = z - np.dot(self.H, self.x) # Inovação
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R # Covariância de inovação
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S)) # Ganho de Kalman

        self.x = self.x + np.dot(K, y)
        I = np.eye(6, dtype=np.float32)
        self.P = np.dot(I - np.dot(K, self.H), self.P)

    @property
    def position(self) -> Tuple[float, float, float]:
        return (float(self.x[0]), float(self.x[1]), float(self.x[2]))

    @property
    def velocity(self) -> Tuple[float, float, float]:
        return (float(self.x[3]), float(self.x[4]), float(self.x[5]))

    @property
    def speed(self) -> float:
        """Retorna a velocidade escalar 3D (m/s)."""
        return float(np.sqrt(self.x[3]**2 + self.x[4]**2 + self.x[5]**2))

    @property
    def ground_speed(self) -> float:
        """Retorna a velocidade no plano do chão X-Z da sala (m/s), imune a oscilações verticais."""
        return float(np.sqrt(self.x[3]**2 + self.x[5]**2))

