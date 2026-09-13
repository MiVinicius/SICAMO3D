"""
Filtro de Kalman 3D para rastreamento suave de posição e velocidade no espaço real.
"""
import numpy as np
from typing import Tuple

class KalmanFilter3D:
    def __init__(self, init_pos: Tuple[float, float, float], dt: float = 1.0 / 30.0):
        self.dt = dt
        # Estado: [X, Y, Z, Vx, Vy, Vz]
        self.x = np.array([init_pos[0], init_pos[1], init_pos[2], 0.0, 0.0, 0.0], dtype=np.float32)

        # Matriz de transição de estado F
        self.F = np.array([
            [1.0, 0.0, 0.0, dt,  0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, dt,  0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, dt ],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)

        # Matriz de medição H (medimos apenas X, Y, Z)
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        ], dtype=np.float32)

        # Covariância de erro de estado P
        self.P = np.eye(6, dtype=np.float32) * 0.1
        self.P[3:, 3:] *= 1.0

        # Ruído de processo Q (menor ruído de posição para trajetórias suaves sem jitter)
        q_pos = 0.005
        q_vel = 0.05
        self.Q = np.diag([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel]).astype(np.float32)

        # Ruído de medição R (filtra flutuações do sensor Kinect v2)
        self.R = np.eye(3, dtype=np.float32) * 0.06

    def predict(self) -> np.ndarray:
        """Prediz o próximo estado com base na velocidade estimada."""
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x[:3]

    def update(self, measurement: Tuple[float, float, float]):
        """Corrige o estado com uma nova medição observada."""
        z = np.array(measurement, dtype=np.float32)
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
        """Retorna a velocidade escalar (m/s)."""
        return float(np.sqrt(self.x[3]**2 + self.x[4]**2 + self.x[5]**2))
