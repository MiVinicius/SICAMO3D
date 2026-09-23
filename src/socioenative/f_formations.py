"""
Módulo de Detecção de F-Formations (Facing Formations) e Espaço Transacional (o-space).
Fundamentação teórica:
- Kendon, A. (1990). "Conducting Interaction: Patterns of behavior in focused encounters."
- Cristani, M., Bazzani, L., Paggetti, G., Vinciarelli, A., & Murino, V. (2011). 
  "Socially focused: predicting social interaction from spatial and behavioral patterns."
- Setti, F., Russell, C., Bassetti, C., & Cristani, M. (2015). "F-formation detection."

Identifica arranjos espaciais de interação social diádica e grupal:
- Diádicos: vis-a-vis (face a face), l-shape (perpendicular em L), side-by-side (lado a lado).
- Grupais (>= 3 pessoas): circular (fechado ao redor do o-space) e semicircular (aberto).
- Aferição de inclusão do artefato (pelúcia) no o-space da formação.
"""

import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

@dataclass
class FFormationGroup:
    group_id: int
    members: List[int]
    formation_type: str  # 'vis-a-vis', 'l-shape', 'side-by-side', 'circular', 'semicircular'
    o_space_center: Tuple[float, float]  # (X, Z) no chão da sala
    o_space_radius: float  # Raio estimado do o-space em metros
    confidence: float
    plush_inside_o_space: bool
    mean_interpersonal_dist_m: float

class FFormationDetector:
    def __init__(self,
                 max_interpersonal_dist_m: float = 2.20,
                 focal_dist_m: float = 0.70,
                 max_center_dist_m: float = 0.85):
        """
        max_interpersonal_dist_m: distância física máxima entre dois membros para pertencerem à mesma formação (p-space).
        focal_dist_m: distância projetada para frente da pessoa onde se concentra o foco de atenção/transação.
        max_center_dist_m: distância máxima entre os focos projetados para considerar convergência no o-space.
        """
        self.max_interpersonal_dist_m = max_interpersonal_dist_m
        self.focal_dist_m = focal_dist_m
        self.max_center_dist_m = max_center_dist_m

    def detect(self, tracks: List, plush_pos: Optional[Tuple[float, float, float]] = None) -> List[FFormationGroup]:
        """
        Detecta grupos de F-formations a partir de uma lista de Track3D no frame atual.
        tracks: lista de Track3D
        plush_pos: tupla (x, y, z) com posição 3D da pelúcia (ou None)
        """
        # Filtra tracks válidos com posição mensurável
        valid_tracks = []
        for t in tracks:
            pos = getattr(t, "position", None)
            if pos is not None and not np.isnan(pos[0]) and not np.isnan(pos[2]):
                valid_tracks.append(t)

        n = len(valid_tracks)
        if n < 2:
            return []

        # Extrai posições 2D (X, Z) e vetores de orientação (heading)
        positions = []
        headings = []
        focal_points = []

        for trk in valid_tracks:
            px, _, pz = trk.position
            positions.append(np.array([px, pz], dtype=np.float32))

            # Obtém heading em graus [-180, 180]
            # Convenção dextra: 0° = +Z (fundo), 90° = +X (direita)
            deg = getattr(trk, "heading_deg", None)
            if deg is None:
                # Se não puder estimar heading, assume vetor nulo
                h_vec = np.zeros(2, dtype=np.float32)
            else:
                rad = math.radians(deg)
                # Vetor unitário apontando na direção do peito
                h_vec = np.array([math.sin(rad), math.cos(rad)], dtype=np.float32)

            headings.append(h_vec)
            # Ponto focal transacional projetado à frente da pessoa
            focal_points.append(positions[-1] + self.focal_dist_m * h_vec)

        # Grafo de adjacência de interação social
        adj: Dict[int, List[int]] = {i: [] for i in range(n)}

        for i in range(n):
            for j in range(i + 1, n):
                p_i, p_j = positions[i], positions[j]
                h_i, h_j = headings[i], headings[j]
                c_i, c_j = focal_points[i], focal_points[j]

                dist_ij = float(np.linalg.norm(p_i - p_j))
                if dist_ij > self.max_interpersonal_dist_m or dist_ij < 0.20:
                    continue

                # Vetor relativo unitário de i para j
                v_ij = (p_j - p_i) / dist_ij
                v_ji = -v_ij

                # Dot products com as direções de olhar
                has_h_i = np.linalg.norm(h_i) > 0.5
                has_h_j = np.linalg.norm(h_j) > 0.5

                # Se ambas as pessoas estiverem olhando para trás (de costas uma para a outra), exclui
                if has_h_i and has_h_j:
                    dot_i = float(np.dot(h_i, v_ij))
                    dot_j = float(np.dot(h_j, v_ji))
                    # De costas: ambos olhando na direção oposta ao parceiro
                    if dot_i < -0.30 and dot_j < -0.30:
                        continue

                # Teste 1: Distância entre os centros focais (convergência de o-space)
                dist_centers = float(np.linalg.norm(c_i - c_j))
                focal_converges = dist_centers <= self.max_center_dist_m

                # Teste 2: Facing mútuo (frente a frente - vis-a-vis)
                mutual_facing = False
                if has_h_i and has_h_j:
                    dot_i = float(np.dot(h_i, v_ij))
                    dot_j = float(np.dot(h_j, v_ji))
                    if dot_i > 0.25 and dot_j > 0.25:
                        mutual_facing = True

                # Teste 3: Lado a lado (side-by-side)
                side_by_side = False
                if has_h_i and has_h_j and dist_ij <= 1.30:
                    dot_hh = float(np.dot(h_i, h_j))
                    # Quase paralelos e olhando na mesma direção perpendicular
                    if dot_hh >= 0.65:
                        # Nem i olhando direto para j, nem de costas
                        if abs(float(np.dot(h_i, v_ij))) < 0.70:
                            side_by_side = True

                if focal_converges or mutual_facing or side_by_side:
                    adj[i].append(j)
                    adj[j].append(i)

        # Encontra componentes conexos
        visited = set()
        components = []

        for i in range(n):
            if i not in visited:
                comp = []
                queue = [i]
                visited.add(i)
                while queue:
                    curr = queue.pop(0)
                    comp.append(curr)
                    for neighbor in adj[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                if len(comp) >= 2:
                    components.append(comp)

        # Caracteriza cada formação encontrada
        groups: List[FFormationGroup] = []
        for g_idx, comp in enumerate(components):
            member_ids = [valid_tracks[idx].track_id for idx in comp]
            comp_positions = [positions[idx] for idx in comp]
            comp_focals = [focal_points[idx] for idx in comp]
            comp_headings = [headings[idx] for idx in comp]

            # Distância interpessoal média no grupo
            dists = []
            for a in range(len(comp)):
                for b in range(a + 1, len(comp)):
                    dists.append(float(np.linalg.norm(comp_positions[a] - comp_positions[b])))
            mean_dist = float(np.mean(dists)) if dists else 1.0

            # Centro do o-space: média dos pontos focais dos membros
            center_x = float(np.mean([pt[0] for pt in comp_focals]))
            center_z = float(np.mean([pt[1] for pt in comp_focals]))
            o_center = (round(center_x, 3), round(center_z, 3))

            # Raio do o-space: dispersão dos focos mais margem anatômica
            focal_dists = [float(np.linalg.norm(pt - np.array([center_x, center_z]))) for pt in comp_focals]
            radius = float(np.clip(max(focal_dists) + 0.25, 0.35, 1.40))

            # Classificação da formação
            num_members = len(comp)
            formation_type = "desconhecida"

            if num_members == 2:
                h1, h2 = comp_headings[0], comp_headings[1]
                dot_h = float(np.dot(h1, h2))
                if dot_h <= -0.50:
                    formation_type = "vis-a-vis"
                elif dot_h >= 0.60:
                    formation_type = "side-by-side"
                else:
                    formation_type = "l-shape"
            else:
                # 3 ou mais membros: analisa distribuição angular ao redor do o-center
                angles = []
                for pt in comp_positions:
                    ang = math.atan2(pt[0] - center_x, pt[1] - center_z)
                    angles.append(ang)
                angles.sort()
                # Calcula maior gap angular entre membros vizinhos
                diffs = []
                for k in range(len(angles)):
                    next_k = (k + 1) % len(angles)
                    diff = (angles[next_k] - angles[k]) % (2 * math.pi)
                    diffs.append(diff)
                max_gap_deg = math.degrees(max(diffs))
                angular_span_deg = 360.0 - max_gap_deg
                if angular_span_deg > 220.0:
                    formation_type = "circular"
                else:
                    formation_type = "semicircular"

            # Verifica se a pelúcia está dentro do o-space
            plush_in = False
            if plush_pos is not None and not np.isnan(plush_pos[0]) and not np.isnan(plush_pos[2]):
                d_plush = math.hypot(plush_pos[0] - center_x, plush_pos[2] - center_z)
                if d_plush <= radius + 0.25:
                    plush_in = True

            conf = 0.90 if num_members == 2 else 0.85

            groups.append(FFormationGroup(
                group_id=g_idx + 1,
                members=member_ids,
                formation_type=formation_type,
                o_space_center=o_center,
                o_space_radius=round(radius, 3),
                confidence=conf,
                plush_inside_o_space=plush_in,
                mean_interpersonal_dist_m=round(mean_dist, 3)
            ))

        return groups
