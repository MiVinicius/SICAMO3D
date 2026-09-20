"""
Dashboard Visual em Tempo Real (Fase 4):
- Suporte a 3 Modos de Visualização (Dashboard Completo, Planta Baixa 3D Fullscreen e Câmera AR Fullscreen)
- Trajetórias Neon Dinâmicas com Decaimento Temporal (Ribbon Trails)
- Integração Multimodal com Segmentação 3D de Nuvem de Pontos (DepthCluster3D)
- Câmera Virtual Estabilizada (Gimbal / Steadicam) e Menu HUD Interativo
"""
import cv2
import numpy as np
import time
from typing import List, Dict, Tuple, Optional
from src.core.config import config

class Dashboard3D:
    def __init__(self, room_width_m: float = 6.0, room_depth_m: float = 6.0):
        self.room_w = room_width_m
        self.room_d = room_depth_m
        self.canvas_w = 1600
        self.canvas_h = 900
        
        # Conexões anatômicas do esqueleto COCO (17 articulações)
        self.skeleton_edges = [
            (0, 1), (0, 2), (1, 3), (2, 4),           # Cabeça
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Braços / Cotovelos / Pulsos
            (5, 11), (6, 12), (11, 12),               # Tronco
            (11, 13), (13, 15), (12, 14), (14, 16)    # Pernas / Joelhos / Pés
        ]

        # Cores vibrantes neon por ID de rastreamento
        self.id_colors = [
            (255, 100, 50), (50, 255, 100), (100, 150, 255),
            (255, 50, 200), (255, 220, 50), (50, 220, 255)
        ]

        # Estado de Estabilização Suave da Câmera de Zoom (Gimbal / EMA)
        self.smooth_cam_cx: Optional[float] = None
        self.smooth_cam_cy: Optional[float] = None
        self.smooth_cam_box: Optional[float] = None
        self.locked_hand_key: Optional[Tuple[int, str]] = None
        self.hand_switch_count: int = 0

        # Buffers de renderização pré-alocados (Zero-Allocation Layout)
        self.canvas = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)
        self.cam_view = self.canvas[0:540, 0:960]
        self.floor_view = self.canvas[0:540, 960:1600]
        self.info_panel = self.canvas[540:900, 0:1600]

    def render(self, 
               color_bgr: np.ndarray, 
               tracks: List, 
               toys: List[Dict], 
               proxemic_events: List[Dict], 
               joint_events: List[Dict], 
               fps: float, 
               gpu_latency_ms: float, 
               hand_crops: Optional[List[Dict]] = None, 
               depth_clusters: Optional[List[Dict]] = None, 
               view_mode: int = 1, 
               show_trajectories: bool = True, 
               skeleton_mode: int = 2, 
               show_hud_help: bool = False,
               scene_id: str = "Cena 1",
               holder_id: Optional[int] = None,
               gpu_name: str = "DirectML (DirectX 12)",
               zones: Optional[List[Dict]] = None) -> np.ndarray:
        """
        view_mode:
          1 = Dashboard Científico Triplo (Padrão: Câmera + Planta 3D + Painel)
          2 = Planta Baixa 3D da Sala em Tela Cheia (1600x900)
          3 = Câmera de Realidade Aumentada em Tela Cheia (1600x900)
        """
        # =========================================================================
        # MODO 2: PLANTA BAIXA 3D EXPANDIDA EM TELA CHEIA (1600 x 900)
        # =========================================================================
        if view_mode == 2:
            self.canvas.fill(18) # Fundo escuro elegante
            self._draw_floor_view(
                target_img=self.canvas,
                tracks=tracks,
                toys=toys,
                proxemic_events=proxemic_events,
                joint_events=joint_events,
                grid_cx=800,
                grid_origin_y=820,
                ppm=110.0,
                depth_clusters=depth_clusters,
                show_trajectories=show_trajectories,
                is_fullscreen=True
            )
            # Cabeçalho flutuante
            hdr = f"PLANTA BAIXA 3D (TELA CHEIA) | FPS: {fps:.1f} | Pessoas: {len(tracks)} | [1] Voltar [M] Ajuda"
            cv2.putText(self.canvas, hdr, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 180), 2)
            if show_hud_help:
                self._draw_hud_help(self.canvas, view_mode, show_trajectories, skeleton_mode)
            return self.canvas

        # =========================================================================
        # MODO 3: CÂMERA DE REALIDADE AUMENTADA EM TELA CHEIA (1600 x 900)
        # =========================================================================
        if view_mode == 3:
            cv2.resize(color_bgr, (1600, 900), dst=self.canvas, interpolation=cv2.INTER_LINEAR)
            self._draw_camera_view(
                target_img=self.canvas,
                color_bgr=color_bgr,
                tracks=tracks,
                toys=toys,
                depth_clusters=depth_clusters,
                scale_x=1600.0 / color_bgr.shape[1],
                scale_y=900.0 / color_bgr.shape[0],
                show_trajectories=show_trajectories,
                skeleton_mode=skeleton_mode,
                is_fullscreen=True
            )
            # Miniatura PiP da mão no canto superior direito
            self._draw_pip_window(
                target_canvas=self.canvas,
                color_bgr=color_bgr,
                tracks=tracks,
                toys=toys,
                depth_clusters=depth_clusters,
                x_pos=1380,
                y_pos=25
            )
            hdr = f"CÂMERA REALIDADE AUMENTADA (1080p AR) | FPS: {fps:.1f} | Latencia IA: {gpu_latency_ms:.1f} ms | [1] Voltar"
            cv2.putText(self.canvas, hdr, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 180), 2)
            if show_hud_help:
                self._draw_hud_help(self.canvas, view_mode, show_trajectories, skeleton_mode)
            return self.canvas

        # =========================================================================
        # MODO 1: DASHBOARD CIENTÍFICO TRIPLO (PADRÃO)
        # =========================================================================
        # 1. Painel Esquerdo Superior: Câmera (960 x 540) gravada direto no canvas
        cam_view = self.cam_view
        cv2.resize(color_bgr, (960, 540), dst=cam_view, interpolation=cv2.INTER_LINEAR)
        self._draw_camera_view(
            target_img=cam_view,
            color_bgr=color_bgr,
            tracks=tracks,
            toys=toys,
            depth_clusters=depth_clusters,
            scale_x=960.0 / color_bgr.shape[1],
            scale_y=540.0 / color_bgr.shape[0],
            show_trajectories=show_trajectories,
            skeleton_mode=skeleton_mode,
            is_fullscreen=False
        )

        # 2. Painel Direito Superior: Planta Baixa 3D da Sala (640 x 540)
        floor_view = self.floor_view
        floor_view.fill(0)
        cv2.rectangle(floor_view, (0, 0), (639, 539), (40, 40, 40), 1)
        self._draw_floor_view(
            target_img=floor_view,
            tracks=tracks,
            toys=toys,
            proxemic_events=proxemic_events,
            joint_events=joint_events,
            grid_cx=320,
            grid_origin_y=500,
            ppm=70.0,
            depth_clusters=depth_clusters,
            show_trajectories=show_trajectories,
            is_fullscreen=False
        )
        cv2.putText(floor_view, "PLANTA BAIXA 3D (SALA)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 3. Painel Inferior: Painel de Métricas Socioenativas (1600 x 360)
        info_panel = self.info_panel
        info_panel.fill(0)
        cv2.rectangle(info_panel, (0, 0), (1599, 359), (30, 30, 30), 1)
        self._draw_metrics_panel(
            info_panel=info_panel,
            tracks=tracks,
            proxemic_events=proxemic_events,
            joint_events=joint_events,
            fps=fps,
            gpu_latency_ms=gpu_latency_ms,
            scene_id=scene_id,
            holder_id=holder_id,
            gpu_name=gpu_name
        )

        # Miniatura PiP na coluna 4 do painel inferior (focalizada no portador ou na pelúcia)
        self._draw_pip_window(
            target_canvas=info_panel,
            color_bgr=color_bgr,
            tracks=tracks,
            toys=toys,
            depth_clusters=depth_clusters,
            x_pos=1390,
            y_pos=105
        )

        cv2.putText(info_panel, f"v{config.version} | Modos: [1] Triplo | [2] Mapa | [3] AR | Operador: [N] Prox Cena | [F] Facilitador | [P] Portador | [M] Ajuda | [Q] Sair", 
                    (25, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (140, 140, 140), 1)


        if show_hud_help:
            self._draw_hud_help(self.canvas, view_mode, show_trajectories, skeleton_mode)

        return self.canvas

    def _draw_camera_view(self, 
                          target_img: np.ndarray, 
                          color_bgr: np.ndarray, 
                          tracks: List, 
                          toys: List[Dict], 
                          depth_clusters: Optional[List[Dict]],
                          scale_x: float, 
                          scale_y: float, 
                          show_trajectories: bool, 
                          skeleton_mode: int,
                          is_fullscreen: bool):
        """Renderiza esqueletos, caixas e trilhas neon reprojetadas na imagem da câmera."""
        # 1. Desenha brinquedos e objetos
        for toy in toys:
            bx1, by1, bx2, by2 = [int(v * (scale_x if i%2==0 else scale_y)) for i, v in enumerate(toy['bbox'])]
            cv2.rectangle(target_img, (bx1, by1), (bx2, by2), (0, 215, 255), 2)
            
            # Encontra a mão mais próxima
            t_pos = np.array(toy['pos_3d'])
            min_hand_dist = 999.0
            for trk in tracks:
                lh, rh = trk.hands_3d
                if lh is not None:
                    min_hand_dist = min(min_hand_dist, float(np.linalg.norm(np.array(lh) - t_pos)))
                if rh is not None:
                    min_hand_dist = min(min_hand_dist, float(np.linalg.norm(np.array(rh) - t_pos)))

            cam_dist_str = f"Z: {toy['pos_3d'][2]:.2f}m"
            hand_str = f" | Mão: {int(min_hand_dist * 100)}cm" if min_hand_dist < 0.60 else ""
            src_tag = " [ROI]" if toy.get('source') == 'hand_roi' else ""
            phys_tag = f" ({int(toy['physical_diameter_m']*100)}cm 3D)" if 'physical_diameter_m' in toy else ""

            lbl = f"{toy['class_name']}{src_tag}{phys_tag} [{cam_dist_str}{hand_str}]"
            cv2.putText(target_img, lbl, (bx1, max(18, by1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.48 if is_fullscreen else 0.42, (0, 215, 255), 2)

        # 2. Desenha pessoas, esqueletos e trilhas neon
        for trk in tracks:
            color = self.id_colors[(trk.track_id - 1) % len(self.id_colors)]
            
            # Trilha Neon com decaimento temporal na altura do tórax
            if show_trajectories:
                # Prioridade 1: Rastro 2D direto na altura do tórax/peito
                trail_pts = getattr(trk, 'trail_history_2d', None) or getattr(trk, 'feet_history_2d', None)
                if trail_pts and len(trail_pts) > 1:
                    t_pts = list(trail_pts)
                    n_pts = len(t_pts)
                    for i in range(1, n_pts):
                        decay = float(i) / float(n_pts) # 0.0 mais antigo -> 1.0 mais recente
                        c_fade = tuple(int(c * (0.20 + 0.80 * decay)) for c in color)
                        th = 1 if decay < 0.6 else (3 if is_fullscreen else 2)

                        u0 = int(t_pts[i-1][0] * scale_x)
                        v0 = int(t_pts[i-1][1] * scale_y)
                        u1 = int(t_pts[i][0] * scale_x)
                        v1 = int(t_pts[i][1] * scale_y)

                        if 0 <= u0 < target_img.shape[1] and 0 <= v0 < target_img.shape[0] and \
                           0 <= u1 < target_img.shape[1] and 0 <= v1 < target_img.shape[0]:
                            cv2.line(target_img, (u0, v0), (u1, v1), c_fade, th)
                elif len(trk.history) > 1:
                    # Fallback com projeção da pelve
                    pts = list(trk.history)
                    n_pts = len(pts)
                    for i in range(1, n_pts):
                        decay = float(i) / float(n_pts)
                        c_fade = tuple(int(c * (0.20 + 0.80 * decay)) for c in color)
                        th = 1 if decay < 0.6 else (3 if is_fullscreen else 2)
                        p0, p1 = pts[i-1], pts[i]
                        if p0[2] > 0.4 and p1[2] > 0.4:
                            u0 = int((960.0 + 1060.0 * p0[0] / p0[2]) * scale_x)
                            v0 = int((540.0 + 1060.0 * (p0[1] + 0.35) / p0[2]) * scale_y)
                            u1 = int((960.0 + 1060.0 * p1[0] / p1[2]) * scale_x)
                            v1 = int((540.0 + 1060.0 * (p1[1] + 0.35) / p1[2]) * scale_y)
                            if 0 <= u0 < target_img.shape[1] and 0 <= v0 < target_img.shape[0] and \
                               0 <= u1 < target_img.shape[1] and 0 <= v1 < target_img.shape[0]:
                                cv2.line(target_img, (u0, v0), (u1, v1), c_fade, th)

            # Caixa delimitadora
            if trk.last_bbox is not None and skeleton_mode > 0:
                bx1, by1, bx2, by2 = [int(v * (scale_x if i%2==0 else scale_y)) for i, v in enumerate(trk.last_bbox)]
                cv2.rectangle(target_img, (bx1, by1), (bx2, by2), color, 2)

            # Esqueleto COCO
            if trk.last_keypoints_2d is not None and skeleton_mode == 2:
                kpts_2d = trk.last_keypoints_2d
                for u, v in self.skeleton_edges:
                    if kpts_2d[u, 2] > 0.20 and kpts_2d[v, 2] > 0.20:
                        p1 = (int(kpts_2d[u, 0] * scale_x), int(kpts_2d[u, 1] * scale_y))
                        p2 = (int(kpts_2d[v, 0] * scale_x), int(kpts_2d[v, 1] * scale_y))
                        cv2.line(target_img, p1, p2, color, 2)
                        cv2.circle(target_img, p1, 3, (255, 255, 255), -1)
                        cv2.circle(target_img, p2, 3, (255, 255, 255), -1)

            # Mãos destacadas (se skeleton_mode >= 1)
            if skeleton_mode >= 1:
                left_2d, right_2d = trk.hands_2d
                left_3d, right_3d = trk.hands_3d
                if left_2d is not None:
                    hx, hy = int(left_2d[0] * scale_x), int(left_2d[1] * scale_y)
                    cv2.circle(target_img, (hx, hy), 7, (0, 255, 255), -1)
                    cv2.circle(target_img, (hx, hy), 10, (255, 255, 255), 2)
                if right_2d is not None:
                    hx, hy = int(right_2d[0] * scale_x), int(right_2d[1] * scale_y)
                    cv2.circle(target_img, (hx, hy), 7, (0, 255, 255), -1)
                    cv2.circle(target_img, (hx, hy), 10, (255, 255, 255), 2)

            # Cabeçalho com ID, Postura e Posição 3D
            px, py, pz = trk.position
            header_text = f"ID #{trk.track_id} [{trk.posture.upper()}] ({px:.2f}m, {pz:.2f}m)"
            if trk.last_bbox is not None:
                tx = int(trk.last_bbox[0] * scale_x)
                ty = int(max(25, trk.last_bbox[1] * scale_y - 10))
            else:
                tx, ty = 40, 40 * trk.track_id
            cv2.putText(target_img, header_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.58 if is_fullscreen else 0.50, color, 2)

    def _draw_floor_view(self, 
                         target_img: np.ndarray, 
                         tracks: List, 
                         toys: List[Dict], 
                         proxemic_events: List[Dict], 
                         joint_events: List[Dict],
                         grid_cx: int, 
                         grid_origin_y: int, 
                         ppm: float,
                         depth_clusters: Optional[List[Dict]],
                         show_trajectories: bool,
                         is_fullscreen: bool):
        """Renderiza a planta baixa 3D métrica com trilhas neon e conexões socioenativas."""
        max_h, max_w = target_img.shape[:2]

        # Grade métrica concêntrica
        for m in range(1, int(self.room_d) + 2):
            y_line = int(grid_origin_y - m * ppm)
            if 0 < y_line < max_h:
                cv2.line(target_img, (20, y_line), (max_w - 20, y_line), (40, 40, 40), 1)
                cv2.putText(target_img, f"{m}m", (25, y_line - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 100), 1)

        # Câmera Kinect
        cv2.circle(target_img, (grid_cx, grid_origin_y), 9, (0, 200, 255), -1)
        cv2.putText(target_img, "KINECT v2", (grid_cx - 40, grid_origin_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
        # Cone FOV
        fov_w = int(240 * (ppm / 70.0))
        cv2.line(target_img, (grid_cx, grid_origin_y), (grid_cx - fov_w, 60), (60, 60, 60), 1)
        cv2.line(target_img, (grid_cx, grid_origin_y), (grid_cx + fov_w, 60), (60, 60, 60), 1)

        # Brinquedos na sala
        for toy in toys:
            tx, ty, tz = toy['pos_3d']
            map_x = int(grid_cx + tx * ppm)
            map_y = int(grid_origin_y - tz * ppm)
            if 0 < map_x < max_w and 0 < map_y < max_h:
                cv2.rectangle(target_img, (map_x - 8, map_y - 8), (map_x + 8, map_y + 8), (0, 215, 255), -1)
                cv2.putText(target_img, toy['class_name'], (map_x + 11, map_y + 4), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.44 if is_fullscreen else 0.40, (0, 215, 255), 1)

        # Clusters 3D de profundidade (Fase 4)
        if depth_clusters:
            for dc in depth_clusters:
                cx, cy, cz = dc['centroid_3d']
                dc_x = int(grid_cx + cx * ppm)
                dc_y = int(grid_origin_y - cz * ppm)
                r_cluster = max(4, int((dc['diameter_m'] * ppm) / 2))
                cv2.circle(target_img, (dc_x, dc_y), r_cluster, (0, 255, 120), 1)

        # Pessoas, trajetórias neon e mãos
        for trk in tracks:
            color = self.id_colors[(trk.track_id - 1) % len(self.id_colors)]
            pts = list(trk.history)

            # Rastro de trajetória neon com decaimento temporal
            if show_trajectories and len(pts) > 1:
                n_pts = len(pts)
                for i in range(1, n_pts):
                    decay = float(i) / float(n_pts)
                    c_fade = tuple(int(c * (0.2 + 0.8 * decay)) for c in color)
                    th = 1 if decay < 0.6 else (3 if is_fullscreen else 2)

                    p_prev = (int(grid_cx + pts[i-1][0] * ppm), int(grid_origin_y - pts[i-1][2] * ppm))
                    p_curr = (int(grid_cx + pts[i][0] * ppm), int(grid_origin_y - pts[i][2] * ppm))
                    cv2.line(target_img, p_prev, p_curr, c_fade, th)

            # Posição central do indivíduo
            px, py, pz = trk.position
            cur_x = int(grid_cx + px * ppm)
            cur_y = int(grid_origin_y - pz * ppm)
            if 0 < cur_x < max_w and 0 < cur_y < max_h:
                cv2.circle(target_img, (cur_x, cur_y), 11 if is_fullscreen else 9, color, -1)
                cv2.circle(target_img, (cur_x, cur_y), 15 if is_fullscreen else 12, (255, 255, 255), 2)
                cv2.putText(target_img, f"ID #{trk.track_id}", (cur_x + 14, cur_y + 4), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50 if is_fullscreen else 0.44, (255, 255, 255), 2)

            # Mãos 3D no chão
            lh, rh = trk.hands_3d
            if lh is not None:
                lx, ly = int(grid_cx + lh[0] * ppm), int(grid_origin_y - lh[2] * ppm)
                if 0 < lx < max_w and 0 < ly < max_h:
                    cv2.circle(target_img, (lx, ly), 4, (0, 255, 255), -1)
            if rh is not None:
                rx, ry = int(grid_cx + rh[0] * ppm), int(grid_origin_y - rh[2] * ppm)
                if 0 < rx < max_w and 0 < ry < max_h:
                    cv2.circle(target_img, (rx, ry), 4, (0, 255, 255), -1)

        # Conexões proxêmicas (Zonas de Hall)
        for pe in proxemic_events:
            p1_x = int(grid_cx + pe['pos1'][0] * ppm)
            p1_y = int(grid_origin_y - pe['pos1'][2] * ppm)
            p2_x = int(grid_cx + pe['pos2'][0] * ppm)
            p2_y = int(grid_origin_y - pe['pos2'][2] * ppm)
            line_color = (0, 0, 255) if pe['zone'] == 'intima' else (0, 255, 255)
            cv2.line(target_img, (p1_x, p1_y), (p2_x, p2_y), line_color, 2)

    def _draw_metrics_panel(self, 
                            info_panel: np.ndarray, 
                            tracks: List, 
                            proxemic_events: List[Dict], 
                            joint_events: List[Dict], 
                            fps: float, 
                            gpu_latency_ms: float,
                            scene_id: str = "Cena 1",
                            holder_id: Optional[int] = None,
                            gpu_name: str = "DirectML (DirectX 12)"):
        """Desenha o cabeçalho de status e as 3 colunas científicas do painel inferior."""
        status_line = (f"SICAMO3D | GPU: {gpu_name} | "
                       f"FPS: {fps:.1f} | Latência IA: {gpu_latency_ms:.1f} ms | "
                       f"Cena: {scene_id} | Portador: #{holder_id if holder_id is not None else 'NENHUM'}")
        cv2.putText(info_panel, status_line, (25, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 180), 2)
        cv2.line(info_panel, (25, 50), (1575, 50), (60, 60, 60), 1)

        # Coluna 1: Estados Individuais (Posturas, Papéis e Velocidades)
        cv2.putText(info_panel, "[ PARTICIPANTES & PAPÉIS ]", (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 80), 2)
        y_off = 115
        if not tracks:
            cv2.putText(info_panel, "Nenhuma pessoa no stand", (30, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)
        for trk in tracks[:5]:
            is_holder = (trk.track_id == holder_id)
            holder_tag = " [PORTADOR]" if is_holder else ""
            role_tag = f" ({trk.role.upper()})" if hasattr(trk, 'role') else ""
            pres_tag = f" - {trk.presence_state}" if hasattr(trk, 'presence_state') else ""
            txt = f"ID #{trk.track_id}{role_tag}: {trk.posture.upper()} | {trk.speed:.2f} m/s{pres_tag}{holder_tag}"
            color = (0, 255, 255) if is_holder else self.id_colors[(trk.track_id - 1) % len(self.id_colors)]
            cv2.putText(info_panel, txt, (30, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)
            y_off += 28


        # Coluna 2: Proxêmica e Distâncias Interpessoais
        cv2.putText(info_panel, "[ PROXÊMICA & DISTÂNCIAS ]", (520, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 80), 2)
        y_off = 115
        if not proxemic_events:
            cv2.putText(info_panel, "Aguardando mais de 1 pessoa na sala", (520, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)
        for pe in proxemic_events[:5]:
            t1, t2 = pe["id_pair"]
            txt = f"Pares #{t1} <-> #{t2}: {pe['distance_m']:.2f} m [Zona {pe['zone'].upper()}]"
            color = (0, 100, 255) if pe['zone'] == 'intima' else (0, 255, 200)
            cv2.putText(info_panel, txt, (520, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)
            y_off += 28

        # Coluna 3: Eventos com Brinquedos e Atenção Conjunta
        cv2.putText(info_panel, "[ ATENÇÃO CONJUNTA & TOYS ]", (980, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 80), 2)
        y_off = 115
        if not joint_events:
            cv2.putText(info_panel, "Nenhum evento de atenção conjunta no momento", (980, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)
            y_off += 28
        for je in joint_events[:3]:
            txt = f" ATENÇÃO CONJUNTA! Pessoas {je['participant_ids']} no objeto: {je['toy_name']}"
            cv2.putText(info_panel, txt, (980, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            y_off += 28

    def _draw_pip_window(self, 
                         target_canvas: np.ndarray, 
                         color_bgr: np.ndarray, 
                         tracks: List, 
                         toys: List[Dict], 
                         depth_clusters: Optional[List[Dict]],
                         x_pos: int, 
                         y_pos: int):
        """Renderiza a miniatura PiP com foco inteligente na interação mão-brinquedo estabilizada."""
        hand_candidates = []
        for trk in tracks:
            lh_2d, rh_2d = trk.hands_2d
            lh_3d, rh_3d = trk.hands_3d
            if rh_2d is not None and rh_2d[2] > 0.10:
                hand_candidates.append({"track_id": trk.track_id, "label": "Mão Dir", "h2d": rh_2d, "h3d": rh_3d, "held_toy": trk.held_toy})
            if lh_2d is not None and lh_2d[2] > 0.10:
                hand_candidates.append({"track_id": trk.track_id, "label": "Mão Esq", "h2d": lh_2d, "h3d": lh_3d, "held_toy": trk.held_toy})

        candidate_target = None
        active_toy = None
        min_toy_dist = 999.0

        if toys and hand_candidates:
            for cand in hand_candidates:
                h2, h3 = cand["h2d"], cand["h3d"]
                for t in toys:
                    t_pos = np.array(t['pos_3d'])
                    d_3d = float(np.linalg.norm(np.array(h3) - t_pos)) if h3 is not None else 999.0
                    tbx1, tby1, tbx2, tby2 = t['bbox']
                    dist_2d = float(np.hypot(h2[0] - (tbx1 + tbx2)/2.0, h2[1] - (tby1 + tby2)/2.0))
                    score = min(d_3d, dist_2d / 350.0)
                    if score < min_toy_dist:
                        min_toy_dist = score
                        candidate_target = cand
                        active_toy = t

        target_hand = None
        if candidate_target is not None and min_toy_dist < 0.70:
            cand_key = (candidate_target["track_id"], candidate_target["label"])
            if self.locked_hand_key == cand_key or self.locked_hand_key is None:
                self.locked_hand_key = cand_key
                self.hand_switch_count = 0
                target_hand = candidate_target
            else:
                self.hand_switch_count += 1
                if self.hand_switch_count >= 6:
                    self.locked_hand_key = cand_key
                    self.hand_switch_count = 0
                    target_hand = candidate_target
                else:
                    prev = [c for c in hand_candidates if (c["track_id"], c["label"]) == self.locked_hand_key]
                    target_hand = prev[0] if prev else candidate_target
        else:
            self.locked_hand_key = None
            self.hand_switch_count = 0
            target_hand = hand_candidates[0] if hand_candidates else None

        if target_hand is not None:
            hx, hy, _ = target_hand["h2d"]
            depth_m = target_hand["h3d"][2] if (target_hand["h3d"] is not None and target_hand["h3d"][2] > 0.4) else 1.3

            if active_toy is not None and min_toy_dist < 0.70:
                tbx1, tby1, tbx2, tby2 = active_toy['bbox']
                target_cx = 0.5 * hx + 0.5 * (tbx1 + tbx2) / 2.0
                target_cy = 0.5 * hy + 0.5 * (tby1 + tby2) / 2.0
            else:
                target_cx, target_cy = hx, hy

            target_box = float(np.clip((0.55 * 1060.0) / max(0.5, depth_m), 180, 520))

            if self.smooth_cam_cx is None:
                self.smooth_cam_cx, self.smooth_cam_cy, self.smooth_cam_box = float(target_cx), float(target_cy), float(target_box)
            else:
                if np.hypot(target_cx - self.smooth_cam_cx, target_cy - self.smooth_cam_cy) > 3.0:
                    self.smooth_cam_cx = 0.80 * self.smooth_cam_cx + 0.20 * float(target_cx)
                    self.smooth_cam_cy = 0.80 * self.smooth_cam_cy + 0.20 * float(target_cy)
                self.smooth_cam_box = 0.92 * self.smooth_cam_box + 0.08 * float(target_box)

            half_b = int(self.smooth_cam_box // 2)
            scx, scy = int(self.smooth_cam_cx), int(self.smooth_cam_cy)
            x1 = max(0, scx - half_b)
            y1 = max(0, scy - half_b)
            x2 = min(color_bgr.shape[1], scx + half_b)
            y2 = min(color_bgr.shape[0], scy + half_b)

            if x2 > x1 + 50 and y2 > y1 + 50:
                raw_crop = color_bgr[y1:y2, x1:x2]
                c_img = cv2.resize(raw_crop, (180, 180))

                has_toy = False
                toy_detected_name = None
                for t in toys:
                    tbx1, tby1, tbx2, tby2 = t['bbox']
                    tcx, tcy = (tbx1 + tbx2) / 2.0, (tby1 + tby2) / 2.0
                    if x1 <= tcx <= x2 and y1 <= tcy <= y2:
                        has_toy = True
                        toy_detected_name = t['class_name']
                        sx, sy = 180.0 / (x2 - x1), 180.0 / (y2 - y1)
                        p1 = (int((tbx1 - x1) * sx), int((tby1 - y1) * sy))
                        p2 = (int((tbx2 - x1) * sx), int((tby2 - y1) * sy))
                        cv2.rectangle(c_img, p1, p2, (0, 255, 255), 2)
                        cv2.putText(c_img, toy_detected_name, (max(5, p1[0]), max(12, p1[1] - 4)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)

                cv2.putText(c_img, f"ID#{target_hand['track_id']} {target_hand['label']}", (5, 172), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 150), 1)

                # Grava no canvas destino
                target_canvas[y_pos:y_pos + 180, x_pos:x_pos + 180] = c_img
                border_col = (0, 255, 255) if has_toy else (80, 80, 80)
                cv2.rectangle(target_canvas, (x_pos - 1, y_pos - 1), (x_pos + 181, y_pos + 181), border_col, 2)
                tag_txt = f"{toy_detected_name.upper()} DETECTADO!" if has_toy else "Procurando objeto..."
                cv2.putText(target_canvas, tag_txt, (x_pos - 10, y_pos + 198), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255) if has_toy else (140, 140, 140), 1)

    def _draw_hud_help(self, target_img: np.ndarray, view_mode: int, show_trajectories: bool, skeleton_mode: int):
        """Desenha um card de ajuda e atalhos semi-transparente (Glassmorphism HUD)."""
        card_w, card_h = 360, 240
        x1, y1 = 1210, 80
        x2, y2 = x1 + card_w, y1 + card_h

        # Overlay semi-transparente
        sub = target_img[y1:y2, x1:x2].copy()
        dark = np.zeros_like(sub)
        cv2.addWeighted(dark, 0.82, sub, 0.18, 0, sub)
        target_img[y1:y2, x1:x2] = sub
        cv2.rectangle(target_img, (x1, y1), (x2, y2), (0, 255, 180), 2)

        cv2.putText(target_img, f"[ ATALHOS RAPIDOS - v{config.version} ]", (x1 + 20, y1 + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 220), 2)
        
        mode_names = {1: "Dashboard Triplo", 2: "Planta Baixa 3D", 3: "Câmera AR 1080p"}
        skel_names = {0: "Oculto", 1: "Apenas Mãos", 2: "Completo"}

        shortcuts = [
            f"[1, 2, 3] Modo de Tela: {mode_names.get(view_mode)}",
            f"[T] Trajetórias Neon: {'LIGADAS' if show_trajectories else 'DESLIGADAS'}",
            f"[S] Esqueleto Anatômico: {skel_names.get(skeleton_mode)}",
            f"[M] Alternar este Menu de Ajuda",
            f"[ESC / Q] Encerrar e Salvar Datasets",
            f"Fase 4: 3D Depth Cluster ATIVO"
        ]

        y_text = y1 + 65
        for s in shortcuts:
            cv2.putText(target_img, s, (x1 + 20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1)
            y_text += 28
