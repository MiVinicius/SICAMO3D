# SICAMO3D: Sistema de Rastreamento Socioenativo 3D (v0.2.0)
### Rastreamento de Pessoas, Pelúcia Móvel e Análise de Cenas Teatrais em Tempo Real

[![Versão](https://img.shields.io/badge/Versão-v0.2.0-brightgreen.svg)](https://github.com)
[![Plataforma](https://img.shields.io/badge/Plataforma-Windows%2011%20%7C%20DirectX%2012-blue)](https://microsoft.com)
[![Hardware](https://img.shields.io/badge/GPU-AMD%20Radeon%20RX%206600-ED1C24)](https://amd.com)
[![Sensor](https://img.shields.io/badge/Sensor-Microsoft%20Kinect%20v2-0078D7)](https://developer.microsoft.com/en-us/windows/kinect/)
[![Taxa de Quadros](https://img.shields.io/badge/FPS-30%20(Hardware%20Locked)-brightgreen)](#otimizações-de-desempenho-e-latência)
[![IA Backend](https://img.shields.io/badge/Backend-DirectML%20%7C%20ONNX%20Runtime-orange)](https://github.com/microsoft/DirectML)
[![Testes](https://img.shields.io/badge/Testes-14%2F14%20Passando%20(pytest)-success)](tests/)

---

## 📖 Visão Geral

O **SICAMO3D** é uma plataforma de pesquisa científica e monitoramento socioenativo 3D em tempo real, desenvolvida nativamente para **Windows 11** com aceleração por hardware via **DirectML (DirectX 12)** em GPUs **AMD Radeon RX 6600** (e nós secundários com GPUs NVIDIA/Intel).

O sistema estrutura-se em torno do paradigma: **Pessoas + Artefato Móvel Único (Pelúcia do Espetáculo) + Cenas Teatrais**, com portador exclusivo por vez, detecção de passagens de posse, papéis e zonas no espaço físico:

1. **Rastreamento Biomecânico e Referencial da Sala:** Rastreamento anatômico de 17 articulações COCO com alinhamento ao plano do chão ($Y = 0$), cálculo de velocidade horizontal ($X, Z_{sala}$) e filtro de Kalman com $dt$ dinâmico real.
2. **Inferência de Portador Único (`holder_inference.py`):** Atribuição exclusiva e contínua do portador da pelúcia utilizando máquina de estados finita (`COM_PORTADOR`, `PASSAGEM`, `SEM_PORTADOR`, `INDETERMINADO`), histerese temporal e suporte geométrico ao **abraço** (distância ao tronco e correlação de co-movimento).
3. **Detecção de Passagens de Posse (`handoff`):** Identificação de entregas doador $\rightarrow$ receptor com anotação da flag `id_ambiguous` para cruzamentos em que há incerteza ou nascimento recente de tracks.
4. **Métricas Científicas de Circulação (`plush_metrics.py`):** Cálculo de tempo de posse normalizado pelo tempo de permanência no stand, Coeficiente de Gini de concentração da posse, matriz de transição de posse e atenção angular da plateia direcionada ao portador, tudo segmentado por cena teatral (`scene_id`).
5. **Zonas Espaciais e Papéis (`zone_manager.py` e `config/zones.yaml`):** Delimitação métrica da sala (`stand_total`, `palco_facilitador`, `area_interacao_criancas`), separação entre passantes e participantes ativos, e sugestão automática do papel de facilitador para quem atua na estação fixa.
6. **Interface do Operador (Modo A):** Controle em tempo real pelo operador para avanço de cenas (`N`), atribuição/override de portador (`P`) e demarcação de facilitador (`F`).
7. **Gravação Científica Schema v3:** Registro frame-a-frame ininterrupto (mesmo com stand vazio) em `trajectories.csv`, `plush_state.csv`, `events.csv`, `session.yaml` e logs brutos em `raw_sensor.jsonl`.
8. **Conformidade Ética e CEP:** Processamento e gravação estritamente numéricos com esqueletos anonimizados e observador ao vivo, sem retenção de vídeos de cor com rostos de crianças.

---

## 🏗️ Arquitetura do Sistema

```text
                      +-----------------------------+
                      |   Sensor Kinect v2 (USB)    |
                      |  RGB 1080p (30Hz) + Depth   |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |       KinectSensor          |
                      | (CoordinateMapper + Chão Y=0)|
                      +--------------+--------------+
                                     |
             +-----------------------+-----------------------+
             |                                               |
             v                                               v
+-------------------------+                     +-------------------------+
|     YOLO11s-Pose        |                     |   YOLO11s-Plush         |
|  17 Keypoints (DirectML)|                     |  Detector Ajustado      |
+------------+------------+                     +------------+------------+
             |                                               |
             +-----------------------+-----------------------+
                                     |
                                     v
                      +-----------------------------+
                      |   Tracker 3D (Kalman 3D)    |
                      | dt dinâmico, chão XZ, roles |
                      +--------------+--------------+
                                     |
             +-----------------------+-----------------------+
             |                                               |
             v                                               v
+-------------------------+                     +-------------------------+
|    HolderInference      |                     |      ZoneManager        |
| - Posse Exclusiva       |                     | - Zonas (Stand/Palco)   |
| - Máquina de Estados    |                     | - Cenas Teatrais        |
| - Handoff & id_ambiguous|                     | - Presença & Facilitador|
+------------+------------+                     +------------+------------+
             |                                               |
             +-----------------------+-----------------------+
                                     |
                                     v
                      +-----------------------------+
                      |    PlushMetricsAnalyzer     |
                      | - Tempo de Posse Normalizado|
                      | - Gini de Concentração      |
                      | - Grafo Doador -> Receptor  |
                      | - Atenção Angular da Plateia|
                      +--------------+--------------+
                                     |
             +-----------------------+-----------------------+
             |                                               |
             v                                               v
+-------------------------+                     +-------------------------+
| ScientificLogger v3     |                     | Dashboard 3D (Triplo)   |
| - trajectories.csv      |                     | - Câmera + Planta Baixa |
| - plush_state.csv       |                     | - Painel de Métricas    |
| - events.csv / session  |                     | - PiP do Portador       |
+-------------------------+                     +-------------------------+
```

---

## 📂 Estrutura do Repositório

```text
teste/
├── .gitattributes                 # Normalização de quebras de linha e arquivos binários
├── .gitignore                     # Regras de exclusão do Git (.venv, pesos, gravações)
├── LICENSE                        # Licença de código aberto (MIT)
├── README.md                      # Documentação completa do projeto
├── requirements.txt               # Dependências Python essenciais
├── main.py                        # Ponto de entrada do sistema com interface do operador
├── edge_node.py                   # Ponto de entrada do nó sensor secundário (Kinect #2)
├── config/
│   └── zones.yaml                 # Configuração poligonal de zonas da sala e cenas teatrais
├── recordings/                    # Datasets científicos salvos no Schema v3 (ignorados no Git)
│   └── session_YYYYMMDD_HHMMSS/
│       ├── session.yaml           # Metadados da sessão, protocolo ético e parâmetros
│       ├── trajectories.csv       # Trajetórias contínuas, roles, presence e ground speed
│       ├── plush_state.csv        # Estado da pelúcia (portador exclusivo, coordenadas)
│       ├── events.csv             # Handoffs, mudanças de cena e marcações do operador
│       └── raw_sensor.jsonl       # Log bruto anonimizado para replay e reprocessamento
├── src/
│   ├── ai/
│   │   ├── directml_inference.py  # Motor ONNX com DirectML (device_id explícito)
│   │   ├── hand_roi_detector.py   # Inspeção em alta definição nas mãos com TTL
│   │   └── depth_clustering_3d.py # Agrupamento de nuvem de pontos 3D (experimental)
│   ├── core/
│   │   ├── config.py              # Parâmetros de AI, tracking, holder, proxêmica e rede
│   │   ├── kinect_sensor.py       # Driver do Kinect v2 com CoordinateMapper e NaN handling
│   │   ├── pipeline.py            # Orquestrador central modular do fluxo de dados
│   │   ├── room_calibration.py    # Transformação de coordenadas e plano do chão (Y = 0)
│   │   └── zone_manager.py        # Avaliação espacial de zonas e controle de cenas
│   ├── network/
│   │   └── network_protocol.py    # Comunicação UDP não-bloqueante entre nós e Hub
│   ├── socioenative/
│   │   ├── holder_inference.py    # Atribuição de portador, histerese e detecção de handoff
│   │   ├── plush_metrics.py       # Cálculo de Gini, matriz de transição e atenção angular
│   │   ├── posture_classifier.py  # Classificação postural (em pé, sentado, agachado)
│   │   ├── proxemics.py           # Zonas de proxêmica de Hall calibradas para crianças
│   │   └── scientific_logger.py   # Gravador científico com Schema v3 contínuo
│   ├── tracking/
│   │   ├── kalman_filter_3d.py    # Filtro de Kalman 3D com dt dinâmico e velocidade no chão
│   │   ├── keypoint_filter.py     # Filtro temporal EMA anti-jitter de keypoints
│   │   └── tracker_3d.py          # Rastreamento espacial húngaro com supressão de fantasmas
│   └── visualization/
│       └── dashboard_3d.py        # Dashboard visual triplo com mapa da sala e PiP do portador
├── tests/
│   ├── test_holder_inference.py   # Testes sintéticos de posse, abraço e handoffs
│   ├── test_plush_metrics.py      # Testes de Gini, tempo normalizado e Schema v3
│   ├── test_tracker_synthetic.py  # Testes de Kalman, ground speed e descarte de fantasmas
│   ├── test_tools_evaluation.py   # Testes de ponta a ponta para evaluate.py e replay.py
│   ├── test_kinect_capture.py     # Teste com hardware físico conectado
│   └── test_system_pipeline.py    # Teste de 5 frames com hardware físico
├── tools/
│   ├── evaluate.py                # Avaliação científica com precisão, recall e F1 contra gabarito
│   ├── replay.py                  # Player visual de sessões gravadas (sem Kinect)
│   ├── benchmark_directml.py      # Perfilamento de latência na GPU
│   └── download_weights.py        # Download e exportação de modelos ONNX
└── weights/                       # Modelos neurais em formato ONNX
    ├── yolo11s-pose.onnx          # Pose estimation (17 articulações COCO)
    └── yolo11s_plush.onnx         # Detector da pelúcia (classe única)
```

---

## 🚀 Como Executar

### 1. Instalação de Dependências
Certifique-se de utilizar o Python 3.11 no Windows 11 com o ambiente virtual ativado:
```powershell
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Execução dos Testes Automatizados (Sem Hardware)
Valide a integridade de todos os algoritmos matemáticos e da máquina de estados sem precisar do Kinect conectado:
```powershell
python -m pytest -v
```

### 3. Operação em Tempo Real (Com Kinect v2)
Inicie o sistema no PC principal conectado ao sensor Kinect v2:
```powershell
python main.py
```

#### Controles Interativos do Operador (Modo A):
- **`N`**: Avança para a próxima cena teatral (`Cena 1` $\rightarrow$ `Cena 2` $\rightarrow$ `Cena 3`).
- **`P`**: Atribui manualmente a posse da pelúcia ao participante em foco.
- **`F`**: Alterna o papel de facilitador de um participante selecionado.
- **`1` / `2` / `3`**: Alterna os modos de tela (Dashboard Triplo / Planta Baixa 3D Full / Câmera AR).
- **`T`**: Ativa/desativa as trajetórias contínuas em neon.
- **`S`**: Alterna o modo de esqueleto (Oculto / Mãos / Completo).
- **`M`**: Abre/fecha o menu HUD de ajuda rápida.
- **`Q` ou `ESC`**: Encerra o sistema gravando com segurança os arquivos no disco.

### 4. Reprodução de Sessões Gravadas (Replay Offline)
Para reproduzir visualmente uma sessão gravada e inspecionar a dinâmica da pelúcia:
```powershell
python tools/replay.py recordings/session_20260919_213000/
```

### 5. Avaliação Científica com Gabarito
Para comparar os eventos de passagem detectados contra anotações manuais de um observador independente:
```powershell
python tools/evaluate.py recordings/session_20260919_213000/ --gt caminho/gabarito_events.csv
```

---

## ⚖️ Licença e Dependências de Terceiros
- O código-fonte deste projeto é distribuído sob a licença **MIT** (veja [LICENSE](LICENSE)).
- O treinamento e exportação de modelos derivados da arquitetura YOLO11 utilizam a biblioteca **Ultralytics**, distribuída sob a licença **AGPL-3.0**. Certifique-se de cumprir os termos correspondentes caso distribua modelos ajustados.
