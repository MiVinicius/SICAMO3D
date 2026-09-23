# SICAMO3D: Sistema de Inferência com Captura de Movimento de Objetos em Ambiente 3D (v0.4.0)
### Rastreamento de Pessoas, Pelúcia Móvel e Análise de Cenas Teatrais em Tempo Real

[![Versão](https://img.shields.io/badge/Versão-v0.3.0-brightgreen.svg)](https://github.com)
[![Plataforma](https://img.shields.io/badge/Plataforma-Windows%2011%20%7C%20DirectX%2012-blue)](https://microsoft.com)
[![Hardware](https://img.shields.io/badge/Acelera%C3%A7%C3%A3o-DirectML%20(AMD%20%7C%20NVIDIA%20%7C%20Intel)-ED1C24)](https://github.com/microsoft/DirectML)
[![Sensor](https://img.shields.io/badge/Sensor-Microsoft%20Kinect%20v2-0078D7)](https://developer.microsoft.com/en-us/windows/kinect/)
[![Taxa de Quadros](https://img.shields.io/badge/FPS-30%20(Hardware%20Locked)-brightgreen)](#otimizações-de-desempenho-e-latência)
[![IA Backend](https://img.shields.io/badge/Backend-DirectML%20%7C%20ONNX%20Runtime-orange)](https://github.com/microsoft/DirectML)
[![Testes](https://img.shields.io/badge/Testes-39%2F39%20Passando%20(pytest)-success)](tests/)

---

## 📖 Visão Geral

O **SICAMO3D** (**S**istema de **I**nferência com **CA**ptura de **M**ovimento de **O**bjetos em Ambiente **3D**) é uma plataforma de pesquisa científica e monitoramento socioenativo em tempo real, desenvolvida nativamente para **Windows 11** com aceleração por hardware via **DirectML (DirectX 12)** compatível com GPUs modernas (AMD Radeon, NVIDIA GeForce e Intel Arc/Iris Xe).

O sistema estrutura-se em torno do paradigma: **Pessoas + Artefato Móvel Único (Pelúcia do Espetáculo) + Cenas Teatrais**, com portador exclusivo por vez, detecção de passagens de posse, papéis e zonas no espaço físico:

1. **Rastreamento Biomecânico e Referencial da Sala:** Rastreamento anatômico de 17 articulações COCO com alinhamento ao plano do chão ($Y = 0$), cálculo de velocidade horizontal no chão ($X, Z_{sala}$) e filtro de Kalman 3D com $dt$ dinâmico real.
   - *Nota de Engenharia — Âncora Anatômica (Issue M1-06):* Em divergência deliberada à menção preliminar da proposta ("cabeça e ombros"), a âncora 3D de centro corporal adotada no código utiliza o centro do tronco formado pelos **ombros e quadris** (articulações 5, 6, 11 e 12 do COCO, sem o nariz). Justificativa técnica: a cabeça sofre frequentes oclusões em ambientes com plateia aglomerada e possui rotações e movimentos independentes de alta frequência (olhar para os lados, abaixar a cabeça). Os ombros e quadris fornecem a estimativa física mais estável do centro de massa e da velocidade de deslocamento no solo.
2. **Inferência de Portador Único (`holder_inference.py`):** Atribuição exclusiva e contínua do portador da pelúcia utilizando máquina de estados finita (`COM_PORTADOR`, `PASSAGEM`, `SEM_PORTADOR`, `INDETERMINADO`), histerese temporal e suporte geométrico ao **abraço** (distância ao tronco e correlação de co-movimento).
3. **Detecção de Passagens de Posse (`handoff`):** Identificação de entregas doador $\rightarrow$ receptor com anotação da flag `id_ambiguous` para cruzamentos em que há incerteza ou nascimento recente de tracks.
4. **Supressão de Pose Duplicada na Pelúcia:** Filtro automático que descarta esqueletos humanos espúrios detectados sobre o próprio corpo da pelúcia (cruzamento de IoU $> 0.30$ com biometria corporal infantil mínima).
5. **Métricas Científicas de Circulação (`plush_metrics.py`):** Cálculo de tempo de posse normalizado pelo tempo de permanência no stand, Coeficiente de Gini de concentração da posse (isolando facilitadores e passantes), matriz de transição de posse e atenção angular da plateia direcionada ao portador, tudo segmentado por cena teatral (`scene_id`).
6. **Zonas Espaciais e Papéis (`zone_manager.py` e `config/zones.yaml`):** Delimitação métrica da sala (`stand_total`, `palco_facilitador`, `area_interacao_criancas`), separação entre passantes e participantes ativos, e sugestão automática do papel de facilitador para quem atua na estação fixa.
7. **Interface do Operador Interativa (Modo A):** Seleção de participante por clique com o mouse no mapa da planta baixa ou na imagem da câmera, alternância com tecla `TAB`, avanço de cenas (`N`), marcação/override de portador (`P`) e demarcação de facilitador (`F`).
8. **Detector de Pelúcia Desacoplado:** O sistema inicializa e opera no Modo A mesmo sem o arquivo de pesos da pelúcia (`yolo11s_plush.onnx`), dependendo apenas do detector de pose humana e anotações do operador.
9. **Gravação Científica Schema v3 & Resumos JSON:** Registro frame-a-frame ininterrupto em `trajectories.csv`, `plush_state.csv`, `events.csv`, `session.yaml`, exportação consolidada em `metrics_summary.json` por cena e logs brutos em `raw_sensor.jsonl`.
10. **Conformidade Ética e CEP:** Processamento e gravação estritamente numéricos com esqueletos anonimizados e observador ao vivo, sem retenção de vídeos de cor com rostos de crianças.
11. **Detecção de F-Formations e Arranjos Socioespaciais (`f_formations.py` e `proxemics.py`):** Identificação contínua de configurações de interação interpessoal (*vis-à-vis*, lado a lado, circular/em leque) baseadas no modelo *o-space* de Kendon / Cristani, com classificação da zona proxêmica no instante exato de cada passagem de posse (*handoff*).
12. **Pipeline ETL e BI de Dados Científicos (`tools/etl.py`):** Processamento automatizado de sessões gravadas com extração de métricas de circulação, consolidação estatística, geração de relatórios em Markdown e JSON estruturado para análise socioenativa.

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
|  17 Keypoints (DirectML)|                     |  Detector (Opcional)    |
+------------+------------+                     +------------+------------+
             |                                               |
             +-----------------------+-----------------------+
                                     |
                                     v
                      +-----------------------------+
                      |  Filtro Anti-Fantasma       |
                      | (Supressão de Pose Pelúcia) |
                      +--------------+--------------+
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
| - Override do Operador  |                     +-------------------------+
+------------+------------+                                  |
             |                                               |
             +-----------------------+-----------------------+
                                     |
                                     v
                      +-----------------------------+
                      |    PlushMetricsAnalyzer     |
                      | - Tempo de Posse Normalizado|
                      | - Gini Plateia vs Facilitad.|
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
| - plush_state.csv       |                     | - Zonas Poligonais      |
| - events.csv / session  |                     | - Seleção Clique / TAB  |
| - metrics_summary.json  |                     | - Painel de Métricas    |
+-------------------------+                     +-------------------------+
```

---

## 📂 Estrutura do Repositório

```text
teste/
├── .gitattributes                 # Normalização de quebras de linha e arquivos binários
├── .gitignore                     # Regras de exclusão do Git (pesos, gravações, calibração, runs)
├── LICENSE                        # Licença de código aberto (MIT)
├── README.md                      # Documentação completa do projeto
├── pytest.ini                     # Configurações do framework de testes
├── requirements.txt               # Dependências essenciais de runtime em produção (MIT/Apache-2.0)
├── requirements-dev.txt           # Dependências de desenvolvimento, testes e treinamento (AGPL Ultralytics)
├── main.py                        # Ponto de entrada do sistema com interface do operador
├── edge_node.py                   # Ponto de entrada do nó sensor secundário (Kinect #2)
├── config/
│   └── zones.yaml                 # Configuração poligonal de zonas da sala e cenas teatrais
│   # Nota: calibration.json é gerado automaticamente na calibração física e ignorado no Git
├── docs/
│   ├── ROADMAP.md                 # Roadmap completo com marcos e issues do projeto
│   ├── dataset_pelucia.md         # Protocolo de captura e anotação do dataset da pelúcia
│   ├── avaliacao_detector_pelucia.md # Matriz de avaliação quantitativa do detector
│   ├── protocolo_anotacao.md      # Protocolo de anotação manual e ground truth para validação
│   └── testes_risco_m0.md         # Checklist de testes físicos de risco com Kinect v2
├── recordings/                    # Diretório de gravações locais (ignorado no Git; apenas .gitkeep versionado)
│   └── .gitkeep                   # Preserva o diretório no clone
├── src/
│   ├── ai/
│   │   └── directml_inference.py  # Motor ONNX Runtime com DirectML (device_id explícito)
│   ├── core/
│   │   ├── config.py              # Parâmetros de AI, tracking, holder, proxêmica e rede
│   │   ├── kinect_sensor.py       # Driver do Kinect v2 com CoordinateMapper e NaN handling
│   │   ├── pipeline.py            # Orquestrador central modular do fluxo de dados
│   │   ├── room_calibration.py    # Transformação de coordenadas e plano do chão (Y = 0)
│   │   └── zone_manager.py        # Avaliação espacial de zonas e controle de cenas
│   ├── experimental/              # Módulos legados e protótipos exploratórios arquivados
│   │   ├── README.md              # Documentação e justificativas de arquivamento
│   │   ├── toy_interaction.py     # Heurística inicial de interação (substituída por holder_inference)
│   │   ├── hand_roi_detector.py   # Crop de alta definição nas mãos
│   │   └── depth_clustering_3d.py # Agrupamento de nuvem de pontos 3D
│   ├── network/
│   │   └── network_protocol.py    # Comunicação UDP não-bloqueante entre nós e Hub
│   ├── socioenative/
│   │   ├── f_formations.py        # Detecção de F-Formations e arranjos sócio-espaciais (Kendon/Cristani)
│   │   ├── holder_inference.py    # Atribuição de portador, histerese, handoff e proximidade
│   │   ├── plush_metrics.py       # Cálculo de Gini, matriz de transição e atenção angular
│   │   ├── posture_classifier.py  # Classificação postural adaptada a crianças (relação tronco/perna)
│   │   ├── proxemics.py           # Zonas proxêmicas de Hall e classificação pontual em handoffs
│   │   └── scientific_logger.py   # Gravador científico com Schema v3 contínuo
│   ├── tracking/
│   │   ├── kalman_filter_3d.py    # Filtro de Kalman 3D com dt dinâmico e velocidade no chão
│   │   ├── keypoint_filter.py     # Filtro temporal EMA anti-jitter de keypoints
│   │   └── tracker_3d.py          # Rastreamento espacial húngaro com supressão de fantasmas
│   └── visualization/
│       └── dashboard_3d.py        # Dashboard visual triplo com mapa da sala e PiP do portador
├── tests/                         # Suíte de testes automatizados (39 testes passando)
│   ├── test_etl_and_proxemics.py  # Testes de ETL, relatórios Markdown e proxêmica em handoffs
│   ├── test_f_formations.py       # Testes de F-formations (vis-à-vis, lado a lado, circular)
│   ├── test_holder_inference.py   # Testes sintéticos de posse, abraço e handoffs
│   ├── test_holder_weights_and_modes.py # Testes de pesos/limiares dinâmicos e comparador A vs B
│   ├── test_pipeline_smoke.py     # Testes de fumaça fim-a-fim do pipeline (Modo A e B)
│   ├── test_plush_ghost_suppression.py # Testes de supressão de pose duplicada na pelúcia
│   ├── test_plush_metrics.py      # Testes de Gini, tempo normalizado e Schema v3
│   ├── test_room_calibration.py   # Testes de geometria do chão (Y=0, normal up, quadrantes)
│   ├── test_tools_evaluation.py   # Testes de evaluate.py (1:1 bipartite matching, ID switches)
│   ├── test_tracker_synthetic.py  # Testes de Kalman, ground speed e descarte de fantasmas
│   ├── test_zone_events.py        # Testes de transição de zonas, passante vs espectador e alcance
│   ├── test_kinect_capture.py     # Teste manual com hardware físico conectado
│   └── test_system_pipeline.py    # Teste de 5 frames com hardware físico
├── tools/
│   ├── etl.py                     # Pipeline ETL, BI e relatórios de sessões gravadas
│   ├── compare_modes.py           # Comparador analítico Modo A (Operador) vs Modo B (Automático)
│   ├── train_plush_detector.py    # Treinamento e exportação ONNX do detector YOLO11 da pelúcia
│   ├── evaluate.py                # Avaliação científica com casamento 1:1 e ID switches
│   ├── replay.py                  # Player visual de sessões gravadas (sem Kinect)
│   ├── benchmark_directml.py      # Perfilamento de latência na GPU
│   └── download_weights.py        # Download e exportação de modelos ONNX
└── weights/                       # Diretório de modelos (pesos .onnx/.pt ignorados no Git)
    ├── .gitkeep                   # Preserva o diretório no repositório
    └── README.md                  # Instruções para download e exportação dos pesos
```


## 🚀 Como Executar

### 1. Instalação de Dependências
Certifique-se de utilizar o Python 3.11 no Windows 11 com o ambiente virtual ativado:
```powershell
.venv\Scripts\activate

# Para execução em produção / runtime (ONNX DirectML - 100% permissivo MIT):
pip install -r requirements.txt

# Para desenvolvimento, execução dos testes ou treinamento do detector:
pip install -r requirements-dev.txt
```

### 2. Execução dos Testes Automatizados (Sem Hardware)
Valide a integridade de todos os algoritmos matemáticos, geometria, calibração, F-formations e máquina de estados sem precisar de hardware físico:
```powershell
pytest -v
```
*(39 testes unitários e de integração executados com 100% de sucesso em ~10s).*

### 3. Obtenção dos Modelos Neurais (Pesos ONNX)
Como modelos de redes neurais binários pesados não são versionados no Git, baixe e prepare os pesos pré-treinados antes de iniciar o sistema com sensor:
```powershell
python tools/download_weights.py
```
*(Baixa e exporta o modelo de pose `yolo11s-pose.onnx` compatível com DirectML).*

### 4. Operação em Tempo Real (Com Kinect v2)
Inicie o sistema no PC principal conectado ao sensor Kinect v2:
```powershell
python main.py
```

#### Controles Interativos do Operador (Modo A):
- **Clique no Mapa ou Câmera**: Seleciona diretamente o participante na sala (halo dourado `[SELECIONADO]`).
- **`TAB`**: Alterna sequencialmente o foco entre os participantes ativos na sala.
- **`P`**: Atribui manualmente a posse da pelúcia ao participante selecionado (com duração de 6s ou até próxima entrega).
- **`F`**: Alterna o papel de facilitador do participante selecionado (isolando-o do Gini da plateia).
- **`N`**: Avança para a próxima cena teatral (`Cena 1` $\rightarrow$ `Cena 2` $\rightarrow$ `Cena 3`) e salva resumo parcial.
- **`1` / `2` / `3`**: Alterna os modos de tela (Dashboard Triplo / Planta Baixa 3D Full / Câmera AR).
- **`T`**: Ativa/desativa as trajetórias contínuas em neon.
- **`S`**: Alterna o modo de esqueleto (Oculto / Mãos / Completo).
- **`M`**: Abre/fecha o menu HUD de ajuda rápida.
- **`Q` ou `ESC`**: Encerra o sistema gravando com segurança os datasets e `metrics_summary.json`.

### 5. Reprodução de Sessões Gravadas (Replay Offline)
Para reproduzir visualmente uma sessão gravada e inspecionar a dinâmica da pelúcia:
```powershell
python tools/replay.py recordings/session_20260919_213000/
```

### 6. Avaliação Científica com Gabarito
Para comparar os eventos de passagem detectados contra anotações manuais com casamento estrito 1:1 e contagem de ID switches:
```powershell
python tools/evaluate.py recordings/session_20260919_213000/ --gt caminho/gabarito_events.csv
```
Com suporte opcional a gabarito contínuo frame-a-frame:
```powershell
python tools/evaluate.py recordings/session_20260919_213000/ --gt caminho/gabarito_events.csv --gt-frames caminho/gabarito_frames.csv
```

### 7. Pipeline ETL e Relatórios de BI
Gere análises consolidadas e relatórios completos em formato Markdown e JSON para publicação científica:
```powershell
python tools/etl.py recordings/session_20260919_213000/ --output-dir relatorios/
```

### 8. Comparativo Modo A (Operador) vs Modo B (Automático)
Compare a concordância de inferência entre a marcação do operador e a detecção automática com matriz de confusão e métricas de posse:
```powershell
python tools/compare_modes.py recordings/session_20260919_213000/
```

### 9. Treinamento do Detector da Pelúcia (Opcional - Requer GPU)
Treine ou faça fine-tuning do detector YOLO11 em dataset próprio e exporte automaticamente para ONNX:
```powershell
python tools/train_plush_detector.py --data config/dataset_plush.yaml --epochs 50 --batch 16
```

---

## ⚖️ Licença e Arquitetura de Dependências

O projeto **SICAMO3D** foi arquitetado com isolamento estrito entre o ambiente de inferência em produção e o ambiente de desenvolvimento/treinamento:

1. **Código-fonte do Projeto:**
   - Todo o código do SICAMO3D é distribuído sob a licença **MIT** (veja [LICENSE](LICENSE)).
2. **Ambiente de Produção / Runtime (`requirements.txt`):**
   - O runtime de inferência utiliza **ONNX Runtime com DirectML** (`onnxruntime-directml`), biblioteca mantida pela Microsoft sob licença **MIT**, além de bibliotecas padrão científicas sob licenças permissivas (**NumPy**, **OpenCV-Python BSD**, **SciPy**, **PyYAML**).
   - O pipeline em tempo real **não** importa nem executa o pacote Ultralytics.
3. **Ambiente de Treinamento e Exportação (`requirements-dev.txt`):**
   - A ferramenta de treinamento [`tools/train_plush_detector.py`](tools/train_plush_detector.py) utiliza o framework **Ultralytics** (YOLO11), distribuído sob a licença **AGPL-3.0** (ou licença comercial Enterprise).
   - O uso da Ultralytics fica restrito ao fluxo de desenvolvimento e treinamento offline. A exportação gera artefatos de pesos abertos no padrão neutro `.onnx`, consumidos posteriormente pelo motor DirectML do SICAMO3D de forma 100% desacoplada.

