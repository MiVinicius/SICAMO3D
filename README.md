# Sistema de Captura de Movimento em 3D Multi-Alvo (v0.1.0)
### Rastreamento Espacial 3D, Pose Anatômica, Proxêmica e Interação com Brinquedos em Tempo Real

[![Versão](https://img.shields.io/badge/Versão-v0.1.0-brightgreen.svg)](https://github.com)
[![Plataforma](https://img.shields.io/badge/Plataforma-Windows%2011%20%7C%20DirectX%2012-blue)](https://microsoft.com)
[![Hardware](https://img.shields.io/badge/GPU-AMD%20Radeon%20RX%206600-ED1C24)](https://amd.com)
[![Sensor](https://img.shields.io/badge/Sensor-Microsoft%20Kinect%20v2-0078D7)](https://developer.microsoft.com/en-us/windows/kinect/)
[![Taxa de Quadros](https://img.shields.io/badge/FPS-30%20(Hardware%20Locked)-brightgreen)](#otimizações-de-desempenho-e-latência)
[![IA Backend](https://img.shields.io/badge/Backend-DirectML%20%7C%20ONNX%20Runtime-orange)](https://github.com/microsoft/DirectML)

---

## 📖 Visão Geral

Este projeto consiste em uma plataforma de pesquisa científica e monitoramento socioenativo 3D em tempo real, inspirada no paradigma **OpenPTrack**, desenvolvida nativamente para **Windows 11** com aceleração por hardware via **DirectML (DirectX 12)** em GPUs **AMD Radeon RX 6600** (e nós secundários com GPUs NVIDIA/Intel).

O sistema rastreia simultaneamente múltiplos indivíduos e objetos em um espaço físico (sala), reconstruindo métricas biomecânicas e socioemocionais:
1. **Pessoas e Biomecânica:** Rastreamento anatômico contínuo (17 articulações COCO em metros 3D reais), velocidade linear e classificação postural (*em pé*, *sentado no chão*, *agachado*, *debruçado*).
2. **Brinquedos e Objetos (Vocabulário Aberto):** Detecção de bonecos e brinquedos (ex: boneco Toad, pelúcias, carrinhos) combinando varredura global e **inspeção em alta definição nas mãos (1080p Hand ROI)**.
3. **Diferenciação Métrica 3D:** Separação automática entre bonecos/brinquedos antropomórficos e pessoas humanas reais com base em altura métrica e distância biacromial (ombros).
4. **Proxêmica Contínua:** Análise de zonas de Edward T. Hall (*íntima*, *pessoal*, *social*, *pública*) entre todos os pares de participantes.
5. **Atenção Conjunta e Posse:** Detecção de manipulação de brinquedos pela proximidade 3D das mãos e identificação de atenção conjunta quando dois ou mais participantes interagem com o mesmo objeto.
6. **Câmera Virtual Estabilizada (Gimbal):** Janela PiP em tempo real focalizada na mão que interage com o brinquedo, com amortecimento EMA e histerese anti-trepidação.
7. **Registro Científico Automatizado:** Exportação contínua de trajetórias e eventos sociais em arquivos `.csv` com carimbo de milissegundos na pasta `recordings/`.

---

## 🏗️ Arquitetura do Sistema

```
                      +-----------------------------+
                      |   Sensor Kinect v2 (USB)    |
                      |  RGB 1080p (30Hz) + Depth   |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |    Módulo KinectSensor      |
                      | (Desprojeção Pinhole 2D->3D)|
                      +--------------+--------------+
                                     |
             +-----------------------+-----------------------+
             |                                               |
             v                                               v
+-------------------------+                     +-------------------------+
|     YOLO11s-Pose        |                     |   YOLOv8m-Worldv2       |
|  17 Keypoints (DirectML)|                     |  Detecção de Brinquedos |
+------------+------------+                     +------------+------------+
             |                                               |
             +-----------------------+-----------------------+
                                     |
                                     v
                      +-----------------------------+
                      |   Hand-Centric ROI (1080p)  |
                      |  Detecção em Alta Resolução |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | Tracker 3D (Kalman+Hungarian|
                      |   + Filtro de Articulações) |
                      +--------------+--------------+
                                     |
             +-----------------------+-----------------------+
             |                                               |
             v                                               v
+-------------------------+                     +-------------------------+
| Análise Socioenativa    |                     | Dashboard 3D & Gimbal   |
| - Classificação Postural|                     | - Imagem Colorida + 3D  |
| - Proxêmica (Hall)      |                     | - Planta Baixa da Sala  |
| - Atenção Conjunta      |                     | - Miniatura PiP da Mão  |
+------------+------------+                     +-------------------------+
             |
             v
+-------------------------+
| ScientificLogger (.csv) |
| Gravação de Trajetórias |
+-------------------------+
```

---

## ⚡ Otimizações de Desempenho e Latência

O sistema foi rigorosamente perfilado e otimizado para operar no **teto físico do Kinect v2 (30 FPS contínuos e travados)** na GPU AMD Radeon RX 6600:

| Componente Otimizado | Diagnóstico Inicial | Solução Implementada | Impacto |
| :--- | :--- | :--- | :--- |
| **Timer do Windows Kernel** | `cv2.waitKey(1)` consumia 16.7 ms por causa do quantum padrão de 15.6 ms do SO | Chamada a `ctypes.windll.winmm.timeBeginPeriod(1)` | Latência caiu para **1.89 ms** (**+14.8 ms recuperados por frame**) |
| **Renderização do Dashboard** | Criação de 4 arrays NumPy por frame (~15 MB/frame = 450 MB/s de GC) | Buffers pré-alocados gravados via slices (`dst=self.cam_view`) | Renderização caiu de 3.41 ms para **1.60 ms** (**5.2x mais rápido**) |
| **E/S de Disco (Logger)** | `open(..., 'a')` e `close()` a cada 3 frames geravam stalls no NTFS e antivírus | Descritores mantidos abertos com flush periódico | **Eliminação completa** de micro-travamentos |
| **Pós-processamento de IA** | Loops `for` em Python decodificando caixas e 17 articulações | Vetorização completa em C/NumPy | Decodificação em **0.30 ms** (**6x mais rápido**) |
| **Câmera de Zoom PiP** | Trepidação e alternância rápida entre as mãos | Filtro suave EMA ($\alpha=0.20$), deadzone de 3px e histerese de troca (6 frames) | Imagem estável com sensação de Steadicam |

---

## 📂 Estrutura do Repositório

```text
teste/
├── .gitattributes                 # Normalização de quebras de linha e arquivos binários
├── .gitignore                     # Regras de exclusão do Git (.venv, pesos, gravações)
├── LICENSE                        # Licença de código aberto (MIT)
├── README.md                      # Documentação completa do projeto
├── requirements.txt               # Dependências Python essenciais
├── main.py                        # Ponto de entrada do Hub Principal (Processamento e Dashboard)
├── edge_node.py                   # Ponto de entrada do Nó Secundário (Multi-câmeras)
├── recordings/                    # Datasets científicos salvos (.csv são ignorados no Git)
│   └── .gitkeep
├── weights/                       # Modelos de rede neural (ONNX/PyTorch ignorados no Git)
│   ├── .gitkeep
│   ├── README.md                  # Instruções para download dos modelos
│   ├── yolo11s-pose.onnx          # Rastreamento de 17 articulações humanas
│   └── yolov8m-worldv2.onnx       # Detecção de brinquedos (vocabulário aberto)
├── tests/                         # Scripts de validação e testes
│   ├── test_kinect_capture.py     # Validação da captura do Kinect v2
│   └── test_system_pipeline.py    # Validação sintética do pipeline completo
├── tools/                         # Utilitários e benchmarks
│   ├── benchmark_directml.py      # Medição de latência dos modelos na GPU AMD
│   └── download_weights.py        # Download e exportação automática dos modelos ONNX
└── src/                           # Módulos do sistema
    ├── ai/
    │   ├── directml_inference.py  # Motor de inferência ONNX Runtime via DirectML (AMD GPU)
    │   ├── hand_roi_detector.py   # Recorte métrico de alta resolução nas mãos (1080p)
    │   └── depth_clustering_3d.py # Segmentação Geométrica 3D por Nuvem de Pontos
    ├── core/
    │   ├── config.py              # Dataclasses de configuração e resolução de pesos
    │   └── kinect_sensor.py       # Driver PyKinect2 otimizado com unprojection métrica
    ├── network/
    │   └── network_protocol.py    # Comunicação UDP para rede de múltiplos sensores (Hub/Edge)
    ├── socioenative/
    │   ├── posture_classifier.py  # Classificador baseado na geometria das articulações 3D
    │   ├── proxemics.py           # Análise de zonas proxêmicas (Edward T. Hall)
    │   ├── toy_interaction.py     # Detecção de posse de brinquedos e atenção conjunta
    │   └── scientific_logger.py   # Gravador científico em buffer contínuo de alta velocidade
    ├── tracking/
    │   ├── tracker_3d.py          # Associação espacial 3D multi-alvo (Algoritmo Húngaro)
    │   ├── kalman_filter_3d.py    # Filtro de Kalman com modelo de velocidade constante
    │   └── keypoint_filter.py     # Filtro temporal EMA para esqueletos sem trepidação
    └── visualization/
        └── dashboard_3d.py        # Dashboard multi-painel com 3 modos de exibição e HUD interativo
```

---

## 🛠️ Requisitos de Hardware e Software

### Hardware Recomendado
- **Sensor:** Microsoft Kinect for Windows v2 (com adaptador de energia e USB 3.0 dedicado).
- **GPU Principal:** AMD Radeon RX 6600 (ou superior) com suporte a DirectX 12.
- **Processador:** AMD Ryzen 5 / Intel Core i5 de 6 núcleos ou superior.
- **RAM:** 16 GB DDR4/DDR5.

### Software
- **Sistema Operacional:** Windows 10 ou Windows 11 (64-bit).
- **Driver Oficial:** [Kinect for Windows SDK v2.0](https://www.microsoft.com/en-us/download/details.aspx?id=44561).
- **Interpretador:** Python 3.11 (64-bit).

---

## 🚀 Guia de Instalação

### 1. Clonar o Repositório e Criar Ambiente Virtual
Abra o PowerShell no diretório do projeto:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Instalar Dependências
```powershell
pip install -r requirements.txt
```

### 3. Configurar Aceleração DirectML para GPU AMD
> [!IMPORTANT]
> A biblioteca padrão `onnxruntime` executará apenas na CPU. Para utilizar a GPU **AMD Radeon RX 6600**, certifique-se de que o pacote `onnxruntime-directml` está instalado:

```powershell
pip uninstall -y onnxruntime
pip install --force-reinstall --no-deps onnxruntime-directml
```

Para verificar se a GPU AMD foi detectada:
```powershell
.\.venv\Scripts\python.exe -c "import onnxruntime as ort; print('Provedores:', ort.get_available_providers())"
```
*Deve constar `['DmlExecutionProvider', 'CPUExecutionProvider']`.*

### 4. Obter Modelos de Rede Neural (ONNX)
Caso esteja iniciando em uma máquina nova sem os modelos locais na pasta `weights/`, execute:
```powershell
.\.venv\Scripts\python.exe tools/download_weights.py
```

---

## 🎮 Como Executar

### 1. Executar o Hub Principal (Kinect v2 + GPU AMD)
Certifique-se de que o Kinect v2 está conectado à porta USB 3.0 (LED branco aceso) e execute:
```powershell
.\.venv\Scripts\python.exe main.py
```

### 2. Executar Nó Secundário (Opcional - Rede Multi-Câmeras)
Se houver um segundo PC ou notebook (ex: com GPU NVIDIA GTX 1060 e outro Kinect v2):
```powershell
.\.venv\Scripts\python.exe edge_node.py --hub-ip 192.168.1.100 --hub-port 5555 --sensor-id kinect_sala_2
```

### 3. Atalhos e Controles Interativos da Interface (Fase 4)
O sistema permite alternar telas e elementos visuais em tempo real durante a execução:

| Tecla | Função | Descrição |
| :---: | :--- | :--- |
| **`1`** | **Dashboard Triplo Científico** | Modo padrão completo: Câmera (960x540) + Planta Baixa (640x540) + Métricas (1600x360). |
| **`2`** | **Planta Baixa 3D (Tela Cheia)** | Expande a sala em 1600x900 com grid métrico ampliado e trajetórias detalhadas. |
| **`3`** | **Câmera Realidade Aumentada** | Visão 1080p da câmera em 1600x900 com esqueletos, caixas 3D e caminhos neon no chão. |
| **`T`** | **Trajetórias Neon** | Liga/desliga as linhas contínuas com gradiente temporal que seguem as pessoas. |
| **`S`** | **Esqueleto Anatômico** | Alterna entre: *Completo (17 keypoints)* $\rightarrow$ *Apenas Mãos/Cabeça* $\rightarrow$ *Oculto*. |
| **`M`** | **Menu de Ajuda (HUD)** | Exibe/oculta um card translúcido no canto da tela com os atalhos rápidos. |
| **`ESC` / `Q`** | **Sair** | Encerra a captura e salva com segurança todos os dados científicos na pasta `recordings/`. |

### 4. Scripts de Teste e Validação
- **Validar Captura do Sensor Kinect v2 (Frames RGB e Profundidade)**:
  ```powershell
  .\.venv\Scripts\python.exe tests/test_kinect_capture.py
  ```
- **Validar Pipeline Completo (Inferência, Rastreamento e Métricas)**:
  ```powershell
  .\.venv\Scripts\python.exe tests/test_system_pipeline.py
  ```
- **Aferição de Latência e Desempenho da GPU (DirectML Benchmark)**:
  ```powershell
  .\.venv\Scripts\python.exe tools/benchmark_directml.py
  ```

---

## 📊 Estrutura dos Arquivos de Dados Científicos

Os dados são salvos na pasta `recordings/` com resolução temporal de ~10 Hz:

### 1. `trajectories_YYYYMMDD_HHMMSS.csv`
Contém o deslocamento e estado físico de cada indivíduo:
| Coluna | Tipo | Descrição |
| :--- | :--- | :--- |
| `timestamp_ms` | Inteiro | Marcação temporal absoluta em milissegundos (Epoch) |
| `track_id` | Inteiro | Identificador persistente da pessoa na sala |
| `pos_x` | Float | Posição lateral no espaço em metros (Kinect = 0.0) |
| `pos_y` | Float | Posição vertical em metros |
| `pos_z` | Float | Distância de profundidade da câmera em metros |
| `speed_mps` | Float | Velocidade linear instantânea em metros por segundo |
| `posture` | String | Postura (*em_pe*, *sentado_chao*, *agachado*, *debruçado*) |
| `held_toy` | String | Nome do brinquedo em posse do indivíduo (ou *nenhum*) |

### 2. `interactions_YYYYMMDD_HHMMSS.csv`
Registra a dinâmica socioespacial e atenção conjunta:
| Coluna | Tipo | Descrição |
| :--- | :--- | :--- |
| `timestamp_ms` | Inteiro | Marcação temporal em milissegundos |
| `track_id_1` | Inteiro | ID da primeira pessoa |
| `track_id_2` | Inteiro | ID da segunda pessoa |
| `distance_m` | Float | Distância euclidiana 3D entre o par em metros |
| `proxemic_zone` | String | Zona de Hall (*intima*, *pessoal*, *social*, *publica*) |
| `joint_toy` | String | Nome do brinquedo compartilhado (caso haja atenção conjunta) |
| `event_type` | String | Categoria do evento (*proxemica* ou *atencao_conjunta*) |

---

## 🔬 Modelos de Redes Neurais Utilizados

- **Pose Humana:** `yolo11s-pose.onnx` — 17 articulações anatômicas (COCO Keypoints), inferência média de **10.2 ms** na RX 6600.
- **Objetos e Brinquedos:** `yolov8m-worldv2.onnx` (Backbone Médio) — inferência média de **17.8 ms** na RX 6600.
  - Vocabulário aberto compilado para: `boneco_toad`, `pelucia`, `action_figure`, `boneco`, `brinquedo`, `carrinho`, `bola`.
- **Inspeção de Mão:** Janela de 55 cm no espaço métrico real projetada a partir da direção do antebraço ($\text{pulso} + 0.25 \times (\text{pulso} - \text{cotovelo})$), permitindo detectar objetos pequenos mesmo com a pessoa a mais de 1.30 m da câmera.

---

## 📄 Licença e Créditos
Desenvolvido para pesquisa em Interação Humano-Computador (IHC), Ambientes Socioenativos e Computação Ubíqua.
Baseado nos conceitos arquiteturais do ecossistema [OpenPTrack](https://github.com/OpenPTrack/openptrack_v2).
