# Módulos Experimentais e Legados (Arquivamento - Issue T-03)

Este diretório contém protótipos e implementações exploratórias desenvolvidas nas fases iniciais do projeto **SICAMO3D**, desativadas no pipeline de produção em favor de arquiteturas mais eficientes e estáveis.

---

## 1. Módulos Arquivados

### `toy_interaction.py`
- **Função original:** Heurística baseada em distâncias euclidianas 3D entre mãos (`hands_3d`) e brinquedos (`toys_3d`) com lógica simples de atenção compartilhada.
- **Motivo do arquivamento:** Permitía atribuição simultânea espúria do mesmo objeto a múltiplos portadores e não tratava histerese temporal nem transferências de posse (handoffs).
- **Substituto oficial:** [`src/socioenative/holder_inference.py`](../socioenative/holder_inference.py) — que garante exclusividade estrita de portador único, histerese temporal com limiares de aquisição/manutenção configuráveis e registro de handoffs em conformidade com o protocolo socioenativo.

### `hand_roi_detector.py`
- **Função original:** Inspeção recortada em alta definição (crop ROI 2D com Time-To-Live) em torno dos pulsos para detecção focada de objetos pequenos.
- **Motivo do arquivamento:** Gerava overhead substancial de inferência (múltiplas passagens no DirectML por frame) e era suscetível a tremores nas caixas delimitadoras. Foi superado pela detecção single-pass direta da pelúcia em resolução nativa pelo detector YOLO ajustado.

### `depth_clustering_3d.py`
- **Função original:** Agrupamento de nuvem de pontos 3D (*depth point cloud clustering*) ao redor das mãos para confirmação física de volume agnóstica a cores e iluminação.
- **Motivo do arquivamento:** Custoso computacionalmente no fluxo de tempo real e sensível a oclusões e ruído do sensor de profundidade. Mantido aqui como referência algorítmica para futuras extensões de segmentação volumétrica.

---

## 2. Compatibilidade e Uso

Estes módulos **não** devem ser importados pelo pipeline em produção (`src/core/pipeline.py`). Caso sejam necessários em scripts de validação cruzada ou pesquisas offline, importe-os explicitamente:

```python
from src.experimental.toy_interaction import ToyInteractionDetector
from src.experimental.hand_roi_detector import HandROIDetector
from src.experimental.depth_clustering_3d import DepthCluster3D
```
