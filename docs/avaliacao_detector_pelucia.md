# Protocolo de Avaliação do Detector da Pelúcia (M2-03)

Este documento estabelece a metodologia experimental, métricas e matriz de testes para validação do detector da pelúcia (`yolo11s_plush.onnx` e fallbacks) no espaço cênico do projeto SICAMO3D.

---

## 1. Objetivos da Avaliação

1. Quantificar a precisão e completude da detecção (`mAP50`, `Precision`, `Recall`) em distâncias operacionais reais do sensor Kinect v2 (2.0m a 4.0m).
2. Avaliar a robustez do detector frente a oclusões parciais típicas da interação teatral (segurar pelas extremidades, apoiar contra o tronco, abraçar).
3. Avaliar a estabilidade temporal das detecções ao longo de sequências contínuas de vídeo (taxa de dropout / flicker).

---

## 2. Matriz Experimental de Teste

A avaliação é estruturada em uma grade $3 \times 3$ (3 distâncias $\times$ 3 situações de interação), totalizando 9 condições experimentais com 30 segundos de captura contínua a 30 FPS cada (~900 frames por condição, ~8.100 frames no total):

| ID Condição | Distância ($Z$) | Situação de Interação | Nível de Oclusão Esperado |
| :--- | :--- | :--- | :--- |
| **C1-D2-REP** | 2.0 m | Repouso (sobre mesa cênica ou chão) | 0% (sem oclusão) |
| **C2-D2-MAO** | 2.0 m | Nas mãos (estendida ou manipulada) | 10% a 25% (apenas dedos/palmas) |
| **C3-D2-ABR** | 2.0 m | Abraçado junto ao tronco / peito | 30% a 60% (braços e antebraços cruzados) |
| **C4-D3-REP** | 3.0 m | Repouso (sobre mesa cênica ou chão) | 0% (sem oclusão) |
| **C5-D3-MAO** | 3.0 m | Nas mãos (estendida ou manipulada) | 10% a 25% (apenas dedos/palmas) |
| **C6-D3-ABR** | 3.0 m | Abraçado junto ao tronco / peito | 30% a 60% (braços e antebraços cruzados) |
| **C7-D4-REP** | 4.0 m | Repouso (sobre mesa cênica ou chão) | 0% (sem oclusão, resolução aparente reduzida) |
| **C8-D4-MAO** | 4.0 m | Nas mãos (estendida ou manipulada) | 10% a 25% |
| **C9-D4-ABR** | 4.0 m | Abraçado junto ao tronco / peito | 30% a 60% |

---

## 3. Métricas de Avaliação

### 3.1 Métricas de Visão Computacional (Frame a Frame)
- **mAP@50 (Mean Average Precision em IoU=0.50):** Avaliado contra as anotações manuais ground truth no formato YOLO (`pelucia`).
- **Precision ($P$):** $\frac{TP}{TP + FP}$ — fração das detecções emitidas que correspondem à pelúcia real.
- **Recall ($R$):** $\frac{TP}{TP + FN}$ — fração dos frames com pelúcia em cena que foram detectados.
- **Recall em Oclusão Parcial ($R_{ocl}$):** Calculado especificamente sobre os subconjuntos de teste com situação "Abraçado" (C3, C6, C9).

### 3.2 Métricas de Estabilidade Temporal (Sequência Contínua)
- **Taxa de Flicker:** Proporção de transições de presença/ausência frame a frame ($N_{transitions} / N_{frames}$).
- **Comprimento Máximo de Falha Consecutiva ($L_{max\_gap}$):** Número máximo de frames consecutivos em que a pelúcia visível em cena não produziu nenhuma bounding box válida com confiança $\ge 0.35$.
- **Latência de Recuperação ($T_{recov}$):** Tempo (em ms ou frames) necessário para o detector reencontrar o objeto após oclusão total transitória.

---

## 4. Critérios de Aceitação (Pass / Fail)

Para homologação do modelo treinado no Milestone M2:

1. **Recall Geral a até 3.5m:** $\ge 85\%$ em condições de repouso e nas mãos.
2. **Recall em Oclusão Parcial a até 3.0m:** $\ge 70\%$ na condição abraçada.
3. **Persistência Temporal:** O detector não deve perder a pelúcia por mais de **5 frames consecutivos** ($\sim 160\text{ ms}$ a 30 FPS) a até $3.5\text{ m}$ em movimentação cênica normal ($v \le 1.2\text{ m/s}$).
4. **Falsos Positivos em Cenário Vazio:** 0 detecções persistentes (nenhum cluster $\ge 3$ frames falsos em 2 minutos de cena sem o objeto).

---

## 5. Procedimento de Execução do Teste

1. Posicionar o sensor Kinect v2 calibrado no pedestal central a $1.4\text{ m}$ de altura.
2. Demarcar no chão as linhas de $2.0\text{ m}$, $3.0\text{ m}$ e $4.0\text{ m}$ usando fita crepe e conferir com trena.
3. Gravar sessões curtas com a flag `--save-raw` ativada:
   ```powershell
   python main.py --mode B --save-raw --weights weights/yolo11s_plush.onnx
   ```
4. Processar e comparar os resultados contra as anotações de referência ou marcações manuais do Modo A através da ferramenta de diagnóstico:
   ```powershell
   python tools/compare_modes.py --session output/sessions/sessao_teste --ground-truth data/annotations/gt_labels.csv
   ```
5. Anexar o relatório gerado na documentação de validação da release M2.
