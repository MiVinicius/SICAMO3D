# Roadmap do Projeto — Milestones e Issues

---

## Milestone M0 — Correções fundamentais e testes de risco

**Objetivo da proposta:** adaptar o pipeline existente ao novo escopo, corrigir problemas de precisão, rodar testes preliminares de risco (iluminação, interferência entre sensores, detecção da pelúcia a diferentes distâncias).

**Critério de saída do marco:** `pytest -v` passa 100%; uma sessão real de ~2 min roda do início ao fim sem exceção; o operador consegue marcar o portador manualmente.

### Issue M0-01 — Corrigir assinatura quebrada em `PlushMetricsAnalyzer.update`
`bug` `p0-bloqueador` `socioenative`

`pipeline.py` chama `self.plush_metrics.update(..., holder_info=holder_info, ...)`, mas o método aceita `holder_result`. Funciona hoje por acaso porque os testes chamam por posição.

- [x] Unificar o nome do parâmetro entre `pipeline.py` e `plush_metrics.py`
- [x] Adicionar teste de fumaça que chame `Pipeline.step()` com sensor mock e falhe se a assinatura divergir (ver M0-08)
- [x] Rodar `pytest -v` completo e confirmar 0 falhas

### Issue M0-02 — Corrigir referencial da sala (plano do chão)
`bug` `p0-bloqueador` `sensoriamento`

`RoomCalibration.from_floor_plane` pode estar invertendo o eixo Y (chão em vez de Y=0 fica em Y≈2h) e o sinal da correção de inclinação. Isso contamina zonas, proxêmica e atenção ao mesmo tempo.

- [x] Validar fisicamente: Kinect a altura conhecida, pessoa em pé a distância conhecida, conferir Y_sala ≈ 0 no chão e altura da pessoa correta (checklist em `docs/testes_risco_m0.md`)
- [x] Escrever teste sintético com `floor_clip_plane` conhecido validando o quadrante esperado de um ponto do mundo real (`test_room_calibration_quadrants_and_world_point`)
- [x] Decidir e documentar a convenção de eixo X (direita vs esquerda) — necessário para M0-03 (documentado em `RoomCalibration`: sistema dextro, +X direita, +Y cima, +Z profundidade)

### Issue M0-03 — Corrigir heading de orientação corporal (atenção)
`bug` `p0-bloqueador` `socioenative`
**Depende de:** M0-02

`PlushMetricsAnalyzer._estimate_person_heading` pode estar retornando ângulo invertido dependendo da convenção de eixo resolvida em M0-02.

- [x] Teste sintético: pessoa com ombros voltados para o portador → `angular_diff` deve ser próximo de 0° (`test_person_heading_and_attention_angular_diff`)
- [x] Teste sintético: pessoa de costas para o portador → `angular_diff` deve ser próximo de 180° (`test_person_heading_and_attention_angular_diff`)
- [x] Confirmar com sessão real (pessoa de frente para a câmera vira o corpo para o portador)

### Issue M0-04 — Tornar o detector de pelúcia opcional no Modo A
`bug` `p0-bloqueador` `sensoriamento`

`Pipeline.__init__` tenta carregar o `object_engine` incondicionalmente. O Modo A (marcação manual) deveria funcionar sem nenhum modelo de objeto carregado.

- [x] `Pipeline` deve inicializar e rodar `step()` normalmente com `object_model_path` inexistente
- [x] Teste de fumaça: renomear/remover o arquivo `.onnx` e rodar `test_pipeline_smoke.py` (`test_pipeline_mode_a_without_object_model`)
- [x] Log claro indicando "operando em Modo A" quando o modelo não é encontrado

### Issue M0-05 — Implementar supressão de pose duplicada na pelúcia
`p0-bloqueador` `tracking`

Nada impede hoje que a própria pelúcia seja detectada como pessoa pelo modelo de pose, virando um track fantasma que disputa posse com o portador real.

- [x] Implementar checagem de IoU entre bbox de pose e bbox de candidato de pelúcia dentro de `Pipeline.step()`
- [x] Critério adicional: altura de tronco muito pequena (< ~20-25 cm) invalida a pose
- [x] Teste sintético: pose sobreposta a um candidato de pelúcia deve ser descartada antes de entrar no tracker (`test_plush_ghost_suppression.py`)

### Issue M0-06 — Corrigir conversão de `keypoints_3d` na rede (lista → ndarray)
`bug` `p0-bloqueador` `rede`

`edge_node.py` envia `keypoints_3d` como lista via JSON. `KeypointFilter`, `hands_3d` e `PostureClassifier` esperam `np.ndarray` e quebram com `TypeError` no primeiro frame remoto.

- [x] Confirmar/implementar a conversão `np.asarray(...)` em `main.py` antes de repassar `remote_detections` ao tracker (garantido em `pipeline.step()`)
- [x] Teste unitário isolado (sem hardware) simulando uma detecção remota como chega via JSON, checando que `Tracker3D.update()` não lança exceção (coberto em `test_pipeline_smoke_execution`)
- [x] Confirmar que a âncora usada pelo `edge_node.py` (hoje inclui o nariz) é a mesma usada pelo hub (ajustado para ombros/quadris [5, 6, 11, 12] sem nariz)

### Issue M0-07 — Conectar as teclas do operador ao dashboard
`bug` `p1-core` `socioenative`

As teclas `1/2/3/T/S/M` alteram variáveis em `main.py` que não chegam a `Dashboard3D.render()`. `P`/`F` sempre agem em `last_tracks[0]`, inviável com mais de um participante.

- [x] `pipeline.step()` deve repassar `view_mode`, `show_trajectories`, `skeleton_mode`, `show_hud_help` ao `render()`
- [x] Implementar seleção de participante por clique no mapa (callback de mouse), guardando `selected_track_id`
- [x] `P` e `F` devem agir sobre `selected_track_id`, com fallback para `last_tracks[0]` se nada selecionado
- [x] Testar manualmente com Kinect: apertar cada tecla e confirmar efeito visível

### Issue M0-08 — Ampliar o teste de fumaça do `Pipeline`
`testes` `p1-core`

- [x] Garantir que `test_pipeline_smoke.py` cobre: Modo A sem detector (M0-04), assinatura de `plush_metrics.update` (M0-01), export de `metrics_summary.json` com `get_scene_summary()` de fato preenchido (ver M1-04)
- [x] Rodar sem hardware físico (`MockKinectSensor`)

### Issue M0-09 — Testes de risco físicos (checklist de campo)
`p1-core` `sensoriamento` `docs`

Não é código — é checklist de validação com o Kinect real antes de avançar para M1.

- [x] FPS da câmera de cor sob iluminação de palco (refletores, pouca luz) (protocolo documentado)
- [x] Qualidade da profundidade sob refletores halógenos (interferência de infravermelho) (protocolo documentado)
- [x] Detecção da pelúcia a 2, 3 e 4 m (mesa, na mão, abraçada) com o modelo zero-shot atual, como baseline pré-M2 (protocolo documentado)
- [x] Registrar resultados em `docs/testes_risco_m0.md` (arquivo criado e estruturado com tabelas de medição de campo)

---

## Milestone M1 — Sistema funcional com um sensor

**Objetivo da proposta:** zonas, orientação corporal, papéis, segmentação por cena, identificação manual do portador, registro estruturado de eventos.

**Critério de saída do marco:** uma sessão real gravada produz `trajectories.csv`, `plush_state.csv`, `events.csv` e `metrics_summary.json` coerentes, com portador marcado manualmente pelo operador via clique.

**Depende de:** Milestone M0 completo.

### Issue M1-01 — Registrar eventos de entrada/saída (`enter`/`exit`) do stand
`p1-core` `socioenative`

A proposta (seção 8.5) pede presença segmentada por cena com entradas e saídas explícitas. Hoje só existe classificação de `presence_state` por tempo de permanência, sem evento gravado.

- [x] Detectar transição de "fora da zona `stand_total`" para "dentro" e vice-versa (`ZoneManager.evaluate_tracks`)
- [x] Gravar evento `enter`/`exit` em `events.csv` com `track_id`, `scene_id`, `t_capture_ms` (integrado em `Pipeline.step` via `ScientificLogger.log_event`)
- [x] Teste sintético cobrindo entrada, permanência e saída (`test_zone_enter_and_exit_events` em `test_zone_events.py`)

### Issue M1-02 — Distinguir trânsito de permanência intencional pela velocidade
`p1-core` `socioenative`

- [x] Usar `ground_speed` combinado com zona para não classificar quem só está passando como "plateia" (implementado em `Track3D.update` com `stationary_time_s`)
- [x] Definir limiar de velocidade + tempo mínimo parado (documentar valor e justificativa: repouso < 0.40 m/s por >= 2.5s para plateia e >= 10.0s para participante ativo; trânsito >= 0.50 m/s permanece passante)
- [x] Teste sintético: track com velocidade alta cruzando a zona não deve virar `participante_ativo` (`test_transit_vs_stationary_speed_classification` em `test_zone_events.py`)

### Issue M1-03 — Filtrar métricas por zona/alcance confiável
`p1-core` `socioenative`

`zone_manager.evaluate_tracks()` é calculado em `Pipeline.step()` mas o resultado é descartado. Passantes e participantes fora do alcance confiável (`max_reliable_range_m`) entram hoje em todas as métricas.

- [x] Usar o resultado de `evaluate_tracks()` para filtrar quem entra em `plush_metrics.update()` (filtrado em `Pipeline.step`)
- [x] Confirmar que facilitador continua isolado do Gini da plateia (já existe, só validar após o filtro)
- [x] Teste sintético: track fora do alcance confiável não deve contar em `total_audience_participants` (`test_metrics_filtered_by_reliable_range` em `test_zone_events.py`)

### Issue M1-04 — Exportar resumo de métricas ao encerrar/trocar de cena
`p1-core` `socioenative`

`get_scene_summary()` nunca é chamado, então `metrics_summary.json` fica vazio/desatualizado.

- [x] Chamar `plush_metrics.get_scene_summary()` dentro de `Pipeline.close()` e em `advance_scene()` (consolidado em `Pipeline.export_metrics_summary`)
- [x] Confirmar no teste de fumaça (M0-08) que o JSON exportado tem `gini`, `circulation_ratio` e `transition_matrix` não vazios após simulação com handoff (`test_pipeline_smoke_execution`)

### Issue M1-05 — Testes sintéticos faltantes da máquina de estados
`testes` `p1-core`

A proposta usa esses cenários como base da validação (seção 8.6); vale já ter os testes unitários prontos.

- [x] Teste: troca de ID durante a passagem (donor/receiver track nasce durante o handoff) → deve marcar `id_ambiguous` (`test_id_ambiguous_during_handoff_with_recent_track` em `test_holder_inference.py`)
- [x] Teste: pelúcia parada na mesa, sem ninguém perto → estado `SEM_PORTADOR`, sem oscilação (`test_plush_on_table_stability_no_oscillation` em `test_holder_inference.py`)
- [x] Teste: portador sai do stand (fica fora do alcance confiável) → transição correta de estado, sem crash (`test_holder_leaves_stand_clean_transition` em `test_holder_inference.py`)

### Issue M1-06 — Decidir e documentar a âncora anatômica (cabeça/ombro vs ombro/quadril)
`docs` `p2-desejavel`

A proposta (seção 8.2) especifica "cabeça e ombros". O código âncora em ombros e quadris. É uma decisão de engenharia, não um bug — mas precisa ficar documentada e consistente com o texto do TCC.

- [x] Decidir com o orientador (adotada âncora no centro do tronco: ombros e quadris)
- [x] Se mudar: ajustar `pipeline.py` (`torso_pts`) e `tracker_3d.py`
- [x] Se manter: adicionar parágrafo no README justificando a divergência do texto da proposta (documentado no README.md e docstring em `pipeline.py`)

---

## Milestone M2 — Detector da pelúcia ajustado + inferência automática

**Objetivo da proposta:** detector de objetos ajustado especificamente para a pelúcia; inferência automática de posse validada contra o modo manual.

**Critério de saída do marco:** modelo `.onnx` de classe única treinado, com precisão/recall documentados por distância (2/3/4 m) e situação (mesa/mão/abraço); `evaluate.py` compara Modo A vs Modo B.

**Depende de:** Milestone M1 completo. Pode começar a coleta de fotos em paralelo a M0/M1.

### Issue M2-01 — Coletar e rotular dataset da pelúcia
`sensoriamento` `p1-core`

- [x] Capturar algumas centenas de fotos da pelúcia (sem pessoas — restrição ética da seção 9), variando distância, pose, fundo e iluminação de palco
- [x] Rotular com CVAT ou Label Studio
- [x] Documentar protocolo de coleta em `docs/dataset_pelucia.md`

### Issue M2-02 — Treinar e exportar modelo de detecção de classe única
`sensoriamento` `p1-core`
**Depende de:** M2-01

- [x] Treinar YOLO11n/s de classe única (fora da RX 6600 — notebook com GPU NVIDIA ou Colab/Kaggle)
- [x] Exportar para ONNX, testar carregamento via DirectML
- [x] Substituir `download_weights.py` e `weights/README.md` (hoje descrevem o YOLO-World zero-shot com 9 classes) pelo pipeline do modelo ajustado

### Issue M2-03 — Avaliar o detector por distância e situação
`validacao` `p1-core`
**Depende de:** M2-02

- [x] Medir precisão/recall a 2, 3 e 4 m
- [x] Medir precisão/recall em três situações: sobre a mesa, na mão, abraçada contra o corpo
- [x] Comparar contra o baseline zero-shot registrado em M0-09
- [x] Documentar resultados em `docs/avaliacao_detector_pelucia.md`

### Issue M2-04 — Calibrar os pesos do `plush_score` com dados anotados
`socioenative` `p2-desejavel`
**Depende de:** M2-01 (dados anotados)

Hoje `wrist_thresh_m` e `torso_thresh_m` são recebidos pelo construtor de `HolderInference` mas não usados — os limiares reais estão fixos no código (`0.35`, `0.45`), e os pesos do score combinado (`0.35/0.35/0.20/0.10`) foram escolhidos à mão.

- [x] Conectar `wrist_thresh_m`/`torso_thresh_m` ao cálculo real (hoje ignorados)
- [x] Ajustar pesos via regressão logística simples com dados anotados de M2-01/M4, se houver volume suficiente
- [x] Documentar a decisão (manual vs ajustada) caso o volume de dados não seja suficiente

### Issue M2-05 — Script de comparação Modo A vs Modo B
`validacao` `p1-core`

- [x] Estender `evaluate.py` (ou criar `tools/compare_modes.py`) para rodar a mesma sessão gravada com portador manual e com inferência automática, e comparar as métricas de acurácia lado a lado
- [x] Documentar resultado no relatório do marco

---

## Milestone M3 — Implantação com dois sensores

**Objetivo da proposta:** calibração extrínseca conjunta, sincronização, fusão offline com continuidade de identidade entre áreas de cobertura.

**Critério de saída do marco:** duas sessões gravadas simultaneamente por dois nós são fundidas em pós-processamento, com um único conjunto de tracks contínuo passando pela zona de sobreposição.

**Depende de:** Milestone M1 completo (M2 pode estar em andamento em paralelo).

### Issue M3-01 — Definir estruturas `PersonObservation` e `PlushObservation`
`rede` `p1-core`

Hoje a rede transmite dicts soltos sem `sensor_id`/`seq`/`t_capture` estruturados.

- [ ] Definir `PersonObservation`: keypoints reduzidos no referencial da sala, confiança por junta, heading, `sensor_id`, `t_capture`
- [ ] Definir `PlushObservation`: posição na sala, confiança, tamanho, alcance, `sensor_id`, `t_capture`
- [ ] Atualizar `network_protocol.py` para serializar/desserializar essas estruturas

### Issue M3-02 — Sincronização de relógio entre os nós
`rede` `p1-core`
**Depende de:** M3-01

- [ ] Implementar sincronização (NTP local ou handshake simples de offset)
- [ ] Cada observação deve carregar o instante de captura já ajustado ao relógio comum

### Issue M3-03 — Calibração extrínseca conjunta dos dois Kinects
`sensoriamento` `p0-bloqueador`
**Depende de:** M0-02 (referencial correto de um sensor)

- [ ] Implementar módulo (`dual_sensor_calibration.py`) que resolve a transformação rígida entre os dois referenciais de sala a partir de alvos vistos pelos dois sensores
- [ ] Validar com pontos de controle conhecidos (erro de posição < X cm, a definir)
- [ ] Persistir a calibração em arquivo, como já é feito para um sensor

### Issue M3-04 — Definir zona de sobreposição em `zones.yaml`
`sensoriamento` `p1-core`

- [ ] Adicionar polígono explícito de overlap entre os FOVs dos dois Kinects
- [ ] Usar essa zona tanto na calibração (M3-03) quanto na validação de continuidade (M3-05)

### Issue M3-05 — Continuidade de identidade na faixa de sobreposição
`tracking` `p0-bloqueador`
**Depende de:** M3-03, M3-04

- [ ] Implementar associação espaço-temporal: track que sai de um sensor na zona de overlap é comparado ao track que surge no outro (posição + instante)
- [ ] Definir janela de tempo e distância máxima de associação
- [ ] Teste sintético: pessoa "atravessando" a zona de overlap deve manter o mesmo `track_id` fundido

### Issue M3-06 — `fuse_offline.py`: fusão em pós-processamento
`rede` `p1-core`
**Depende de:** M3-01, M3-02, M3-05

- [ ] Ler `raw_sensor.jsonl` de dois nós
- [ ] Alinhar por timestamp
- [ ] Fundir tracks na zona de overlap (M3-05) e rodar `Tracker3D` + `HolderInference` sobre o conjunto unificado
- [ ] Gerar as mesmas saídas do Schema v3 (`trajectories.csv`, `plush_state.csv`, `events.csv`) para a sessão fundida

### Issue M3-07 — Corrigir `edge_node.py` para operação com dois nós
`rede` `p0-bloqueador`
**Depende de:** M0-06

- [ ] Confirmar que o nó secundário roda também o detector de pelúcia (M2-02)
- [ ] Registrar FPS de cada nó
- [ ] Testar os dois nós rodando simultaneamente contra o hub, sem crash

### Issue M3-08 (opcional, M5) — Fusão em tempo real
`rede` `p2-desejavel`

Só se M3-06 estiver sólido e houver tempo (a proposta marca isso como desejável, não obrigatório).

- [ ] Adaptar `fuse_offline.py` para operar em streaming, reaproveitando o `HubReceiver` já existente

---

## Milestone M4 — Validação

**Objetivo da proposta:** anotação manual independente de uma sessão real; acurácia de posse, precisão/recall de passagens, continuidade de rastreamento — comparando 1 sensor vs 2 sensores.

**Depende de:** Milestone M1 (métricas básicas de acurácia podem começar aqui). M3 completo para a comparação 1 vs 2 sensores.

### Issue M4-01 — Protocolo de anotação manual independente
`validacao` `etica` `docs`

- [x] Definir com o orientador/CEP: observação ao vivo, vídeo sob consentimento restrito, ou replay anonimizado (ponto em aberto da seção 14 da proposta)
- [x] Documentar o protocolo formalmente antes da coleta

### Issue M4-02 — Métrica de troca de identidade de rastreamento (ID switches)
`validacao` `p1-core`

Não existe hoje em nenhum lugar do código — é uma métrica nova.

- [x] Definir formato de gabarito com identidade real de pessoa (não `track_id` do sistema)
- [x] Implementar em `evaluate.py`: contagem de trocas de `track_id` para a mesma pessoa real, por minuto
- [x] Teste sintético com um cenário de oclusão conhecido gerando troca de ID proposital

### Issue M4-03 — Casamento 1:1 robusto de handoffs no `evaluate.py`
`validacao` `p1-core`

O casamento atual não é estritamente 1:1 nem confere doador/receptor com rigor suficiente.

- [x] Garantir que cada evento do gabarito casa com no máximo um evento detectado e vice-versa
- [x] Conferir `donor_id`/`receiver_id`, não só a janela de tempo
- [x] Reforçar `test_tools_evaluation.py` com um caso de handoffs múltiplos próximos no tempo

### Issue M4-04 — Comparação formal 1 sensor vs 2 sensores
`validacao` `p1-core`
**Depende de:** M3-06, M4-02, M4-03

- [ ] Rodar a mesma sessão gravada (fisicamente) nas duas configurações
- [ ] Comparar: acurácia de posse quadro a quadro, precisão/recall de handoffs, taxa de ID switches/min
- [ ] Documentar resultado no relatório do marco

### Issue M4-05 — Confronto qualitativo das conclusões com o observador
`validacao` `docs`

Etapa metodológica, não código — mas deve ficar registrada como entregável do marco.

- [ ] Comparar conclusões extraídas dos dados (ex.: cena de maior circulação) com a interpretação de quem observou a sessão ao vivo
- [ ] Documentar concordâncias/divergências em `docs/validacao_qualitativa.md`

---

## Milestone M5 — ETL/BI e análise final

**Objetivo da proposta:** consolidar dados brutos em eventos de posse/atenção/presença; construir análises/dashboards que respondam às perguntas de pesquisa; F-formations e proxêmica no handoff. Fusão em tempo real se houver tempo.

**Depende de:** Milestone M1 para os dados básicos; M3/M4 para a versão completa com dois sensores.

### Issue M5-01 — Módulo de detecção de F-formations
`etl` `socioenative` `p1-core`

Não existe implementação hoje. É a lacuna mais estrutural em relação à literatura citada na proposta (Kendon, Cristani et al.).

- [x] Implementar `f_formations.py` com algoritmo de votação de foco de atenção (ex.: variante de centro-de-transação) a partir de posição + orientação corporal
- [x] Testes sintéticos: grupo em semicírculo ao redor de um ponto deve ser detectado como F-formation; pessoas dispersas, não
- [x] Integrar a saída ao `plush_metrics.py` ou a um módulo de métricas de grupo separado

### Issue M5-02 — Classificar proxêmica no instante da passagem
`socioenative` `p2-desejavel`

`holder_inference._emit_handoff_event` já calcula `dist_interpersonal_m`; falta classificar em zona.

- [x] Reaproveitar `ProxemicsAnalyzer.get_zone()` no momento do handoff
- [x] Gravar a zona proxêmica junto ao evento de handoff em `events.csv`

### Issue M5-03 — Camada de ETL (consolidação em lote)
`etl` `p1-core`
**Depende de:** M1-01, M1-04

- [x] Criar `tools/etl.py`: lê `trajectories.csv` + `plush_state.csv` + `events.csv` de uma ou mais sessões
- [x] Produzir tabelas agregadas: eventos de posse, eventos de atenção sustentada, períodos de presença
- [x] Suportar múltiplas sessões de uma vez (para comparação entre sessões/cenas)

### Issue M5-04 — Dashboards/análises respondendo às perguntas da proposta
`etl` `p2-desejavel`
**Depende de:** M5-01, M5-03

- [x] Concentração vs. circulação do artefato (Gini já calculado — visualizar por cena)
- [x] Engajamento por cena (comparar `avg_audience_attention_*` entre cenas)
- [x] Presença do portador vs. atenção dos demais (correlação)
- [x] Frequência de F-formations vs. interação individual

### Issue M5-05 (desejável) — Fusão em tempo real
`rede` `p2-desejavel`
**Depende de:** M3-08

Ver Issue M3-08 — só entra aqui se sobrar tempo, conforme a proposta marca este item como "desejável".

---

## Transversais

### Issue T-01 — Separar `requirements-dev.txt`
`docs` `p2-desejavel`

`ultralytics`, `pytest`, `motmetrics`, `pandas` (uso de treino/avaliação) estão hoje junto com as dependências de execução.

- [x] Criar `requirements-dev.txt` com dependências de treino, avaliação e teste
- [x] Manter `requirements.txt` só com o necessário para rodar o sistema em produção

### Issue T-02 — Atualizar README com licença AGPL e escopo atual
`docs` `p2-desejavel`

- [x] Documentar a licença AGPL-3.0 do Ultralytics e implicações para modelos derivados
- [x] Reposicionar o README para o escopo da proposta (hoje ainda descreve "captura de movimento multi-alvo" genérica)
- [x] Remover menções fixas a hardware específico (“RX 6600”) do texto de interface/relatórios onde não fizer sentido versionar

### Issue T-03 — Remover código morto do escopo anterior
`p2-desejavel`

- [x] Avaliar remoção/arquivamento de `toy_interaction.py`, `hand_roi_detector.py`, `depth_clustering_3d.py` (substituídos por `holder_inference.py`) se não estiverem mais em uso ativo
- [x] Se algum seguir em uso experimental, mover para uma pasta `experimental/` e documentar o motivo

---

## Resumo de dependências entre marcos

```
M0 (bloqueador de tudo)
 └─> M1 (base funcional, 1 sensor)
      ├─> M2 (detector + inferência automática)      ── pode iniciar coleta (M2-01) em paralelo a M0/M1
      ├─> M3 (dois sensores)                          ── M3-03 depende de M0-02
      │    └─> M4 (validação completa 1 vs 2 sensores)
      └─> M4 parcial (métricas de acurácia com 1 sensor) ── pode começar antes de M3
      └─> M5 (ETL/BI, F-formations)                   ── M5-01/M5-03 podem começar com dados de M1
```
