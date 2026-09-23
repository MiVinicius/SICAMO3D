# Checklist de Testes de Risco Físicos — Milestone M0 (Issue M0-09)

Este documento estabelece o protocolo experimental de validação física com o sensor **Microsoft Kinect v2** real antes de avançar para a gravação com múltiplos participantes do Milestone M1.

---

## 1. Contexto e Objetivos

O ambiente de espetáculo teatral e estandes interativos apresenta desafios físicos específicos para sensoriamento por tempo de voo (ToF) e visão computacional:
1. **Iluminação Cênica Variável:** Refletores de alta potência, contraluz ou baixa luminosidade que afetam a exposição e o FPS do fluxo RGB;
2. **Interferência Infravermelha (IR):** Lâmpadas halógenas e canhões incandescentes emitem radiação no espectro infravermelho próximo (~850nm), potencialmente saturando ou introduzindo ruído no sensor de profundidade ToF do Kinect;
3. **Detecção e Alcance da Pelúcia:** Linha de base de precisão/recall do detector em distâncias de 2m, 3m e 4m em três posturas típicas (repouso na mesa, segurada nas mãos, abraçada contra o tronco).

---

## 2. Teste Físico 1: FPS e Exposição RGB sob Iluminação de Palco

**Procedimento:**
1. Posicionar o Kinect v2 montado a ~1.40m de altura com inclinação calibrada;
2. Ajustar a iluminação do local para o cenário de apresentação teatral (refletores frontais e penumbra circundante);
3. Executar o script de captura ou a aplicação principal:
   ```powershell
   .\.venv\Scripts\python main.py
   ```
4. Observar o FPS medido pelo EMA do dashboard e a taxa de quadros entregue pelo driver libfreenect2 / SDK Kinect v2.

| Cenário de Iluminação | Nível de Lux Estimado | FPS Medido (RGB) | Estabilidade Visual | Observações |
| :--- | :--- | :--- | :--- | :--- |
| Luz Ambiente Normal | ~300 - 500 lux | 30.0 fps | Estável, sem ruído | Baseline de laboratório |
| Iluminação Cênica Frontal (Foco) | > 800 lux | [A preencher] | [A preencher] | Testar se há estouro de branco na pelúcia |
| Penumbra / Baixa Luminosidade | < 50 lux | [A preencher] | [A preencher] | Avaliar se tempo de exposição do sensor derruba FPS |
| Contraluz Intenso | Foco contra a lente | [A preencher] | [A preencher] | Verificar silhueta e detecção de pose |

**Critério de Aceitação:** FPS mínimo estável de $\ge 25\text{ fps}$ (idealmente cravado em $30\text{ fps}$) sem travamento no loop de captura.

---

## 3. Teste Físico 2: Qualidade do Mapa de Profundidade sob Fontes IR (Refletores Halógenos)

**Procedimento:**
1. Apontar o Kinect para a área de atuação ($0.5\text{m} \le Z \le 4.5\text{m}$);
2. Ligar fontes de calor/iluminação incandescente ou halógena nas proximidades do feixe do Kinect;
3. Executar `tests/test_kinect_capture.py` e inspecionar a imagem gerada `sample_kinect_depth.jpg`;
4. Avaliar ocorrência de buracos negros (*depth drops*) ou ruído de sal-e-pimenta nas bordas da silhueta dos participantes.

| Tipo de Fonte de Luz | Distância da Fonte ao Feixe | % de Perda de Profundidade | Ruído Observado | Conclusão / Risco |
| :--- | :--- | :--- | :--- | :--- |
| LED Frio / Fluorescente | Qualquer | < 1% | Inexistente | Sem interferência no infravermelho |
| Refletor Halógeno | 2 a 3 metros | [A preencher] | [A preencher] | [A preencher] |
| Luz Solar Direta (Janela) | Incidência direta | [A preencher] | [A preencher] | Risco conhecido de saturação ToF |

**Critério de Aceitação:** Área útil de profundidade ($Z \le 4.0\text{m}$) mantendo $< 5\%$ de pixels nulos sobre o corpo das pessoas e do chão do estande.

---

## 4. Teste Físico 3: Detecção da Pelúcia por Distância e Postura (Linha de Base Pré-M2)

**Procedimento:**
1. Com o modelo atual carregado (ou em Modo A), posicionar a pelúcia nos seguintes pontos fixos:
   - $Z = 2.0\text{m}$
   - $Z = 3.0\text{m}$
   - $Z = 4.0\text{m}$
2. Em cada distância, testar as três situações de interação:
   - **Mesa:** Artefato em repouso sobre superfície plana sem contato humano;
   - **Na mão:** Artefato sustentado pelo participante à frente do corpo;
   - **Abraçada:** Artefato junto ao tórax com oclusão parcial pelos braços.

| Distância | Situação | Detecção 2D (BBox) | Confiança Estimada | Ponto 3D Válido ($Z_{sala}$) | Rastreamento Estável |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **2.0 m** | Sobre a mesa | [A preencher] | [A preencher] | [A preencher] | [A preencher] |
| **2.0 m** | Segurada na mão | [A preencher] | [A preencher] | [A preencher] | [A preencher] |
| **2.0 m** | Abraçada ao tronco | [A preencher] | [A preencher] | [A preencher] | [A preencher] |
| **3.0 m** | Sobre a mesa | [A preencher] | [A preencher] | [A preencher] | [A preencher] |
| **3.0 m** | Segurada na mão | [A preencher] | [A preencher] | [A preencher] | [A preencher] |
| **3.0 m** | Abraçada ao tronco | [A preencher] | [A preencher] | [A preencher] | [A preencher] |
| **4.0 m** | Sobre a mesa | [A preencher] | [A preencher] | [A preencher] | [A preencher] |
| **4.0 m** | Segurada na mão | [A preencher] | [A preencher] | [A preencher] | [A preencher] |
| **4.0 m** | Abraçada ao tronco | [A preencher] | [A preencher] | [A preencher] | [A preencher] |

---

## 5. Validação Física do Referencial do Chão (Issue M0-02)

**Procedimento:**
1. Medir com fita métrica a altura real do centro óptico do Kinect ao piso: $h_{real} = \_\_\_\_\text{ m}$;
2. Posicionar um participante de altura conhecida em pé a $3.0\text{m}$ do sensor;
3. Executar o pipeline e inspecionar os valores registrados em `trajectories.csv`:
   - $Y$ dos pés: deve ser $\approx 0.00\text{ m} \pm 0.05\text{ m}$;
   - $Y$ do topo da cabeça: deve coincidir com a altura real do participante $\pm 0.05\text{ m}$;
   - Participante caminhando para a sua direita deve produzir $X_{sala} > 0$.
