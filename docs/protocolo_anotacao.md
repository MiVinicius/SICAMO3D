# Protocolo de Anotação Manual Independente (Milestone M4)

Este documento estabelece a metodologia de anotação manual, critérios de marcação de eventos e conformidade ética com o Comitê de Ética em Pesquisa (CEP/CONEP) para validação do sistema SICAMO3D.

---

## 1. Diretrizes Éticas e Proteção de Dados (CEP/CONEP)

Em conformidade com a Resolução CNS 510/2016 e as diretrizes do projeto:
1. **Gravações em Vídeo:** Quando houver gravação visual de participantes no espaço teatral, os dados de vídeo RGB bruto devem ser mantidos estritamente sob controle do pesquisador responsável em ambiente com criptografia.
2. **Replay Anonimizado:** Para anotações por terceiros ou assistentes de pesquisa, deve-se priorizar o uso do utilitário `tools/replay.py` operando sobre as projeções esqueléticas e centroides métricos 3D (`trajectories.csv`), sem exibição de faces ou dados biométricos diretos.
3. **Observação ao Vivo:** Nos ensaios abertos, dois observadores humanos independentes anotam os eventos cênicos de posse e trânsito usando planilhas sincronizadas por relógio digital comum.

---

## 2. Critérios Operacionais de Anotação

### 2.1 Intervalos de Posse do Artefato (Pelúcia)
- **Início da Posse:** Instante exato em que o participante fecha as mãos sobre o objeto ou o apoia intencionalmente contra o corpo (abraço/sustentação). Contatos acidentais inferiores a $0.3\text{ s}$ não contam como posse.
- **Fim da Posse:** Instante em que o participante solta completamente o objeto (seja depositando na mesa/chão ou liberando para outro participante).
- **Abraço vs Pegada:** Se o participante envolver o objeto junto ao tórax, anota-se o estado `COM_PORTADOR` com flag qualitativa `abraco`.

### 2.2 Eventos de Passagem de Posse (Handoffs)
- **Início da Passagem:** Quando o receptor toca o objeto que ainda está sob controle do doador.
- **Conclusão da Passagem:** Quando o doador solta completamente o objeto e o receptor assume o controle exclusivo.
- **Timestamp de Referência:** O instante em que a posse se consolida nas mãos do receptor ($t_{\text{handoff}}$).
- **Atributos Mandatórios:** `t_capture_ms`, `donor_id` (identidade do doador), `receiver_id` (identidade do receptor).

### 2.3 Rastreamento de Identidades Reais (ID Switches)
- Para avaliar a estabilidade do rastreador, cada pessoa no espaço cênico recebe um rótulo imutável (`P1`, `P2`, `P3`, etc.).
- Sempre que o sistema atribuir um novo `track_id` à mesma pessoa real (ex.: o participante `P1` foi identificado como `track 1`, sofreu oclusão e reapareceu como `track 4`), computa-se **1 ID Switch**.

---

## 3. Esquemas de Arquivos de Gabarito (Ground Truth)

### 3.1 Eventos de Gabarito (`gt_events.csv`)
Utilizado para validar detecção de handoffs pelo `tools/evaluate.py`:
```csv
t_capture_ms,event_type,scene_id,donor_id,receiver_id
12450,handoff,Cena 1,1,2
28700,handoff,Cena 1,2,3
```

### 3.2 Identidades Reais Frame a Frame (`gt_identities.csv`)
Utilizado para cálculo da taxa de ID switches por minuto:
```csv
t_capture_ms,frame_idx,real_person_id,track_id
1000,0,P1,1
1000,0,P2,2
1033,1,P1,1
1033,1,P2,2
...
```
Se em determinado instante `P1` passar a ser associado a `track_id = 5`, o script `tools/evaluate.py` detectará a transição $1 \to 5$ e incrementará o contador de ID switches.
