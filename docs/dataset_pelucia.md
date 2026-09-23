# Protocolo de Coleta e Rotulagem do Dataset da Pelúcia (Issue M2-01)

Este documento descreve o protocolo formal de coleta fotográfica e anotação do artefato cênico (**Pelúcia Móvel do Espetáculo**) para treinamento do detector de classe única do SICAMO3D.

---

## 1. Diretrizes Éticas e de Privacidade (Conformidade CEP)

Em rigorosa consonância com a seção 9 e 14 do projeto de pesquisa e as normas do Comitê de Ética em Pesquisa (CEP):
- **Isolamento de Pessoas:** Nenhuma criança, ator ou participante deve ter rosto ou identificadores biométricos presentes nas imagens do dataset de treino.
- **Sustentação do Artefato:** A pelúcia deve ser fotografada em suportes inanimados (mesa, pedestal, cadeira) ou, caso sustentada manualmente durante a coleta de posturas de mão/abraço, o enquadramento deve ser restrito exclusivamente ao artefato e vestimenta neutra, cortando rostos e traços fenotípicos individuais.
- **Armazenamento Seguro:** As imagens brutas e rótulos são salvos em ambiente local e restrito aos pesquisadores.

---

## 2. Estratificação da Amostragem (Volume: ~400 a 600 fotos)

Para garantir robustez e generalização aos desafios físicos identificados em M0-09, a coleta fotográfica divide-se proporcionalmente nos seguintes estratos:

### 2.1 Distância ao Sensor
| Faixa de Distância | Proporção | Descrição |
| :--- | :--- | :--- |
| **Próxima (1.5 m a 2.5 m)** | 35% | Pelúcia em alta resolução espacial, capturando detalhes de textura e bordas |
| **Média (2.5 m a 3.5 m)** | 40% | Faixa típica de interação principal na área das crianças |
| **Distante (3.5 m a 4.5 m)** | 25% | Limite de alcance confiável da sala, artefato com menor resolução em pixels |

### 2.2 Posturas e Situações do Artefato
| Situação | Proporção | Detalhes da Postura |
| :--- | :--- | :--- |
| **Sobre a mesa / Chão** | 30% | Artefato isolado em repouso, em diferentes orientações (em pé, deitado, de costas) |
| **Sustentada nas mãos** | 40% | Artefato com oclusão parcial na base por luvas ou mãos neutras |
| **Abraçada contra o tronco** | 30% | Artefato com oclusão frontal severa (> 40%), simulando abraço infantil |

### 2.3 Condições de Iluminação Cênica
| Iluminação | Proporção | Desafios Específicos |
| :--- | :--- | :--- |
| **Luz Ambiente Normal** | 40% | Iluminação difusa de laboratório (~300 lux) |
| **Refletores Cênicos Frontais** | 30% | Alto contraste e sombras duras na área de apresentação |
| **Penumbra / Baixa Luminosidade** | 20% | Ruído térmico do sensor e desfoque de movimento |
| **Contraluz Intenso** | 10% | Silhueta escura e estouro de exposição ao fundo |

---

## 3. Protocolo de Rotulagem e Estrutura de Arquivos

### 3.1 Ferramentas Recomendadas
- **CVAT (Computer Vision Annotation Tool)** ou **Label Studio**;
- Caixa delimitadora (*Bounding Box* 2D) ajustada firmemente ao contorno visível do artefato;
- **Classe única:** `pelucia` (ID da classe: `0`).

### 3.2 Estrutura do Diretório YOLO
```text
dataset_plush/
├── dataset.yaml
├── images/
│   ├── train/       # 80% das imagens
│   └── val/         # 20% das imagens
└── labels/
    ├── train/       # Arquivos .txt (0 cx cy w h)
    └── val/
```

### 3.3 Arquivo de Configuração (`dataset.yaml`)
```yaml
path: ../dataset_plush
train: images/train
val: images/val

names:
  0: pelucia
```
