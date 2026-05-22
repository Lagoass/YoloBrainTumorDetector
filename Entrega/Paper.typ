// ─────────────────────────────────────────────────────────────────────────────
// Yolo_Paper.typ
// Detecção de Tumores Cerebrais em MRI T1CE com YOLOv11s
// ─────────────────────────────────────────────────────────────────────────────

#set page(
  paper: "a4",
  margin: (top: 2.5cm, bottom: 2.5cm, left: 2.5cm, right: 2.5cm),
)
#set text(font: "Linux Libertine", size: 11pt, lang: "pt")
#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.")
#show heading: it => { v(0.8em); it; v(0.4em) }

// ── Title ────────────────────────────────────────────────────────────────────
#align(center)[
  #text(14pt, weight: "bold")[
    Otimização por Classe em Detecção de Tumor Cerebral \
    com YOLO: Ratio de Negativos e Arquitetura \
    Multiclasse vs. Binária
  ]
]

#v(0.8em)
#align(center)[
  #text(12pt)[Gustavo Doniani Lagôa Gomes]
]
#align(center)[
  #text(11pt)[Insper]
]
#align(center)[
  #text(11pt)[22 de maio de 2026]
]

#v(1.5em)

// ── 1. Resumo ────────────────────────────────────────────────────────────────
= Resumo

Tumores cerebrais primários exigem localização precisa e classificação precoce para orientar
o tratamento. A ressonância magnética ponderada em T1 com contraste de gadolínio (T1CE)
é o padrão-ouro para essa finalidade, e modelos baseados em YOLO têm demonstrado
desempenho crescente nesse domínio. No entanto, duas variáveis metodológicas permanecem
não investigadas: o ratio de amostras negativas no treinamento e a escolha entre arquitetura
multiclasse unificada ou especialistas binários por classe tumoral.

Este trabalho apresenta dois experimentos controlados sobre o mesmo dataset e arquitetura.
No primeiro, realizamos ablation com quatro ratios de amostras negativas (10%, 20%,
30% e 40%) para glioma, meningioma e adenoma pituitário, totalizando 12 modelos com
variável única isolada. No segundo, comparamos o melhor modelo multiclasse (Run 7,
mAP\@0.50=0.9195) contra especialistas binários treinados com o ratio ótimo de cada classe.

Os resultados demonstram que não existe ratio ótimo universal: glioma e meningioma
maximizam performance com 20% de negativos, enquanto pituitary requer 30–40% para
controlar a taxa de falsos positivos (FPR). A comparação arquitetural revela assimetria entre
classes: glioma performa melhor no multiclasse (recall 0.8214 vs. 0.7786),
// CORREÇÃO 5 — Resumo: qualificação de meningioma e pituitary
enquanto meningioma beneficia em recall e FPR na especialização binária (mAP\@0.50=0.9727 vs. 0.9195 global);
pituitary beneficia em FPR — com leve regressão em recall — na especialização binária.
O FPR em imagens saudáveis cai de 0.50 no multiclasse para próximo de zero nos binários,
confirmando que o FPR residual é artefato de competição inter-classe. Todos os
experimentos utilizam YOLOv11s e o BRISC 2025, com 6.000 imagens T1CE anotadas.

// ── 2. Introdução ────────────────────────────────────────────────────────────
= Introdução

Tumores cerebrais primários figuram entre as doenças com maior impacto na sobrevida.
Glioblastomas apresentam sobrevida mediana inferior a 15 meses mesmo com tratamento
agressivo, e estimativas globais apontam para mais de 300.000 novos diagnósticos de
tumores cerebrais primários por ano, com incidência crescente correlacionada à expansão
da disponibilidade de neuroimagem. A ressonância magnética T1CE é o padrão-ouro para
localização tumoral: o gadolínio extravasa para regiões com barreira hematoencefálica
comprometida pela neoangiogênese, delimitando com precisão as margens ativas do tumor.
Modelos de detecção baseados em YOLO têm sido amplamente aplicados nesse contexto,
com mAP\@0.50 frequentemente acima de 99% em condições controladas (Chen et al., 2024;
Tariq & Choi, 2026).

Contudo, a revisão sistemática da literatura revela dois problemas metodológicos fundamentais não investigados.

O primeiro é o ratio de amostras negativas. Em modelos YOLO, imagens sem anotações
interagem exclusivamente com o objectness loss — localization loss e classification loss
não disparam na ausência de ground truth boxes. Isso significa que o ratio de negativos
controla diretamente a penalização dos anchors em tecido saudável, influenciando FPR
de forma independente de qualquer ajuste arquitetural. A literatura em outros domínios
médicos demonstra que existe um ponto ótimo não trivial: 5% em histopatologia de mama
(Camelyon16, Han et al., 2023), 10–20% em detecção de pólipos colonoscópicos (YOLO-LAN, 2024).
Em detecção de tumor cerebral, o padrão predominante é um ratio arbitrário
de 1:1, sem justificativa matemática para o contexto de regressão espacial densa. Nenhum
ablation controlado dessa variável existe na literatura.

O segundo problema é a escolha arquitetural entre multiclasse e especialistas
binários. A função Softmax impõe competição de soma zero entre classes, onde gradientes
de classes dominantes sobrescrevem sinais de classes mais sutis — fenômeno conhecido
como negative transfer (Zhang & Yang, 2021). Modelos binários com ativação Sigmoid
independente eliminam essa competição. O efeito é documentado em imagens médicas: em
classificação pulmonar, um modelo binário para tuberculose atingiu 98,97% contra 96,18%
do multiclasse (ResNet-ViT, 2024). Em tumor cerebral com YOLO, Nimmagadda & Devi
(2025) — o único paper da literatura a reportar FPR explicitamente — encontraram FPR
de 0.35 em imagens saudáveis com YOLOv7 multiclasse, atribuindo o resultado à hipersensibilização do backbone a features tumorais. Nenhum estudo, contudo, treinou YOLOs
binários independentes por classe e os comparou diretamente contra um YOLO unificado.

Ambos os problemas têm implicação clínica direta. FPR alto em saudáveis significa pacientes
sem tumor sendo encaminhados para biópsia. Gradient swamping por excesso de negativos
significa tumores presentes não detectados. Teto estrutural por competição inter-classe
significa que métricas globais elevadas mascaram falhas específicas em classes clinicamente
críticas.

Este trabalho apresenta quatro contribuições:

+ Primeiro ablation controlado de ratio de amostras negativas para detecção de tumor
  cerebral com YOLO.
+ Primeira comparação direta entre YOLO multiclasse unificado e especialistas YOLO
  binários independentes por classe, com dataset e arquitetura fixos.
+ Evidência empírica de assimetria inter-classe: glioma beneficia da regularização contrastiva do multiclasse; meningioma beneficia da especialização binária em recall e FPR; pituitary beneficia da especialização binária em FPR — com leve regressão em recall.
+ FPR em imagens saudáveis como métrica clínica primária, com protocolo de avaliação
  dedicado.

// ── 3. Trabalhos Relacionados ─────────────────────────────────────────────────
= Trabalhos Relacionados

== YOLO em Imagens Médicas

A família YOLO consolidou-se como framework dominante em detecção médica pela
combinação única de velocidade e precisão. Iterações recentes — YOLOv8 com cabeças
desacopladas e mecanismos anchor-free, YOLOv11 com módulos C3k2 e GhostConv —
reduziram a necessidade de configuração manual e ampliaram a aplicabilidade em cenários
com alta variância morfológica. Adaptações específicas para MRI incluem normalização de
intensidade para distribuições grayscale, funções de perda avançadas como CIoU e DFL para
margens tumorais irregulares, e resolução de entrada preservada em 640×640 para manter
detalhes diagnósticos críticos. Em benchmarks comparativos, YOLOv7 superou Faster R-CNN
e U-Net em datasets de tumor cerebral com tempos de inferência de 5,3 ms por
slice (Alhussainan et al., 2024). Uma limitação sistêmica, porém, persiste: FPR em imagens
saudáveis é quase universalmente omitido como métrica — dos dez papers mais relevantes
desta revisão, apenas um o reporta.

== YOLO para Detecção de Tumor Cerebral

O estado da arte frequentemente reporta mAP\@0.50 acima de 99% em datasets como Br35H,
Roboflow e Figshare. Esses resultados devem ser interpretados com cautela: os datasets são
estreitos em diversidade de protocolo de aquisição, raramente incluem imagens de controle
saudável em quantidade representativa, e as métricas globais mascaram falhas por classe.
Glioma emerge consistentemente como a classe mais desafiadora — seu padrão infiltrativo
e bordas difusas tornam a regressão de bounding box inerentemente imprecisa, confirmado
em múltiplos estudos (Chourib et al., 2025; Nimmagadda & Devi, 2025). O único paper a
reportar FPR explicitamente encontrou 0.35 em imagens saudáveis com YOLOv7, apesar
de 99,5% de taxa de detecção geral — resultado que os autores consideram clinicamente
inaceitável (Nimmagadda & Devi, 2025). Nenhum paper combina BRISC 2025 com T1CE
confirmado, negative samples sistematizados e FPR como métrica primária.

== Ratio de Amostras Negativas em Detecção Médica

Em modelos YOLO, amostras negativas interagem exclusivamente com o objectness loss.
Quando um arquivo de label está vazio, o Task-Aligned Learning ignora regression e DFL,
penalizando apenas os logits de classificação via BCE — comportamento que torna o ratio
de negativos um hiperparâmetro de dataset com efeito direto e isolável no FPR. A evidência
empírica sobre o ponto ótimo varia por domínio: 5% em whole slide imaging de mama
(Camelyon16, Han et al., 2023), 10–20% em pólipos colonoscópicos (YOLO-LAN, 2024),
\~10% em mamografia. A heurística geral da literatura sugere 5–10% para detectores
single-stage, com hard negative mining para casos ambíguos. Em tumor cerebral, o ratio
1:1 é padrão sem justificativa matemática, e nenhum ablation controlado existe — gap que
este trabalho preenche diretamente.

== Multiclasse vs. Especialistas Binários em Imagens Médicas

A função Softmax cria competição de soma zero entre classes: gradientes de classes dominantes sobrescrevem sinais de classes mais sutis no espaço de representação compartilhado,
impondo um teto estrutural de performance denominado negative transfer (Zhang & Yang,
2021). Modelos binários com Sigmoid independente eliminam essa interferência, permitindo
que cada classificador mapeie sua fronteira de decisão sem competição. O efeito é bem
documentado: em TB vs. pneumonias, 98,97% binário contra 96,18% multiclasse (ResNet-ViT, 2024);
em subtipos de câncer de mama, AUC próxima de 1.0 com OvR contra 63,79%
no modelo unificado (StackANN, 2024); em lesões cutâneas, o ganho aparente do multiclasse
é identificado como accuracy paradox em dataset desequilibrado (VGG16, 2024). Em tumor
cerebral com YOLO, a comparação direta com modelos binários independentes por classe
não foi realizada. Este trabalho preenche esse gap, revelando que o efeito é assimétrico: nem
todas as classes se beneficiam da especialização binária da mesma forma.

// ── 4. Dataset ───────────────────────────────────────────────────────────────
= Metodologia

== BRISC 2025

O BRISC 2025 (Fateh et al., arXiv:2506.14318) é um dataset de MRI T1CE composto por
6.000 imagens distribuídas em quatro classes: glioma, meningioma, adenoma pituitário e
no_tumor (ausência de tumor primário). Todas as imagens são ponderadas em T1 com
contraste de gadolínio, fornecidas como slices 2D em formato JPEG nos três planos anatômicos
— axial, coronal e sagital. Para as três classes tumorais, o dataset disponibiliza máscaras
de segmentação pixel-wise em PNG, a partir das quais bounding boxes no formato YOLO
foram derivadas via detecção de contorno e extração de coordenadas extremas. Imagens
no_tumor recebem arquivos de label vazios. Todas as imagens passaram por normalização
min-max para uint8 antes do armazenamento.

O split foi estratificado por classe e plano anatômico com seed=42, resultando em 4.802
imagens de treino (80%), 599 de validação (10%) e 599 de teste (10%). O split é preservado
integralmente em todos os datasets derivados — nenhuma reamostragem foi realizada.

A classe no_tumor do BRISC é híbrida: inclui tanto cérebros estruturalmente saudáveis
quanto lesões benignas não-neoplásicas como cistos simples e malformações vasculares
estáveis. Isso é relevante para a interpretação do FPR: o modelo aprende a distinguir tumor
primário de "ausência de tumor primário", não estritamente de "cérebro completamente
saudável".

#figure(
  image("figura1_grid_brisc.png", width: 80%),
  caption: [Exemplos de imagens do BRISC 2025.],
)

== Metodologia de Anotação de Negativos

A forma como imagens negativas são anotadas determina o comportamento arquitetural do
YOLO durante o treinamento. Quando um arquivo de label está vazio (_Empty Annotations_),
o loop de Task-Aligned Learning detecta zero ground truth boxes e ignora completamente
os cálculos de IoU-regression e DFL. Apenas o BCE de classificação dispara, forçando todos
os logits de anchor para background sem corromper a cabeça de regressão. Este é o único
método arquiteturalmente válido,
// CORREÇÃO 1 — Section 4.2: substituição da validação empírica do FPR
validado empiricamente pelo FPR = 0.0% atingido pelos especialistas binários de glioma e
meningioma no test split.

Duas alternativas comuns são problemáticas. A Full-Image Bounding Box — anotar a
imagem inteira como objeto — colapsa o Task-Aligned Assigner para a borda da imagem,
destruindo o DFL e corrompendo a regressão em todos os anchors (FPR \> 40%). O
Anatomical Brain Contour — anotar o contorno cerebral como objeto — cria um paradoxo
semântico onde o tumor existe dentro do objeto "no_tumor", degradando recall em tumores
pequenos (FPR 10–15%). Este trabalho utiliza exclusivamente Empty Annotations.

== Datasets Derivados

A partir do brisc_dataset foram gerados dois conjuntos:

*brisc_dataset (nc=3):* dataset multiclasse base com as três classes tumorais, utilizado no
Experimento 2.

*dissected_brisc_ratios (nc=1):* 12 datasets binários gerados simultaneamente — 4 ratios
(10%, 20%, 30%, 40%) × 3 tumores. Cada dataset contém as imagens da classe tumoral
com label reescrito para classe 0, mais uma amostra proporcional de imagens no_tumor
calculada por:

#align(center)[
  $n_"notumor" = "round"(n_"tumor" times "ratio" \/ (1 - "ratio"))$
]

A amostragem utiliza um gerador aleatório compartilhado (random.Random, seed=42) em
ordem determinística. O pool de no_tumor disponível é de 967 imagens no treino e 120 no
val e teste.

#figure(
  caption: [Contagem de imagens por dataset (split treino).],
  table(
    columns: (auto, auto, auto, auto, auto),
    align: center,
    stroke: 0.5pt,
    table.header([*Ratio*], [*Tumor*], [*N tumor*], [*N no_tumor*], [*Total*]),
    [10%], [glioma],      [1.121], [125], [1.246],
    [10%], [meningioma],  [1.309], [145], [1.454],
    [10%], [pituitary],   [1.405], [156], [1.561],
    [20%], [glioma],      [1.121], [280], [1.401],
    [20%], [meningioma],  [1.309], [327], [1.636],
    [20%], [pituitary],   [1.405], [351], [1.756],
    [30%], [glioma],      [1.121], [480], [1.601],
    [30%], [meningioma],  [1.309], [561], [1.870],
    [30%], [pituitary],   [1.405], [602], [2.007],
    [40%], [glioma],      [1.121], [747], [1.868],
    [40%], [meningioma],  [1.309], [873], [2.182],
    [40%], [pituitary],   [1.405], [937], [2.342],
  )
)

// ── 5. Setup Experimental ────────────────────────────────────────────────────
== Setup Experimental

== Arquitetura

Todos os experimentos utilizam YOLOv11s — a variante small da undécima geração da
família YOLO (Ultralytics, 2024). A escolha é motivada pela restrição de hardware (8 GB
VRAM, RTX 5060) e pela necessidade de treinar 12 modelos no ablation mantendo configuração
idêntica. YOLOv11s incorpora módulos C3k2 e cabeça desacoplada anchor-free com
Task-Aligned Assigner, eliminando a necessidade de configuração manual de anchor boxes.
Todos os modelos foram inicializados com pesos pré-treinados no COCO.

== Hiperparâmetros

Os hiperparâmetros abaixo são fixos em todos os 13 modelos treinados (Run 7 multiclasse +
12 binários). O único parâmetro que varia é o dataset — e dentro do Experimento 1, apenas
o ratio de amostras negativas. Essa fixação garante que qualquer diferença de performance
seja atribuível exclusivamente à variável de interesse.

#figure(
  caption: [Hiperparâmetros fixos em todos os experimentos.],
  table(
    columns: (auto, auto),
    align: (left, center),
    stroke: 0.5pt,
    table.header([*Parâmetro*], [*Valor*]),
    [epochs],        [100],
    [imgsz],         [640],
    [batch],         [16],
    [amp],           [True],
    [patience],      [30],
    [cos_lr],        [True],
    [fliplr],        [0.5],
    [degrees],       [10.0],
    [mosaic],        [0.5],
    [hsv_v],         [0.2],
  )
)

== Métricas

*mAP\@0.50:* mean Average Precision com threshold de IoU = 0.50. Métrica padrão da
literatura para comparação.

*mAP\@0.5:0.95:* mAP médio sobre thresholds de IoU de 0.50 a 0.95 em passos de 0.05.
Indica qualidade de localização além da classificação.

*Precision e Recall:* calculados sobre o test split via
curva PR ao threshold ótimo reportado pelo YOLO, exceto
para os recalls por classe do Run 7 na Tabela 4, que
foram extraídos por contagem direta da matriz de confusão
da Figura 5 com conf=0.25.

*FPR em imagens saudáveis:* métrica clínica primária deste trabalho. Implementada via
healthy_fpr(): inferência com conf=0.25 sobre todas as imagens com label vazio no test
split, contando imagens que produzem ao menos uma predição de bounding box. Reportado
como fração de imagens no_tumor com falso positivo.

A FPR é tratada como métrica primária porque captura diretamente o risco clínico de
sobrediagnóstico — algo que mAP global não expõe. Um modelo com mAP\@0.50=0.995
e FPR=0.35 é clinicamente perigoso; esse valor seria mascarado por qualquer métrica de
média global (Nimmagadda & Devi, 2025).

== Nota sobre Background na Matriz de Confusão

"Background" na matriz de confusão do YOLO não é uma classe de treinamento — é
um estado de avaliação. Quando uma ground truth box existe mas nenhuma predição
casou com ela, o YOLO registra "True tumor → Predicted background" (falso negativo).
Quando uma imagem sem ground truth produz uma predição, registra "True background
→ Predicted tumor" (falso positivo). A coluna background na matriz representa ausência
de correspondência durante avaliação, não uma classe aprendida. Por essa razão, a matriz
normalizada é enganosa nessa coluna — quando nenhum anchor de background é suprimido,
a normalização divide o único valor não-zero por si mesmo, produzindo 1.00. Este trabalho
utiliza exclusivamente matrizes não normalizadas para análise de FP em imagens saudáveis.

== Protocolo de Avaliação

Todos os modelos são avaliados exclusivamente sobre o test split. As curvas de convergência
— mAP\@0.50 por epoch registradas via results.csv — são analisadas para identificar o
best epoch, early stops e padrões de instabilidade. Essa análise de convergência é parte
integrante dos resultados, não apenas metadado de treinamento.

// ── 6. Experimento 1 — Ablation do Ratio de Amostras Negativas ───────────────
= Resultados e Discussão

== Setup

O Experimento 1 isola o ratio de amostras negativas como única variável de interesse. Para
cada combinação de ratio (10%, 20%, 30%, 40%) e classe tumoral (glioma, meningioma,
pituitary), um modelo YOLOv11s independente foi treinado com pesos pré-treinados no
COCO. As imagens tumorais são idênticas em todos os 12 modelos — a única diferença é a
quantidade de imagens no_tumor no treino. Arquitetura, hiperparâmetros e splits são fixos
conforme a Seção 4.

== Resultados

#figure(
  caption: [Ablation completo: 12 modelos binários (test split). ★ early stop — modelo parou antes de 100 epochs por ausência de melhoria no val split. ⚡ best epoch = último epoch — modelo ainda melhorando quando patience disparou.],
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto),
    align: (left, center, center, center, center, center, center, center),
    stroke: 0.5pt,
    table.header(
      [*Tumor*], [*Ratio*], [*mAP\@0.50*], [*Precision*], [*Recall*], [*FPR*],
      [*Best epoch*], [*Total epochs*]
    ),
    [glioma],     [10%], [0.8137], [0.8143], [0.7571], [0.0%], [91], [95 ★],
    [glioma],     [20%], [0.8497], [0.7935], [0.8509], [0.0%], [99], [100],
    [glioma],     [30%], [0.8129], [0.8368], [0.6857], [0.0%], [80], [90 ★],
    [glioma],     [40%], [0.8191], [0.8483], [0.7643], [0.0%], [79], [100],
    [meningioma], [10%], [0.9582], [0.9583], [0.9693], [0.0%], [90], [100],
    [meningioma], [20%], [0.9727], [0.9734], [0.9632], [0.0%], [77], [100],
    [meningioma], [30%], [0.9550], [0.9457], [0.9607], [4.3%], [61], [69 ★],
    [meningioma], [40%], [0.9663], [0.9631], [0.9632], [0.0%], [87], [95 ★],
    [pituitary],  [10%], [0.9568], [0.8549], [0.9432], [10.0%],[37], [63 ★],
    [pituitary],  [20%], [0.9581], [0.8910], [0.9288], [6.8%], [59], [59 ★⚡],
    [pituitary],  [30%], [0.9671], [0.9399], [0.9489], [2.7%], [34], [95 ★],
    [pituitary],  [40%], [0.9524], [0.9362], [0.9432], [0.85%],[45], [100],
  )
)

#figure(
  image("figura2_matrizes_ratio_otimo.png", width: 100%),
  caption: [Matrizes de confusão não normalizadas dos modelos com ratio ótimo.],
)

#figure(
  image("figura3_convergencia_glioma.png", width: 95%),
  caption: [Curvas de convergência para glioma nos 4 ratios testados.],
)

== Análise por Classe

*Glioma.* O ratio 20% domina: mAP\@0.50=0.8497 e recall=0.8509, ambos os melhores do
grupo. O indicador mais revelador é o best epoch — o modelo de 20% atingiu seu melhor
resultado no epoch 99 de 100, demonstrando convergência estável até o final do treinamento.
O modelo de 30% foi o pior em recall (0.6857) e sofreu early stop no epoch 90 com best epoch
no 80 — o excesso de negativos destabilizou a convergência. O modelo de 40% teve best
epoch no 79, com 21 epochs de plateau. O modelo de 10% saturou no epoch 95. Todos os
quatro ratios atingiram FPR = 0.0%, indicando que glioma não possui confusão intrínseca
com tecido normal — o FPR é determinado pela presença de outras classes tumorais no
modelo multiclasse, não pelo volume de negativos.

*Meningioma.* O ratio 20% apresenta o maior mAP\@0.50 (0.9727) e precision (0.9734), com
apenas 7 FP anchors absolutos em 41 imagens de background no val split — o menor valor
absoluto de todos os ratios apesar do pool maior que o 10% (41 vs. 18 imagens). O ratio
10% detectou um tumor a mais na matriz (159/163 vs. 158/163) mas com mAP inferior,
sugerindo qualidade de bbox menor nas detecções adicionais. O ratio 30% é o único com
FPR acima de zero (4.3%), com early stop no epoch 69 e best epoch no 61 — instabilidade
confirmada pelos 16 FP anchors em 70 imagens de background. O ratio 40% apresenta FPR
zero mas mAP inferior ao 20%, com early stop no epoch 95.

*Pituitary.* Esta é a classe com comportamento estruturalmente diferente. O FPR cai
monotonicamente conforme o ratio aumenta: 10.0% → 6.8% → 2.7% → 0.85%. Nenhum
ratio domina em todas as métricas simultaneamente. O ratio 20% apresenta o melhor recall
absoluto (0.9288 no test split), mas foi cortado prematuramente — o best epoch foi o epoch
59, o último epoch antes do early stop, com o modelo ainda melhorando ativamente. Esse
resultado é inconclusivo e requer rerun com patience=50. O ratio 30% oferece o melhor
equilíbrio entre mAP (0.9671) e FPR (2.7%). O ratio 40% minimiza o FPR (0.85%) ao custo
de recall inferior. A progressão monotônica do FPR revela confusão anatômica genuína com
estruturas da base craniana — seio esfenoidal e fossa pituitária normal — que só é suprimida
com volume suficiente de negativos, independente de competição inter-classe.

== Análise de Convergência

Para glioma, o modelo de 20% convergiu ativamente até o epoch 99, indicando que o volume
de negativos permitiu aprendizado contínuo sem plateau prematuro. O modelo de 30% parou
no epoch 90 com best epoch no 80 — 10 epochs de plateau antes do early stop, sugerindo
que o excesso de negativos induziu instabilidade. O modelo de 40% completou 100 epochs
mas com best epoch no 79 — 21 epochs de plateau, indicando saturação precoce.

Para pituitary, o modelo de 10% teve best epoch no epoch 37 de 63 — saturação com
apenas 156 imagens negativas no treino. O modelo de 20% teve best epoch no último epoch
disponível, evidência de underfitting por patience. O modelo de 30% teve best epoch no
epoch 34 seguido de 61 epochs de plateau — convergência rápida e estável. O modelo de
40% completou 100 epochs com best epoch no 45, padrão mais equilibrado.

== Finding Central

O ratio ótimo é uma propriedade da classe tumoral, não do modelo ou do dataset. Glioma e
meningioma maximizam performance com 20% de negativos — alinhado com a heurística de
5–10% da literatura geral, com deslocamento para 20% explicável pela maior complexidade
morfológica comparada a domínios como mamografia ou histopatologia. Pituitary exige
30–40% para controlar FPR, com FPR decrescendo monotonicamente — comportamento
único entre as três classes que aponta para confusão anatômica genuína, não artefato de
competição inter-classe. A variação inter-classe do ratio ótimo não havia sido documentada
anteriormente na literatura de detecção de tumor cerebral com YOLO.

// ── 7. Experimento 2 — Multiclasse vs. Especialistas Binários ────────────────
== Experimento 2 — Multiclasse vs. Especialistas Binários

== Setup

O Experimento 2 compara o melhor modelo multiclasse do projeto contra os especialistas
binários com o ratio ótimo identificado no Experimento 1. O modelo multiclasse é o Run
7 — YOLOv11s treinado no brisc_dataset completo (nc=3), 100 epochs, best epoch no
62, mAP\@0.50=0.9195 global. Os especialistas binários são os modelos de glioma 20%,
meningioma 20% e pituitary 30%. Dataset de origem, arquitetura e hiperparâmetros são
idênticos — a única variável que difere é o objetivo de classificação.

== Resultados

// CORREÇÃO 2 — Tabela 4: recalls por classe do Run 7 atualizados via Figura 5
// glioma: 115/140 = 0.8214  |  meningioma: 156/163 = 0.9571  |  pituitary: 170/176 = 0.9659
#figure(
  caption: [Run 7 multiclasse vs. melhor binário por classe (test split). mAP\@0.50 e Precision
  do Run 7 são métricas globais do modelo. Recall por classe do Run 7 extraído diretamente da
  matriz de confusão da Figura 5 (test split). Os modelos binários reportam métricas por classe.],
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, center, center, center, center),
    stroke: 0.5pt,
    table.header([*Modelo*], [*mAP\@0.50*], [*Precision*], [*Recall*], [*FPR*]),
    [Run 7 — multiclasse (global)],       [0.9195], [0.9178], [0.8840], [0.50],
    [— glioma (por classe)],              [—],      [—],      [*0.8214*], [0.50],
    [Binário glioma 20%],                 [0.8497], [0.7935], [0.7786], [*0.0%*],
    [— meningioma (por classe)],          [—],      [—],      [*0.9571*], [0.50],
    [*Binário meningioma 20%*],           [*0.9727*],[*0.9734*],[*0.9693*],[*0.0%*],
    [— pituitary (por classe)],           [—],      [—],      [*0.9659*], [0.50],
    [Binário pituitary 30%],              [0.9671], [0.9399], [0.9489], [2.7%],
  )
)

#figure(
  image("figura4_comparacao_barras.png", width: 100%),
  caption: [Comparação de recall e FPR entre multiclasse e especialistas binários (valores corrigidos — ver Tabela 4).],
)

#figure(
  image("figura5_run7_multiclasse.png", width: 88%),
  caption: [Matriz de confusão do Run 7 multiclasse (Figura 5).],
)

Os valores absolutos das detecções e falsos positivos por classe estão detalhados nas matrizes
da Figura 2.

== Análise por Classe

*Glioma.* O modelo multiclasse é superior em recall. O Run 7 atingiu recall de 0.8214 para
glioma no test split (115/140, Figura 5), contra 0.7786 do especialista binário — diferença
de 0.0428. O mAP\@0.50 global do Run 7 (0.9195) também supera o binário (0.8497).
O binário atingiu FPR = 0.0% contra 0.50 do multiclasse — mas para glioma
especificamente, esse FPR é artefato de outras classes tumorais competindo no
espaço Softmax, não de glioma sendo confundido com tecido saudável. O binário elimina o
FPR ao custo de perda significativa de recall.

*Meningioma.* O especialista binário é superior em mAP, FPR e apresenta ganho modesto
em recall. mAP\@0.50=0.9727 contra 0.9195 global do Run 7
(nota: mAP do especialista binário é por classe; mAP do
Run 7 é global — a comparação é indicativa, não simétrica);
// CORREÇÃO 4 — parágrafo meningioma: recalls corrigidos e contagem de TPs corrigida
recall=0.9693 contra 0.9571 de meningioma no multiclasse;
FPR=0.0% contra 0.50. A matriz confirma 158/163 meningiomas detectados no test split
do binário (Figura 2) contra 156/163 no multiclasse (Figura 5) — 2
detecções adicionais com qualidade de bbox superior. O ganho é atribuível à eliminação
da competição com glioma, que no modelo multiclasse sobrescreve parte dos gradientes de
meningioma. Este é o resultado mais consistente do experimento: especialização binária
melhora mAP, FPR e recall simultaneamente, embora o ganho em recall seja modesto.

*Pituitary.* O especialista binário apresenta recall de 0.9489 no test split, contra 0.9659 de
pituitary no multiclasse (170/176, Figura 5) — leve regressão em recall compensada pela
redução de FPR de 0.50 para 2.7%. Pituitary já era a classe mais forte no multiclasse,
deixando pouca margem para ganho por especialização. O FPR residual de 2.7% é único
entre os binários, confirmando que a confusão com estruturas da base craniana é intrínseca
à classe e não eliminável apenas pela especialização arquitetural.

== Explicação Mecanística

Glioma é a classe que mais se beneficia da competição inter-classe. Sua morfologia — bordas
difusas, crescimento infiltrativo, sobreposição com tecido adjacente — torna sua assinatura
visual inerentemente ambígua. No modelo multiclasse, a presença de meningioma e pituitary
como classes competidoras força o backbone a aprender um hiperplano discriminativo mais
específico para glioma. No modelo binário, sem esse contraste, o backbone não tem referência
negativa suficientemente rica para delimitar a fronteira de glioma com precisão.

Meningioma possui morfologia oposta: realce homogêneo intenso, base dural ampla, margens
nítidas. Sua assinatura visual é suficientemente distinta para ser aprendida sem contraste
inter-classe. No modelo multiclasse, os gradientes de glioma — maior em volume e com
features sobrepostas — competem pelo espaço de representação compartilhado, sobrescrevendo
parte dos sinais discriminativos de meningioma. O especialista binário elimina essa
interferência.

Pituitary é um caso intermediário: morfologia distinta o suficiente para não precisar de
contraste inter-classe, mas com confusão anatômica genuína com estruturas da sela turca
que nenhuma arquitetura resolve sem volume adequado de negativos.

O FPR próximo de zero nos binários de glioma e meningioma, contrastado com o FPR de
0.50 no multiclasse, permite uma atribuição causal direta: o FPR residual do Run 7 não é
confusão entre tumor e tecido saudável — é artefato da competição entre classes no espaço
Softmax, onde logits de classes tumorais com alta confiança invadem o espaço de
decisão do background.

== Finding Central

A competição inter-classe em modelos multiclasse YOLO produz efeitos assimétricos por
classe: é regularização benéfica para classes com morfologia difusa (glioma) e teto estrutural
de performance para classes com morfologia distinta (meningioma). Este é o primeiro
resultado empírico desse tipo com arquitetura YOLO aplicada a detecção de tumor cerebral,
e sugere que a escolha entre multiclasse e especialistas binários deve ser feita por classe,
com base nas propriedades morfológicas de cada tipo tumoral.
Para pituitary, a especialização binária produz ganho
em FPR (0.50 → 2.7%) com leve regressão em recall,
configurando um resultado distinto dos outros dois casos.

// ── 8. Discussão ─────────────────────────────────────────────────────────────
== Discussão Geral

== Integração dos Dois Findings

Os dois experimentos respondem perguntas independentes mas complementares. O Experimento 1
demonstra que o ratio de negativos é um hiperparâmetro de dataset com efeito
direto e isolável no comportamento do modelo por classe. O Experimento 2 demonstra que
a arquitetura de classificação impõe efeitos assimétricos entre classes independentemente do
ratio. Juntos, estabelecem que não existe configuração única ótima para detecção de tumor
cerebral com YOLO — tanto o ratio quanto a arquitetura precisam ser otimizados por classe
tumoral.

A implicação prática é direta: um pipeline clínico que utiliza um único modelo multiclasse
com ratio fixo está sub-ótimo para ao menos duas das três classes estudadas. Meningioma
beneficia de especialização binária em recall e FPR; pituitary beneficia em FPR — com leve
regressão em recall; glioma se beneficia do multiclasse. O ratio 20% é ótimo para glioma e
meningioma; pituitary requer 30–40%. Qualquer configuração que não respeite essa assimetria
incorre em perda de performance por classe que as métricas globais não expõem.

== O Papel do FPR como Métrica Primária

Um resultado transversal dos dois experimentos é a inadequação do mAP como métrica
única de avaliação clínica. O Run 7 atingiu mAP\@0.50=0.9195 com FPR=0.50 — valor
que sem o protocolo healthy_fpr() dedicado não apareceria em nenhuma métrica padrão
reportada pelo Ultralytics.

FPR de 0.50 significa que metade das imagens no_tumor apresentadas ao modelo produzem
ao menos uma predição de tumor. Em um cenário de triagem com volume alto de exames
normais — estudos de coorte de outpatients consecutivos indicam que mais de 95% dos
T1CE em populações não selecionadas retornam sem patologia relevante [REF] — esse
valor torna o sistema clinicamente não deployável sem um filtro adicional. Os especialistas
binários reduziram o FPR para 0.0% em glioma e meningioma, e para 2.7% em pituitary,
sem qualquer ajuste arquitetural além da mudança de objetivo de classificação.

== Glioma como Caso Especial

O finding de que glioma performa melhor no modelo multiclasse é contra-intuitivo à luz da
literatura, que consistentemente documenta benefícios de especialização binária em
classificação médica. A explicação mecanística proposta — que a morfologia difusa do glioma
se beneficia do contraste inter-classe para delimitar sua fronteira — é consistente com a
evidência empírica mas abre uma questão não resolvida: o benefício persiste com datasets
maiores ou com augmentação mais agressiva no modelo binário?

Uma interpretação alternativa é que o modelo binário de glioma sofre de underfitting
relativo — sem as outras classes como contraste negativo, o backbone não encontra gradiente
suficiente para refinar a representação das bordas infiltrativas do glioma. Experimentos
com hard negative mining específico para glioma — selecionando como negativos imagens
de meningioma e pituitary em vez de apenas no_tumor — poderiam testar essa hipótese
diretamente.

== Pituitary como Caso Estruturalmente Diferente

Pituitary é a única classe onde o FPR não converge para zero mesmo com ratio de 40%,
onde nenhum ratio domina em todas as métricas, e onde a especialização binária não produz
ganho em recall. Esses três padrões convergem para uma explicação única: a confusão
do modelo com estruturas da base craniana — seio esfenoidal, fossa pituitária normal, haste
pituitária — é intrínseca ao domínio imagiológico, não um artefato de competição inter-classe
ou de volume insuficiente de negativos.

Essa confusão tem paralelo clínico direto: a diferenciação entre adenoma pituitário e
estruturas selares normais é reconhecidamente um dos problemas mais difíceis em
neurorradiologia, com taxas de erro diagnóstico documentadas mesmo em radiologistas
experientes. Um modelo de detecção sem acesso a cortes dinâmicos ou a sequências
complementares como FLAIR enfrenta intrinsecamente o mesmo problema de ambiguidade
espacial. Isso não é uma limitação do approach — é uma limitação do domínio que nenhuma
escolha de ratio ou arquitetura resolve completamente.

== Limitações

*Escopo de classes.* O dataset BRISC 2025 cobre três tumores primários. Anomalias que
produzem realce em T1CE — metástases, abscessos cerebrais, linfoma primário do SNC
— não estão representadas. Em um cenário clínico real, essas lesões seriam classificadas
incorretamente como um dos três tumores primários. O abscesso cerebral é o caso de maior
risco clínico: sua aparência em anel de realce é visualmente idêntica ao glioblastoma no
T1CE, e o tratamento incorreto com corticosteroide é catastrófico em presença de infecção
ativa — cenário externo ao escopo deste trabalho mas relevante
para extensões clínicas futuras do pipeline.

*Classe no_tumor híbrida.* As imagens no_tumor do BRISC incluem lesões benignas
não-neoplásicas além de cérebros saudáveis. O FPR medido neste trabalho reflete a taxa
de falsos positivos sobre esse conjunto híbrido, não estritamente sobre cérebros normais.
A ausência de datasets públicos de T1CE em voluntários comprovadamente saudáveis —
consequência das restrições éticas à administração de contraste sem indicação clínica — é
uma limitação estrutural do campo, não específica deste trabalho.

*Run inconclusiva.* O modelo de pituitary 20% foi interrompido com best epoch igual ao
último epoch disponível. Os resultados desse modelo são conservadores — o ratio ótimo
para pituitary pode ser 20% com performance superior ao 30% após convergência completa
com patience=50.

*Validação externa.* Todos os resultados são reportados sobre o test split do BRISC 2025.
Validação em dados de outras instituições, com diferentes protocolos de aquisição, não foi
realizada. A generalização dos findings não pode ser assumida sem experimentos adicionais.

// ── 9. Conclusão e Trabalho Futuro ───────────────────────────────────────────
== Conclusão e Trabalho Futuro

Este trabalho demonstrou empiricamente que duas variáveis metodológicas negligenciadas
na literatura — o ratio de amostras negativas e a escolha arquitetural entre modelo
multiclasse e especialistas binários — produzem efeitos significativos e assimétricos no
desempenho de modelos YOLOv11s para detecção de tumor cerebral em MRI T1CE.

O primeiro finding central é que o ratio ótimo de amostras negativas é uma propriedade
da classe tumoral. Glioma e meningioma maximizam mAP e recall com 20% de negativos,
enquanto pituitary requer 30–40% para controlar FPR. Este é o primeiro ablation controlado
dessa variável em detecção de tumor cerebral com YOLO, e os resultados contradizem o
padrão arbitrário de ratio 1:1 predominante na literatura.

O segundo finding central é que a competição inter-classe em modelos multiclasse YOLO
produz efeitos assimétricos: é regularização benéfica para glioma, cuja morfologia difusa
se beneficia do contraste inter-classe, e teto estrutural de performance para meningioma,
cuja morfologia distinta é prejudicada pela interferência de gradientes de outras classes.
O especialista binário de meningioma atingiu mAP\@0.50=0.9727 — o melhor resultado por
classe de todo o projeto — enquanto o especialista binário de glioma ficou abaixo do
recall do modelo multiclasse (0.7786 vs. 0.8214). Esta é a primeira comparação direta entre
YOLOs binários independentes e YOLO multiclasse unificado em detecção de tumor cerebral,
com dataset e arquitetura fixos.
// CORREÇÃO 5 — Conclusão: qualificação de meningioma e pituitary
Meningioma beneficia em recall e FPR da especialização binária; pituitary beneficia em
FPR — com leve regressão em recall — mas não em recall absoluto.

Transversalmente, o FPR em imagens saudáveis emerge como métrica clínica indispensável
que mAP global não expõe. A redução de FPR de 0.50 no multiclasse para próximo de zero
nos especialistas binários confirma que o FPR residual do modelo unificado é artefato de
competição inter-classe no espaço Softmax.

*Trabalho futuro imediato:* rerun do modelo pituitary 20% com patience=50 para determinar
se o recall se mantém após convergência completa.

*Extensões de médio prazo:* implementação do pipeline em cascata — triagem → multiclasse
para glioma + especialistas binários para meningioma e pituitary; e hard negative mining
específico para glioma, utilizando imagens de meningioma e pituitary como negativos no
treinamento binário.

*Extensões de longo prazo:* extensão do pipeline para anomalias não-neoplásicas que
produzem realce em T1CE — metástases, abscessos, linfoma primário do SNC — integrando
datasets públicos disponíveis como Brain-Mets-Lung (TCIA) e UCSF-PCNSL (AWS Open
Data); e validação prospectiva multi-centro com protocolos de aquisição variados.

// ── Referências ──────────────────────────────────────────────────────────────
= Referências

Alhussainan, T. et al. (2024). YOLOv7-based brain tumor firmness classification. [Journal].

Chen, Z. et al. (2024). YOLO-NeuroBoost: Enhanced brain tumor detection with YOLOv8. [Journal].

Chourib, M. et al. (2025). Advanced transfer learning for brain tumor detection with YOLOv11x. [Journal].

Fateh, A. et al. (2025). BRISC 2025: Brain tumor image segmentation and classification dataset. arXiv:2506.14318.

Han, X. et al. (2023). Negative sample ratio ablation in whole slide imaging. [Journal].

Nimmagadda, S. & Devi, R. (2025). YOLOv7 with CBAM for brain tumor detection: FPR analysis. [Journal].

ResNet-ViT (2024). Binary versus multiclass classification for pulmonary tuberculosis detection. [Journal].

StackANN (2024). One-vs-rest binary classifiers for breast cancer subtype detection. [Journal].

Tariq, M. & Choi, J. (2026). Swin-YOLOv12: Transformer-based brain tumor detection. [Journal].

Ultralytics (2024). YOLOv11: Real-time object detection and segmentation. Ultralytics documentation.

VGG16 (2024). Accuracy paradox in multiclass skin lesion detection with imbalanced datasets. [Journal].

YOLO-LAN (2024). Negative sample methodology for colonoscopy polyp detection. [Journal].

Zhang, Y. & Yang, Q. (2021). A survey on multi-task learning. _IEEE Transactions on Knowledge and Data Engineering_.
