## A Ideia do 2.5D

A estratégia de empilhamento 2.5D é conceitualmente elegante e clinicamente fundamentada. MRI é por natureza um exame volumétrico — o tumor existe no espaço tridimensional e sua aparência muda entre fatias adjacentes. Tentar capturar essa continuidade volumétrica em um modelo 2D, sem o custo computacional de redes 3D, é uma otimização inteligente e amplamente explorada na literatura médica.

A execução foi o problema, não a ideia. O dataset Figshare numera os 3064 arquivos globalmente, sem garantia de que arquivos numericamente adjacentes pertencem ao mesmo paciente ou representam fatias espacialmente consecutivas. Sem metadados de posição Z explícitos nos `.mat` files — confirmado pela inspeção do `orientation_report.txt` — qualquer tentativa de reconstruir a sequência volumétrica seria uma suposição não verificável. O modelo das Runs 1 a 3 aprendeu a detectar tumores apesar de dois dos três canais RGB serem ruído anatômico de pacientes diferentes — o que por si só é um resultado fascinante sobre a robustez do YOLOv11s.

---

## A Migração do Figshare para o BRISC

Três fatores simultâneos explicam o salto de qualidade:

**Volume e diversidade real.** O BRISC trouxe 6000 imagens contra 3064 — mas o ganho não foi apenas quantitativo. As 1635 imagens únicas de meningioma contra 708 do Figshare deram ao modelo diversidade anatômica real que nenhuma técnica de oversampling consegue simular. A diferença entre memorizar 708 tumores e aprender de 1635 pacientes distintos é exatamente o que se reflete no salto de 0.74 para 0.94 na classe meningioma.

**Balanceamento estrutural.** O BRISC foi construído com distribuição equilibrada entre planos (axial, coronal, sagital) e classes. O Figshare, apesar de ter os 3 planos, não tinha labels de plano por imagem — o modelo treinava sem saber qual perspectiva anatômica estava vendo. O BRISC resolve isso pela convenção de nomenclatura dos arquivos.

**O no_tumor como negative sample.** Esta foi a correção mais impactante para o comportamento clínico do modelo. O background→glioma de 0.87 no Run 5 não era um problema de arquitetura nem de hiperparâmetros — era ausência de referência. O modelo nunca havia visto um cérebro saudável em T1CE e não tinha como aprender o que é ausência de tumor. Os 1207 negative samples do BRISC, combinados com o Empty Annotation method validado como padrão definitivo pela literatura, levaram o FPR a 0.0% em 120 imagens saudáveis do test split.

---

## O Papel da Documentação Sistemática

O `run_analysis.md` foi uma decisão de processo que permitiu fazer inferências que seriam impossíveis sem ele. A confusion matrix do Run 1 mostrou o Background→Glioma de 0.77 — mas sem a documentação estruturada das runs seguintes, a correlação entre oversampling com duplicatas exatas e a divergência val-train de 0.784 no Run 5 nunca teria sido identificada como causalidade. Foi a comparação direta entre Run 4 (divergência 0.497, meningioma 0.80) e Run 5 (divergência 0.784, meningioma 0.74) que revelou que o problema não era o dataset — era a qualidade das duplicatas.

A granularidade da análise época a época via `results.csv` foi igualmente determinante. O Run 3 reportava mAP@0.50=0.9290 no PIPELINE SUMMARY — um resultado aparentemente sólido. A análise do CSV revelou que o best epoch foi na época 36 e as 64 épocas seguintes foram oscilação pura. Sem esse diagnóstico, o `cos_lr=True` e o ajuste de patience nunca teriam sido identificados como intervenções necessárias. O Run 7 com best epoch na 62 e oscilação std de 0.0027 nas últimas 20 épocas é consequência direta de ter documentado e analisado o que os números agregados escondiam.