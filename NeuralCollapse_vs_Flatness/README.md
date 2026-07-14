# NeuralCollapse vs Flatness

Este directorio contiene el código y análisis del paper:

> **"Flatness is Necessary, Neural Collapse is Not: Rethinking Generalization via Grokking"**
> Han, Adilova, Petzka, Kleesiek, Kamp (NeurIPS 2025)

## Estructura

```
NeuralCollapse_vs_Flatness/
├── grokking_nc_rf/               # Experimentos de grokking: NC + Relative Flatness
├── nc_experiments/               # Experimentos de Neural Collapse en ResNet-18
├── rf_resnet18/                  # Relative Flatness en ResNet-18
├── rf_vit/                       # Relative Flatness en Vision Transformer
├── rf_bert/                      # Relative Flatness en BERT
├── rf_gpt2/                      # Relative Flatness en GPT-2
├── figure/                       # Figuras del paper
│
├── report_implementation.pdf     # Informe técnico de la implementación
├── analysis_ncc_representations.pdf  # Análisis profundo: NCC y representaciones alternativas
├── README.md
└── requirements.txt
```

## Documentos generados

| Documento | Págs. | Descripción |
|-----------|-------|-------------|
| `report_implementation.pdf` | 11 | Informe técnico detallado de toda la implementación del código |
| `analysis_ncc_representations.pdf` | 28 | Análisis de cómo penalizar NCC produce formas alternativas de colapso |

## Análisis original: Colapso Alternativo bajo Penalización NCC

El documento **`analysis_ncc_representations.pdf`** contiene nuestra contribución original:

- **Hipótesis:** Cuando la métrica NCC se penaliza (loss = CE - λ·NCC), el colapso neural tradicional (NC1-NC4) se rompe, pero **emerge una forma alternativa de colapso** que permite la generalización.
- **4 tipos propuestos:** Subspace Collapse, Soft Collapse, Kernel-Induced Collapse, Rank-Constrained Collapse
- **8 propuestas de análisis** para validar estas hipótesis experimentalmente

## Créditos

Código original: [TrustworthyMachineLearning-Lab/grokking_flatness](https://github.com/TrustworthyMachineLearning-Lab/grokking_flatness)
