# CENIA Research

Repositorio de investigación del **Centro Nacional de Inteligencia Artificial (CENIA)** de Chile.

## Position_symbolic_dynamics

Este subdirectorio contiene el estudio completo sobre **Positional versus Symbolic Attention Heads** basado en el paper:

> **"Positional versus Symbolic Attention Heads: Learning Dynamics, RoPE Geometry, and Length Generalization"**
> Urrutia, Alegría, Sanchez Macias, Salas, Calderon, Rojas (2026)
> arXiv: 2605.31558 | NeurIPS 2026

### Contenido

| Archivo | Descripción |
|---------|-------------|
| `paper_source/paper.pdf` | Artículo original (arXiv) |
| `positional_vs_symbolic_learningdynamics.pdf` | Versión del artículo (PDF adjunto) |
| `report_paper.md` | Informe detallado del artículo |
| `report_code.md` | Informe detallado del código |
├── code/                      # Código fuente original del repositorio
│   ├── LICENSE
│   ├── README.md
│   └── src/
│       ├── cmds/             # Puntos de entrada (experimentos)
│       │   ├── multihop_exp.py          # Number task experiment
│       │   ├── symbolic_exp.py          # Letter task experiment
│       │   ├── get_attention_weights.py # Extracción de atención
│       │   ├── compute_metrics.py       # Positional/symbolic scores
│       │   ├── compute_metrics_multiple_queries.py
│       │   ├── compute_symbolic_metrics.py
│       │   └── schemas.py               # Pydantic configs
│       ├── data/             # Datasets y tokenizadores
│       │   ├── dataset.py               # Number task dataset
│       │   ├── symbolic_dataset.py      # Letter task dataset
│       │   ├── generate_dataset_sym.py  # Generación helper
│       │   └── utils.py                 # Validación
│       ├── lib/              # Librerías
│       │   ├── metrics.py               # Pos/Sym scores
│       │   └── s3.py                    # S3 upload/download
│       └── models/           # Entrenamiento
│           └── train.py                 # Trainer personalizado

### Enlaces

- **Paper:** https://arxiv.org/abs/2605.31558
- **Código original:** https://anonymous.4open.science/r/number-and-letter-neurips26-C754
- **CENIA:** https://cenia.cl
