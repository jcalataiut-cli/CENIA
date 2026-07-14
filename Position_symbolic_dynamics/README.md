# Position_symbolic_dynamics

## Positional versus Symbolic Attention Heads: Learning Dynamics, RoPE Geometry, and Length Generalization

Este directorio contiene el análisis completo del paper de Urrutia et al. (2026) que estudia cómo los attention heads en transformers desarrollan comportamientos posicionales versus simbólicos durante el entrenamiento.

### Estructura

```
Position_symbolic_dynamics/
├── paper_source/
│   └── paper.pdf                    # Artículo original (PDF de arXiv)
├── positional_vs_symbolic_learningdynamics.pdf  # PDF adjunto
├── code_reconstruction/             # Código fuente reconstruido
├── report_paper.md                  # Informe explicativo del artículo
└── report_code.md                   # Informe explicativo del código
```

### Resumen

- **Tarea Numérica (Number Task):** requiere razonamiento posicional (Index) → heads posicionales
- **Tarea de Letras (Letter Task):** requiere razonamiento simbólico (Retrieval) → heads simbólicos
- **Resultado clave:** los mecanismos simbólicos generalizan mucho mejor a secuencias largas que los posicionales
- **Herramienta teórica:** noción de **discrepancia** que cuantifica la degradación con la longitud
