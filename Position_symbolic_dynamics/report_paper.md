# Informe del Artículo: Positional versus Symbolic Attention Heads

## Resumen Ejecutivo

**Título:** Positional versus Symbolic Attention Heads: Learning Dynamics, RoPE Geometry, and Length Generalization

**Autores:** Felipe Urrutia, Juan José Alegría, Cinthia Sanchez Macias, Jorge Salas, Cristian B. Calderon, Cristobal Rojas

**Afiliación:** CENIA & Faculty of Mathematics UC, Santiago, Chile

**arXiv:** 2605.31558 (Mayo 2026)

Este artículo estudia cómo los **attention heads** en transformers desarrollan comportamientos **posicionales** versus **simbólicos** durante el entrenamiento, y cómo estos comportamientos afectan la capacidad de generalización a secuencias más largas.

---

## 1. Motivación y Contexto

Los transformers modernos (GPT, Claude, etc.) usan **Rotary Positional Encoding (RoPE)** para codificar información posicional. Sin embargo, no se entendía bien cómo los attention heads aprenden a usar la información posicional (atender a posiciones específicas) versus información simbólica (atender a tokens específicos sin importar su posición).

Trabajo previo de Urrutia et al. (2025) introdujo **métricas** para clasificar heads como posicionales o simbólicas, y observaron que en modelos reales las capas tempranas tienden a ser posicionales y las profundas simbólicas. Este artículo **extiende ese análisis al estudio de la dinámica de aprendizaje** y a la **generalización a largo contexto**.

---

## 2. Tareas Propuestas

### 2.1 Number Task (Tarea Numérica)

- **Entrada:** Una secuencia de letras seguida de números.
- **Mecanismo:** Cada número indica cuántas posiciones retroceder en la secuencia.
- **Ejemplo (4-hop):** `C B A ... 1 4 1 3 2 1 1` → hop1: retrocede 1 → hop2: retrocede 4 → hop3: retrocede 1 → hop4: retrocede 3 → respuesta: letra en posición final.
- **Tipo de razonamiento:** **Posicional** — requiere seguir posiciones relativas.
- **Vocabulario:** 120 letras (a-z + bigramas aa-dp) + 16 enteros {1,...,16} = 136 tokens.

### 2.2 Letter Task (Tarea de Letras)

- **Entrada:** Una secuencia de pares letra-letra seguida de pares letra-número.
- **Mecanismo:** Cada par (X,Y) indica "busca el par cuya primera letra es Y".
- **Ejemplo (4-hop):** `(I,G) (G,D) (D,E) (E,C) (H,F) ... (C,2)` → hop1: G busca a (I,G) → hop2: D busca a (G,D) →... → respuesta: (C,2).
- **Tipo de razonamiento:** **Simbólico** — requiere seguir asociaciones entre símbolos.
- **Vocabulario:** 8 letras {a,...,h} + 8 enteros {1,...,8} = 64 pares × 2 tipos = 128 tokens.

Ambas tareas son **estructuralmente equivalentes** (misma estructura multi-hop), pero demandan **mecanismos distintos**.

---

## 3. Configuración Experimental

### 3.1 Arquitectura del Modelo

- **Modelo:** GPT-J (decoder-only Transformer)
- **Capas:** 12
- **Attention heads:** 1 por capa (single-head, para aislar mecanismos)
- **Dimensión oculta:** 128
- **Posicional encoding:** RoPE
- **Activación:** GELU
- **Dropout:** 0.1
- **Objetivo:** Last-token prediction (solo se optimiza la pérdida en el último token)

### 3.2 Dataset

- **480,000 secuencias** por tarea
- Cada secuencia: 17 tokens (8 target window + 8 pre-target window + 1 query)
- Balanceado: misma proporción de casos 1-hop, 2-hop, 3-hop, 4-hop
- Split: 90% train / 0.5% validation / 9.5% test

### 3.3 Entrenamiento

- Entrenamiento con 3 NVIDIA RTX A5000 (24GB cada una)
- Tiempo total: ~55 minutos por modelo
- Métricas de positional/symbolic score: 10h GPU + 17h CPU

---

## 4. Métricas: Positional y Symbolic Scores

Basado en Urrutia et al. (2025), se definen dos métricas para cuantificar el comportamiento de un attention head:

### 4.1 Definición Formal

Un head actúa **posicionalmente** si su distribución de atención es **invariante** ante permutaciones de los key vectors. Es decir, atiende a posiciones específicas sin importar qué token esté ahí.

Un head actúa **simbólicamente** si su distribución de atención es **equivariante** ante permutaciones. Es decir, atiende a tokens específicos sin importar su posición.

### 4.2 Scores

Se calculan mediante:
- **Permutaciones por pares** (swaps) entre posiciones i, j
- **Peso de cada permutación** basado en la masa de atención movida (softmax con temperatura)
- **Positional Score:** Similitud coseno entre atención original y atención tras permutar
- **Symbolic Score:** Similitud coseno entre atención original y atención "canónica" tras permutar

---

## 5. Resultados Principales

### 5.1 Dinámica de Aprendizaje (Figura 2)

**Number Task:**
- El accuracy emerge **progresivamente**: primero se aprende 1-hop, luego 2-hop, etc.
- Se requieren **heads posicionales** para el Indexing + **heads simbólicas** para propagación.
- Las heads posicionales aparecen en capas medias (L3-L6).
- Las heads simbólicas aparecen en capas profundas (L9-L11).

**Letter Task:**
- El accuracy emerge **simultáneamente** para todos los hops.
- Solo se requieren **heads simbólicas** (Retrieval).
- La última head simbólica en L11 es necesaria para todos los hops.
- Las heads posicionales no son necesarias.

### 5.2 Pureza de Heads

La **pureza** (heads que son claramente posicionales O simbólicas, no mixtas) es necesaria para alcanzar máxima precisión. Esto valida las métricas como herramienta de interpretabilidad.

### 5.3 Funciones Mecanísticas Identificadas

Tres funciones básicas implementadas por los heads puros:

1. **Selective Index (Index):** Dada una posición con un número n, copia el token n posiciones atrás, pero solo si es una letra. → **Posicional**

2. **Retrieval:** Dado un par (X,Y), busca el par anterior cuya primera letra sea Y y lo copia. → **Simbólico**

3. **Reflexive:** Copia el token actual a la siguiente capa sin cambios. → Puede ser posicional o simbólico (en la práctica, simbólico).

### 5.4 Descomposición Mecanística

Para una entrada con h hops y un modelo con ℓ ≥ h capas:

**Number Task:** `f_NUM(σ) = Reflexive^(ℓ-h)(Index^h(σ))[n]`

**Letter Task:** `f_LET(σ) = Reflexive^(ℓ-h)(Retrieval^h(σ))[n]`

---

## 6. Análisis Teórico: Mecanismos Geométricos en RoPE

### 6.1 Teorema 3.1 (Factibilidad)

Existe un Transformer `T_Index` con atención RoPE 2D que implementa Index, y un `T_Retrieval` que implementa Retrieval, ambos con una sola cabeza de atención y sin funciones de activación.

**Mecanismo para Index (Posicional):**
- Key vectors: todos alineados (iguales a un vector v)
- Query vectors: codifican la posición a atender mediante rotación RoPE
- La atención máxima ocurre exactamente en la posición deseada (n - σ_n)
- Los value vectors solo copian si el token es letra (mapeo diagonal)

**Mecanismo para Retrieval (Simbólico):**
- Key vectors: codifican la letra izquierda del par mediante rotación con ángulo ω
- Query vectors: codifican la letra derecha del par consultante
- La atención máxima ocurre donde la letra izquierda del key coincide con la letra derecha del query
- La posición exacta no importa, solo la identidad del símbolo

### 6.2 Remark 3.1

Index **no puede** implementarse con un head puramente simbólico.
Retrieval **no puede** implementarse con un head puramente posicional.

Esto demuestra la **separación fundamental** entre ambos tipos de mecanismos.

---

## 7. Análisis de Robustez: Discrepancia

### 7.1 Definición de Discrepancia (Definición 3.1)

La **discrepancia Δ** de un attention head es la diferencia máxima entre el logit más grande y el segundo más grande para entradas de cierta longitud.

Intuitivamente: mide qué tan distinguible es el token correcto del resto. Una discrepancia de 0 significa que el modelo no puede distinguir el token correcto.

### 7.2 Teorema 3.2 (Cotas de Discrepancia)

**Para Index (posicional):**
Δ_Index(n) ≤ min{1 - cos(θ), π²/(2n²)}

**Para Retrieval (simbólico):**
Δ_Retrieval(n) ≤ 1 - cos(ω - nθ)

**Interpretación:**
- Index: la discrepancia decae como **O(1/n²)** — se degrada rápidamente con la longitud.
- Retrieval: la discrepancia se mantiene **no nula** incluso para n grandes si θ es pequeño.
- Si θ = 0 (NoPE), Δ_Retrieval se mantiene positiva para cualquier longitud.

Esto demuestra cuantitativamente que **los mecanismos simbólicos generalizan mejor** a secuencias largas.

### 7.3 Validación Experimental (Figura 5)

Los resultados confirman las predicciones teóricas:

**Modelo entrenado GPT-J:**
- Letter task: >90% de precisión hasta 850 tokens (53× la longitud original)
- Number task: la precisión cae rápidamente después de la longitud de entrenamiento

**Modelos frontera (GPT 5.4, GPT 5.5, Claude Sonnet 3.7):**
- Letter task: precisión >0.88 a longitud 32, >0.65 a longitud 100
- Number task: precisión <0.5 a longitud 32, <0.1 a longitud 100
- Consistente con las predicciones del Teorema 3.2

---

## 8. Conclusiones y Contribuciones

### Contribuciones Principales:

1. **Tareas estructuralmente equivalentes** que requieren perfiles de atención distintos (posicional vs. simbólico).

2. **Relación entre pureza de heads y precisión:** La máxima precisión requiere heads puros (claramente posicionales o simbólicos).

3. **Descomposición mecanística** de ambas tareas en funciones básicas (Index, Retrieval, Reflexive).

4. **Construcciones teóricas** que muestran cómo RoPE puede implementar estas funciones con interpretación geométrica.

5. **Noción de discrepancia** para cuantificar la degradación con la longitud.

6. **Separación cuantitativa** entre mecanismos posicionales y simbólicos en términos de generalización.

### Implicaciones:

- Los mecanismos simbólicos son **intrínsecamente más robustos** para longitudes largas.
- Esto sugiere que entrenar modelos para que desarrollen estrategias simbólicas (vs. posicionales) puede mejorar la generalización.
- Las métricas positional/symbolic scores pueden servir como **herramienta de monitoreo** durante el entrenamiento de LLMs.

### Limitaciones:

- Configuración controlada con single-head por capa (no multi-head como modelos reales).
- Vocabulario finito limita pruebas de extrapolación.
- Solo se estudiaron tareas multi-hop simples.

---

## 9. Referencias Clave

- Urrutia et al. (2025) - "Decoupling positional and symbolic attention behavior in transformers" (arXiv:2511.11579) — Introduce las métricas positional/symbolic scores
- Barbero et al. (2024) - "Round and round we go! What makes rotary positional encodings useful?" — Observa que heads simbólicas usan frecuencias bajas de RoPE
- Su et al. (2024) - "RoFormer: Enhanced Transformer with Rotary Position Embedding" — Paper original de RoPE
- Pasten et al. (2025) - "Continuity and isolation lead to doubts or dilemmas in large language models" — Sobre la dispersión de atención con softmax
