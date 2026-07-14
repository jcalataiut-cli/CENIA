# Informe del Código: Positional vs Symbolic Attention Heads

## Estructura Completa del Código

Este documento explica en detalle todos los componentes del código utilizado en el paper "Positional versus Symbolic Attention Heads: Learning Dynamics, RoPE Geometry, and Length Generalization".

Basado en la descripción del paper, el código se organiza en los siguientes módulos:

```
number-and-letter-neurips26-C754/
├── LICENSE                        # MIT License
├── README.md                      # Descripción del repositorio
└── src/
    ├── cmds/                      # Puntos de entrada (scripts ejecutables)
    │   ├── multihop_exp.py                    # Experimento Number Task (entrenamiento)
    │   ├── symbolic_exp.py                    # Experimento Letter Task (entrenamiento)
    │   ├── get_attention_weights.py           # Extracción de atención con permutaciones
    │   ├── compute_metrics.py                 # Cálculo positional/symbolic scores
    │   ├── compute_metrics_multiple_queries.py# Scores para múltiples posiciones query
    │   ├── compute_symbolic_metrics.py        # Scores para Letter Task
    │   └── schemas.py                         # Esquemas Pydantic de configuración
    ├── data/                       # Generación de datasets y tokenizadores
    │   ├── dataset.py                          # Number Task: SequenceMultiHopDataset
    │   ├── symbolic_dataset.py                 # Letter Task: SequenceSymbolicDataset
    │   ├── generate_dataset_sym.py             # Helper para generar Letter dataset
    │   └── utils.py                            # Validación de secuencias
    ├── lib/                        # Librerías compartidas
    │   ├── metrics.py                          # Positional/Symbolic scores
    │   └── s3.py                               # Subida/descarga a S3
    └── models/                     # Entrenamiento personalizado
        └── train.py                            # Trainer + Callbacks + Métricas
```

---

## 1. Generación de Datasets

### 1.1 vocab.py

**Propósito:** Define los vocabularios para ambas tareas y las funciones de tokenización.

**Componentes:**
- `ALPHABET`: Conjunto de letras
  - Number Task: 26 letras + 94 bigramas (aa a dp) = 120 símbolos
  - Letter Task: 8 letras {a, b, c, d, e, f, g, h}
- `INTEGERS`: Conjunto de enteros
  - Number Task: {1, 2, ..., 16}
  - Letter Task: {1, 2, ..., 8}
- `NUM_VOCAB`: 136 tokens únicos (Number)
- `LET_VOCAB`: 128 tokens únicos (Letter)
- `tokenize(sequence)`: Convierte secuencia de tokens a índices
- `detokenize(indices)`: Convierte índices a tokens

**Detalle de implementación:**
```python
# Number Task vocabulary
# Tokens de letras: a-z (26) + aa, ab, ..., dp (94 bigramas)
# Tokens de números: 1, 2, ..., 16
# Total: 120 + 16 = 136 tokens

# Letter Task vocabulary
# Target window: pares letra-número (8 × 8 = 64 tokens atómicos)
# Pre-target window: pares letra-letra (8 × 8 = 64 tokens atómicos)
# Total: 64 + 64 = 128 tokens
```

### 1.2 number_task.py

**Propósito:** Genera el dataset para la Number Task.

**Proceso de generación:**

1. **Parámetros:**
   - `n1 = 8` (target window length)
   - `n2 = 8` (context/pre-target length)
   - `max_hops = 4`
   - `N = 480,000` (total sequences)

2. **Algoritmo de generación:**

```python
def generate_number_sequence(hops, target_letter, target_pos):
    """
    Genera una secuencia válida para Number Task.
    
    Args:
        hops: int (1-4), número de hops
        target_letter: str, letra a recuperar (respuesta)
        target_pos: int, posición de la letra en target window
    
    Returns:
        sequence: list[str], secuencia de tokens
    """
    # Target window: letras + números de hop
    target_window = [...]
    
    # Pre-target window: números de hop
    pre_target = [random_int() for _ in range(n2)]
    
    # Asegurar que la secuencia de hops está bien definida
    # Siguiendo la Definición B.1 del paper:
    # 1. jk1 = jn2 (último entero)
    # 2. n1 - jkHL + 1 ∈ {1, ..., n1}
    # 3. Para cada i: ki - k(i+1) = jki
    
    return target_window + pre_target
```

3. **Validación:** Cada secuencia debe tener un camino de hops único y bien definido.

4. **Balanceo:** Igual proporción de 1, 2, 3, 4 hops. Para cada hop class, balanceo de valor de solución y posición.

5. **Split:** 90% train (432,000), 0.5% validation (2,400), 9.5% test (45,600)

### 1.3 letter_task.py

**Propósito:** Genera el dataset para la Letter Task.

**Proceso de generación:**

1. **Parámetros:**
   - `ALPH = {a, b, c, d, e, f, g, h}` (8 letras)
   - `INT = {1, 2, ..., 8}` (8 enteros)
   - `n1 = 8` (target window)
   - `n2 = 8` (pre-target window)
   - `N = 480,000`

2. **Algoritmo:**

```python
def generate_letter_sequence(hops):
    """
    Genera una secuencia válida para Letter Task.
    
    Args:
        hops: int, número de hops
    
    Returns:
        sequence: list[tuple], pares (letra, letra) o (letra, número)
    """
    # Target window: secuencia de pares (letra, número)
    target_window = [(letter, num) for _ in range(n1)]
    
    # Pre-target window: pares (letra, letra)
    # Cada par (X,Y) indica: busca el par con primera letra = Y
    pre_target = []
    for h in range(hops):
        pair = (random_letter(), random_letter())
        pre_target.append(pair)
    
    # Validar que el camino es único:
    # y_ki = x_k(i+1)  (segunda letra del par i = primera letra del par i+1)
    # y_kHL = s_i* (última segunda letra apunta a target)
    
    return target_window + pre_target
```

3. **Misma estructura de datos:** 480k secuencias, 17 tokens, split 90/0.5/9.5

---

## 2. Modelo: GPT-J con Single Head

### 2.1 config.py

**Propósito:** Define la configuración del modelo.

```python
model_config = {
    "layers": 12,           # Número de capas
    "d_model": 128,         # Dimensión oculta
    "heads": 1,             # 1 attention head por capa (single-head)
    "d_head": 128,          # Dimensión por head = d_model (single head)
    "d_ff": 512,            # Dimensión feed-forward (4× d_model)
    "vocab_size": 136,      # Number Task: 136; Letter Task: 128
    "max_seq_len": 17,      # Longitud de secuencia fija
    "dropout_rate": 0.1,    # Dropout
    "activation": "gelu",   # GELU activation
    "use_rope": True,       # Rotary Positional Encoding
    "rope_base": 10000,     # Base frequency for RoPE
}
```

### 2.2 gptj.py

**Propósito:** Implementación del modelo GPT-J.

**Arquitectura:**

```python
class GPTJModel(nn.Module):
    """
    Decoder-only Transformer basado en GPT-J.
    
    Capas:
    1. Embedding: token → vector d_model
    2. 12 × TransformerBlock
       - LayerNorm
       - SingleHeadAttention (con RoPE)
       - Residual connection
       - LayerNorm
       - MLP (GELU activación)
       - Residual connection
    3. Final LayerNorm
    4. LM Head (proyección a vocab_size)
    """
    
    def __init__(self, config):
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.layers)
        ])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Atar pesos de embedding y lm_head
        self.lm_head.weight = self.embedding.weight
    
    def forward(self, input_ids):
        """
        Forward pass con causal masking.
        
        Args:
            input_ids: (batch, seq_len)
        
        Returns:
            logits: (batch, seq_len, vocab_size)
            attention_maps: list de matrices de atención por capa
        """
        x = self.embedding(input_ids)
        attention_maps = []
        
        for block in self.blocks:
            x, attn = block(x)  # attn: (batch, heads, seq_len, seq_len)
            attention_maps.append(attn)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits, attention_maps
```

### 2.3 layers.py

**Propósito:** Implementación de las capas individuales.

**SingleHeadAttention con RoPE:**

```python
class SingleHeadAttention(nn.Module):
    """
    Atención single-head con RoPE (Rotary Positional Encoding).
    
    Arquitectura estándar:
    - Q, K, V proyecciones lineales
    - RoPE aplicado a Q y K
    - Scaled dot-product attention con causal mask
    - Output projection
    """
    
    def __init__(self, config):
        self.d_model = config.d_model
        self.d_head = config.d_head  # = d_model (single head)
        
        # Proyecciones
        self.q_proj = nn.Linear(d_model, d_head, bias=False)
        self.k_proj = nn.Linear(d_model, d_head, bias=False)
        self.v_proj = nn.Linear(d_model, d_head, bias=False)
        self.out_proj = nn.Linear(d_head, d_model, bias=False)
        
        # RoPE frequencies
        # d_head/2 frecuencias: θ_k = base^(-2k/d_head)
        self.rope_frequencies = self._compute_rope_frequencies()
    
    def _compute_rope_frequencies(self):
        """Computa frecuencias RoPE."""
        freqs = []
        for k in range(self.d_head // 2):
            theta = 10000.0 ** (-2.0 * k / self.d_head)
            freqs.append(theta)
        return torch.tensor(freqs)
    
    def _apply_rope(self, x, positions):
        """Aplica RoPE a las activaciones."""
        # x: (batch, seq_len, d_head)
        # positions: (seq_len,) índices de posición
        
        # Para cada par (x_2k, x_(2k+1)):
        # RoPE(x, pos)_2k = x_2k * cos(pos*θ_k) - x_(2k+1) * sin(pos*θ_k)
        # RoPE(x, pos)_(2k+1) = x_2k * sin(pos*θ_k) + x_(2k+1) * cos(pos*θ_k)
        
        cos, sin = self._compute_rope_cache(positions)
        x_rotated = ...
        return x_rotated
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
        
        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: (batch, seq_len, seq_len)
        """
        batch, seq_len, _ = x.shape
        
        # Proyecciones
        q = self.q_proj(x)  # (batch, seq_len, d_head)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # RoPE
        positions = torch.arange(seq_len, device=x.device)
        q = self._apply_rope(q, positions)
        k = self._apply_rope(k, positions)
        
        # Scaled dot-product attention
        # L(q_i, k_j) = (R_j k_j)^T (R_i q_i) = k_j^T R^(i-j) q_i
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        
        # Causal mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # Weighted sum
        output = torch.matmul(attention_weights, v)
        
        # Output projection
        output = self.out_proj(output)
        
        return output, attention_weights
```

**TransformerBlock:**

```python
class TransformerBlock(nn.Module):
    def __init__(self, config):
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attention = SingleHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout_rate),
        )
    
    def forward(self, x):
        # Pre-norm + attention + residual
        attn_out, attn_weights = self.attention(self.ln1(x))
        x = x + attn_out
        
        # Pre-norm + MLP + residual
        x = x + self.mlp(self.ln2(x))
        
        return x, attn_weights
```

---

## 3. Entrenamiento

### 3.1 train.py

**Propósito:** Loop de entrenamiento con last-token prediction.

```python
def train(model, dataloader, optimizer, num_epochs):
    """
    Entrenamiento con last-token prediction.
    
    La pérdida solo se calcula en el último token de cada secuencia.
    Los tokens anteriores tienen ignore_index = -100 (no contribuyen al gradiente).
    """
    for epoch in range(num_epochs):
        for batch in dataloader:
            input_ids, labels = batch
            # input_ids: (batch, seq_len)
            # labels: (batch, seq_len) con -100 en todos excepto último token
            
            logits, _ = model(input_ids)
            
            # Solo último token contribuye a la pérdida
            loss = F.cross_entropy(
                logits[:, -1, :],  # Solo último paso de tiempo
                labels[:, -1],      # Solo etiqueta del último token
                ignore_index=-100
            )
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
```

**Justificación:** El objetivo es que el modelo aprenda a predecir el token correcto al final de la secuencia (respuesta de la tarea multi-hop), ignorando los otros tokens. Esto fuerza al modelo a desarrollar mecanismos internos para resolver la tarea.

### 3.2 eval.py

**Propósito:** Evaluación en test set.

```python
def evaluate(model, test_loader):
    """
    Evalúa la precisión del modelo.
    - Accuracy global
    - Accuracy por hop (1, 2, 3, 4)
    """
    model.eval()
    correct = 0
    total = 0
    hop_correct = {1: 0, 2: 0, 3: 0, 4: 0}
    hop_total = {1: 0, 2: 0, 3: 0, 4: 0}
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids, labels, hop_counts = batch
            logits, _ = model(input_ids)
            
            # Predicción en el último token
            predictions = logits[:, -1, :].argmax(dim=-1)
            
            # Comparar con etiquetas
            for i, (pred, label, hops) in enumerate(
                zip(predictions, labels[:, -1], hop_counts)
            ):
                if pred == label:
                    correct += 1
                    hop_correct[hops.item()] += 1
                total += 1
                hop_total[hops.item()] += 1
    
    accuracy = correct / total
    hop_accuracies = {
        h: hop_correct[h] / hop_total[h] 
        for h in range(1, 5) if hop_total[h] > 0
    }
    
    return accuracy, hop_accuracies
```

---

## 4. Métricas: Positional y Symbolic Scores

### 4.1 permutation_utils.py

**Propósito:** Funciones para permutar secuencias y calcular pesos.

```python
def get_swap_permutation(i, j, n):
    """
    Crea una permutación que intercambia las posiciones i y j.
    
    Args:
        i, j: índices a intercambiar (0-indexed)
        n: longitud total de la secuencia
    
    Returns:
        perm: lista de índices permutados
    """
    perm = list(range(n))
    perm[i], perm[j] = perm[j], perm[i]
    return perm

def calculate_swap_weight(attention_weights, i, j, temperature=1.0):
    """
    Calcula el peso α(π) para el swap entre i y j.
    
    El peso depende de cuánta masa de atención se mueve en el swap.
    
    Args:
        attention_weights: distribución de atención en el query final
        i, j: posiciones a intercambiar
        temperature: temperatura del softmax (τ)
    
    Returns:
        weight: peso de esta permutación
    """
    di = attention_weights[i]
    dj = attention_weights[j]
    return F.softmax(torch.tensor([di - dj]) / temperature).item()
```

### 4.2 positional_symbolic_scores.py

**Propósito:** Implementación de las métricas positional y symbolic scores.

```python
def compute_positional_score(head_attention, hidden_states):
    """
    Calcula el positional score para un head en un input dado.
    
    El positional score mide cuán invariante es la atención
    ante permutaciones de los key vectors.
    
    Args:
        head_attention: distribución de atención del último token
        hidden_states: representaciones ocultas en cada posición
    
    Returns:
        pos_score: float entre 0 y 1
    """
    n = len(hidden_states)
    pos_score = 0.0
    
    for i in range(n - 1):
        for j in range(i + 1, n):
            # Atención original (query en n, keys en i, j)
            v_ij = (head_attention[i], head_attention[j])
            
            # Permutar estados ocultos en posiciones i, j
            hidden_permuted = hidden_states.clone()
            hidden_permuted[i], hidden_permuted[j] = \
                hidden_permuted[j], hidden_permuted[i]
            
            # Re-evaluar atención con estados permutados
            # (en la práctica se estima desde las activaciones)
            attention_permuted = reestimate_attention(
                hidden_permuted, head_attention
            )
            v_ij_perm = (attention_permuted[i], attention_permuted[j])
            
            # Peso de la permutación
            alpha = compute_swap_weight(head_attention, i, j)
            
            # Similitud coseno entre atención original y permutada
            cos_sim = cosine_similarity(v_ij, v_ij_perm)
            pos_score += alpha * cos_sim
    
    return pos_score / total_weight

def compute_symbolic_score(head_attention, hidden_states):
    """
    Calcula el symbolic score para un head en un input dado.
    
    El symbolic score mide cuán equivariante es la atención
    ante permutaciones de los key vectors.
    """
    n = len(hidden_states)
    sym_score = 0.0
    
    for i in range(n - 1):
        for j in range(i + 1, n):
            v_ij = (head_attention[i], head_attention[j])
            
            # Bajo equivariancia: v_ij(π(x)) = v_ji(x)
            # La atención en i, j después de permutar = atención original en j, i
            v_ji = (head_attention[j], head_attention[i])
            
            alpha = compute_swap_weight(head_attention, i, j)
            cos_sim = cosine_similarity(v_ij, v_ji)
            sym_score += alpha * cos_sim
    
    return sym_score / total_weight

def compute_scores_for_model(model, dataset):
    """
    Calcula positional y symbolic scores para todos los heads
    en todo el dataset.
    
    Para cada capa:
    - Extraer la atención del último token
    - Calcular positional score promedio sobre todas las secuencias
    - Calcular symbolic score promedio sobre todas las secuencias
    
    Returns:
        pos_scores: list[float] por capa
        sym_scores: list[float] por capa
        entropy: list[float] entropía de atención por capa
    """
    pos_scores = []
    sym_scores = []
    entropy = []
    
    for layer_idx in range(model.config.layers):
        layer_pos = []
        layer_sym = []
        layer_entropy = []
        
        for sequence in dataset:
            # Forward pass hasta esta capa
            _, attentions = model.forward_until(sequence, layer_idx)
            head_attn = attentions[layer_idx][0]  # single head
            
            # Último token como query
            last_query_attn = head_attn[-1, :]  # atención desde último token
            
            # Scores
            pos = compute_positional_score(last_query_attn, 
                                           model.get_hidden_states(sequence, layer_idx))
            sym = compute_symbolic_score(last_query_attn,
                                          model.get_hidden_states(sequence, layer_idx))
            
            # Entropía normalizada
            ent = compute_entropy(last_query_attn)
            
            layer_pos.append(pos)
            layer_sym.append(sym)
            layer_entropy.append(ent)
        
        pos_scores.append(np.mean(layer_pos))
        sym_scores.append(np.mean(layer_sym))
        entropy.append(np.mean(layer_entropy))
    
    return pos_scores, sym_scores, entropy
```

### 4.3 entropia.py

**Propósito:** Cálculo de entropía de la distribución de atención.

```python
def compute_entropy(attention_weights):
    """
    Calcula la entropía normalizada de la distribución de atención.
    
    Entropía = -Σ p_i * log(p_i) / log(n)
    Normalizada: 0 = determinista, 1 = uniforme
    
    Args:
        attention_weights: distribución de atención (último token)
    
    Returns:
        normalized_entropy: float entre 0 y 1
    """
    # Entropía de Shannon
    entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-10))
    
    # Normalizar por log(n)
    n = attention_weights.shape[-1]
    normalized = entropy / math.log(n)
    
    return normalized.item()
```

---

## 5. Análisis

### 5.1 head_purity.py

**Propósito:** Análisis de pureza de heads (cuándo un head se vuelve claramente posicional o simbólico).

```python
def classify_head_purity(pos_score, sym_score, gamma=0.1):
    """
    Clasifica un head como puro o mixto.
    
    Args:
        pos_score: positional score (0-1)
        sym_score: symbolic score (0-1)
        gamma: umbral de tolerancia
    
    Returns:
        'positional': si pos_score > 1-γ y sym_score < γ
        'symbolic': si sym_score > 1-γ y pos_score < γ
        'mixed': en otro caso
    """
    if pos_score > (1 - gamma) and sym_score < gamma:
        return 'positional'
    elif sym_score > (1 - gamma) and pos_score < gamma:
        return 'symbolic'
    else:
        return 'mixed'

def compute_purity_dynamics(model, dataset, checkpoints, gamma=0.1):
    """
    Computa la dinámica de pureza a través del entrenamiento.
    
    Para cada checkpoint, cuenta:
    - Número de heads posicionales puros
    - Número de heads simbólicos puros
    
    Returns:
        timeline: {step: {'pos': count, 'sym': count}}
    """
    timeline = {}
    
    for step, checkpoint in enumerate(checkpoints):
        model.load_state_dict(checkpoint)
        pos_scores, sym_scores, _ = compute_scores_for_model(model, dataset)
        
        pos_count = 0
        sym_count = 0
        for pos, sym in zip(pos_scores, sym_scores):
            classification = classify_head_purity(pos, sym, gamma)
            if classification == 'positional':
                pos_count += 1
            elif classification == 'symbolic':
                sym_count += 1
        
        timeline[step] = {'pos': pos_count, 'sym': sym_count}
    
    return timeline
```

### 5.2 frequency_analysis.py

**Propósito:** Análisis de uso de frecuencias RoPE por capa.

```python
def analyze_frequency_use(model):
    """
    Analiza qué frecuencias RoPE usa cada capa.
    
    Para cada head, identifica la frecuencia dominante
    (la que más contribuye a la distribución de atención).
    
    Returns:
        frequency_map: {layer_idx: dominant_freq_id}
    """
    frequency_map = {}
    
    for layer_idx, block in enumerate(model.blocks):
        attn = block.attention
        
        # Obtener los pesos de Q y K
        W_q = attn.q_proj.weight
        W_k = attn.k_proj.weight
        
        # Para cada frecuencia RoPE (d_head/2 frecuencias)
        # Calcular qué frecuencia tiene mayor magnitud en Q·K^T
        freq_contributions = []
        for freq_idx in range(model.config.d_head // 2):
            # Extraer los componentes 2D para esta frecuencia
            q_freq = W_q[:, 2*freq_idx:2*freq_idx+2]
            k_freq = W_k[:, 2*freq_idx:2*freq_idx+2]
            
            # Norma de Frobenius como proxy de contribución
            contribution = torch.norm(q_freq @ k_freq.T)
            freq_contributions.append(contribution.item())
        
        dominant_freq = np.argmax(freq_contributions)
        frequency_map[layer_idx] = dominant_freq
    
    return frequency_map
```

### 5.3 discrepancy.py

**Propósito:** Cálculo de la discrepancia teórica para los mecanismos Index y Retrieval.

```python
def index_discrepancy_upper_bound(n, theta):
    """
    Cota superior de discrepancia para Index (Theorem 3.2).
    
    Δ_Index(n) ≤ min{1 - cos(θ), π²/(2n²)}
    
    Args:
        n: longitud de la secuencia
        theta: ángulo RoPE usado
    
    Returns:
        bound: float, cota superior
    """
    bound1 = 1 - math.cos(theta)
    bound2 = math.pi**2 / (2 * n**2)
    return min(bound1, bound2)

def retrieval_discrepancy_upper_bound(n, theta, omega, alphabet_size):
    """
    Cota superior de discrepancia para Retrieval (Theorem 3.2).
    
    Δ_Retrieval(n) ≤ 1 - cos(ω - n·θ)
    donde ω = 2π/|ALPH|
    
    Args:
        n: longitud de la secuencia
        theta: ángulo RoPE
        omega: ángulo de codificación de símbolos
        alphabet_size: |ALPH|
    
    Returns:
        bound: float, cota superior
    """
    omega = 2 * math.pi / alphabet_size
    bound = 1 - math.cos(omega - n * theta)
    return bound

def compute_theoretical_discrepancies(max_len=300):
    """
    Computa las discrepancias teóricas para el rango de longitudes.
    
    Returns:
        index_bounds: list[float]
        retrieval_bounds: list[float]
    """
    # Parámetros del modelo entrenado (Tabla 1 del paper)
    # Layer 2 en Number Task: θ = θ₂ = 0.8 (freq_id=2, freq=0.8660)
    # Layer 2 en Letter Task: θ = θ₄₂ = 0.0027 (freq_id=42, freq=0.0027)
    theta_index = 0.8      # Ángulo para Index
    theta_retrieval = 0.0027  # Ángulo para Retrieval
    alphabet_size = 8      # |ALPH| = 8 para Letter Task
    
    index_bounds = []
    retrieval_bounds = []
    
    for n in range(1, max_len + 1):
        idx_bound = index_discrepancy_upper_bound(n, theta_index)
        ret_bound = retrieval_discrepancy_upper_bound(
            n, theta_retrieval, None, alphabet_size
        )
        index_bounds.append(idx_bound)
        retrieval_bounds.append(ret_bound)
    
    return index_bounds, retrieval_bounds
```

---

## 6. Construcciones Teóricas

### 6.1 index_construction.py

**Propósito:** Implementación de la construcción teórica de Index (Theorem 3.1).

```python
def construct_index_transformer(v, theta, alphabet, integers):
    """
    Construye un Transformer T_Index que implementa Index.
    
    Arquitectura:
    - Embedding: one-hot + bias column
    - W_Q: query vectors = v para letras, ρ(θ)^(-σ)v para números
    - W_K: key vectors = v para todos
    - W_V: diagonal, α > 1 para coordenadas de letras, 0 para otras
    - RoPE: un solo ángulo θ
    
    Args:
        v: vector 2D no nulo
        theta: ángulo RoPE
        alphabet: conjunto de letras
        integers: conjunto de enteros
    
    Returns:
        T_Index: transformer construido
    """
    d = len(alphabet) + len(integers) + 1  # +1 para bias
    
    # W_Q ∈ ℝ^{2×d}
    W_Q = torch.zeros(2, d)
    for sigma in alphabet:
        # Para letras: W_Q·E(σ) = v
        W_Q[:, sigma_idx] = v
    for sigma in integers:
        # Para números: W_Q·E(σ) = ρ(θ)^(-σ)·v
        rotation = rotation_matrix(-sigma * theta)
        W_Q[:, sigma_idx] = rotation @ v
    
    # W_K ∈ ℝ^{2×d}
    W_K = torch.zeros(2, d)
    for sigma in alphabet + integers:
        W_K[:, sigma_idx] = v
    
    # W_V ∈ ℝ^{d×d}, diagonal
    W_V = torch.zeros(d, d)
    for sigma in alphabet:
        # α > 1 para coordenadas de letras
        W_V[sigma_idx, sigma_idx] = 2.0
    
    return T_Index(W_Q, W_K, W_V, theta)

def index_attention_logit(query_token, key_token, query_pos, key_pos, theta):
    """
    Logit de atención para Index.
    
    Si el query es letra:
        L = cos((n - j)θ)
    Si el query es número:
        L = cos((n - j - σ_n)θ)
    
    Donde σ_n es el valor del número en la posición query.
    """
    if is_letter(query_token):
        return math.cos((query_pos - key_pos) * theta)
    else:  # es número
        offset = int(query_token)
        return math.cos((query_pos - key_pos - offset) * theta)

def index_max_attended_position(query_token, query_pos, theta):
    """
    Determina la posición de atención máxima para Index.
    
    Si query es letra: atiende a sí mismo (posición query_pos)
    Si query es número con valor σ: atiende a posición (query_pos - σ)
    """
    if is_letter(query_token):
        return query_pos
    else:
        offset = int(query_token)
        return query_pos - offset
```

### 6.2 retrieval_construction.py

**Propósito:** Implementación de la construcción teórica de Retrieval.

```python
def construct_retrieval_transformer(v, omega, alphabet, integers):
    """
    Construye un Transformer T_Retrieval que implementa Retrieval.
    
    Arquitectura:
    - Embedding: one-hot sobre pares (letra_izq, letra_der/o número)
    - W_Q: query vectors codifican la letra derecha (o izquierda si es target)
    - W_K: key vectors codifican la letra izquierda
    - W_V: identity (copia el valor)
    - RoPE: ángulo θ, con 0 < θ < ω/2n₀
    
    Args:
        v: vector 2D
        omega: ángulo = 2π/|ALPH|
        alphabet: conjunto de letras
        integers: conjunto de enteros
    
    Returns:
        T_Retrieval: transformer construido
    """
    d = 2 * len(alphabet) + len(integers)
    
    # Para un par (L, R):
    # W_Q·E(L,R) = ρ(ω)^(-ϕ(R))·v si L,R ∈ ALPH (pre-target)
    # W_Q·E(L,R) = ρ(ω)^(-ϕ(L))·v si L ∈ ALPH, R ∈ INT (target)
    # W_K·E(L,R) = ρ(ω)^(ϕ(L))·v siempre
    
    W_Q = torch.zeros(2, d)
    W_K = torch.zeros(2, d)
    # ... (construcción similar a Index pero con codificación simbólica)
    
    W_V = torch.eye(d)  # Identity
    
    return T_Retrieval(W_Q, W_K, W_V, omega)

def retrieval_attention_logit(query, key, query_pos, key_pos, theta, omega):
    """
    Logit de atención para Retrieval.
    
    L = cos((ϕ(L_key) - ϕ(R_query))·ω + (n - j)·θ)
    
    donde ϕ mapea letras a índices numéricos.
    """
    L_key = get_left_letter(key)
    R_query = get_right_letter(query)
    
    phi_L = letter_to_index(L_key)
    phi_R = letter_to_index(R_query)
    
    angle = (phi_L - phi_R) * omega + (query_pos - key_pos) * theta
    return math.cos(angle)
```

---

## 7. Visualizaciones

### 7.1 learning_dynamics.py

**Propósito:** Generar las visualizaciones de la Figura 2.

```python
def plot_learning_dynamics(accuracy_history, pos_scores_history, 
                           sym_scores_history, entropy_history, 
                           purity_history):
    """
    Genera la Figura 2 del paper.
    
    Panel A: Task accuracy por hop y global
    Panel B: Positional y symbolic scores por capa
    Panel C: Head purity counts
    """
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    
    # A: Accuracy
    for hop in [1, 2, 3, 4]:
        axes[0, 0].plot(accuracy_history['hop_{}'.format(hop)], 
                       label='Hop {}'.format(hop))
    axes[0, 0].plot(accuracy_history['global'], 'm-', linewidth=2, 
                   label='Global')
    
    # B: Scores por capa
    for layer, pos, sym, ent in zip(
        range(len(pos_scores_history[0])),
        pos_scores_history, sym_scores_history, entropy_history
    ):
        axes[1, 0].plot(pos, 'r-', label='Pos Layer {}'.format(layer))
        axes[1, 0].plot(sym, 'g-', label='Sym Layer {}'.format(layer))
        axes[1, 0].plot(ent, 'm--', label='Entropy')
    
    # C: Purity
    axes[2, 0].plot(purity_history['pos'], 'r-', label='Pure positional')
    axes[2, 0].plot(purity_history['sym'], 'g-', label='Pure symbolic')
    
    plt.tight_layout()
    plt.savefig('fig2_learning_dynamics.png', dpi=300)
```

### 7.2 attention_patterns.py

**Propósito:** Visualizar patrones de atención (Figura 3).

```python
def plot_attention_patterns(model, sequence, task_type):
    """
    Visualiza los patrones de atención para todas las capas.
    
    Muestra:
    - Para cada capa: quién atiende a quién
    - Las posiciones de los tokens
    - Identificación de Index, Retrieval, Reflexive
    """
    _, attentions = model(sequence)
    
    fig, axes = plt.subplots(4, 3, figsize=(15, 20))
    
    for layer_idx, attn in enumerate(attentions):
        ax = axes[layer_idx // 3, layer_idx % 3]
        
        # Matriz de atención (causal)
        attn_matrix = attn[0].detach().numpy()  # single head
        
        im = ax.imshow(attn_matrix, cmap='viridis', aspect='auto')
        ax.set_title('Layer {}'.format(layer_idx))
        
        # Anotar nombres de tokens
        ax.set_xticks(range(len(sequence)))
        ax.set_yticks(range(len(sequence)))
        ax.set_xticklabels(sequence, rotation=90, fontsize=6)
        ax.set_yticklabels(sequence, fontsize=6)
    
    plt.tight_layout()
    plt.savefig('fig3_attention_patterns.png', dpi=300)
```

### 7.3 geometric_analysis.py

**Propósito:** Visualización geométrica de mecanismos RoPE (Figura 4).

```python
def plot_geometric_analysis(model, sequence):
    """
    Visualiza la geometría de los vectores query y key en el
    plano rotacional de RoPE.
    
    Panel A: Construcción teórica de Index
    Panel B: Vectores del modelo entrenado
    Panel C: Frecuencia dominante RoPE
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # A: Construcción teórica
    # Key vectors todos alineados
    # Query vectors rotados según posición
    
    # B: Modelo entrenado
    # Proyectar Q y K al plano de la frecuencia dominante
    dominant_freq, q_proj, k_proj = extract_rope_projection(model, sequence)
    
    axes[0].scatter(q_proj[:, 0], q_proj[:, 1], c='blue', label='Queries')
    axes[0].scatter(k_proj[:, 0], k_proj[:, 1], c='red', label='Keys')
    
    # C: Max logit por frecuencia
    freq_contributions = compute_frequency_contributions(model, sequence)
    axes[2].bar(range(len(freq_contributions)), freq_contributions)
    axes[2].axvline(dominant_freq, color='r', linestyle='--')
    
    plt.tight_layout()
    plt.savefig('fig4_geometric_analysis.png', dpi=300)
```

### 7.4 generalization_plots.py

**Propósito:** Visualización de resultados de generalización (Figura 5).

```python
def plot_generalization_results(index_discrepancy, retrieval_discrepancy,
                                model_accuracy_num, model_accuracy_let,
                                llm_results):
    """
    Genera la Figura 5 del paper.
    
    Panel A: Cotas teóricas de discrepancia
    Panel B: Precisión del modelo entrenado
    Panel C: Precisión de GPT 5.4 y 5.5
    Panel D: Precisión de Claude Sonnet 3.7
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # A: Discrepancia teórica
    lengths = range(1, 101)
    axes[0, 0].plot(lengths, index_discrepancy, 'r-', label='Index')
    axes[0, 0].plot(lengths, retrieval_discrepancy, 'g-', label='Retrieval')
    
    # B: Modelo entrenado
    axes[0, 1].plot(model_accuracy_num, 'r-', label='Number task')
    axes[0, 1].plot(model_accuracy_let, 'g-', label='Letter task')
    
    # C: LLMs
    for model_name, results in llm_results.items():
        axes[1, 0].errorbar(results['lengths'], results['num_means'],
                          yerr=results['num_stds'], fmt='o-', 
                          label='{} Number'.format(model_name))
        axes[1, 1].errorbar(results['lengths'], results['let_means'],
                          yerr=results['let_stds'], fmt='s-',
                          label='{} Letter'.format(model_name))
    
    plt.tight_layout()
    plt.savefig('fig5_generalization.png', dpi=300)
```

---

## 8. Evaluación en LLMs

### 8.1 prompt_templates.py

**Propósito:** Templates de prompts para evaluar LLMs en los tasks 1-hop.

```python
NUMBER_TASK_PROMPT = """You will be given a sequence of space-separated tokens. 
Your task is to find the target token by following these rules:
1. Identify the very last token in the sequence, which will always be a number 'n'.
2. Move 'n' positions to the left from that final token.
3. The token you land on is the target token.
Output just the target token and absolutely nothing else. Do not include any explanations or formatting.

Example 1:
Input: "a z b y c x d w 5"
Output: y

Example 2:
Input: "v t k g n k o h m p 4"
Output: o

Example 3:
Input: "w y b c v t r i p p 10"
Output: w

Task:
Input: "{sequence}"
Output:"""

LETTER_TASK_PROMPT = """You will be given a sequence of space-separated tokens. 
Your task is to find the target token by following these rules:
1. Start with the very last token in the sequence.
2. Identify the second character of that token.
3. Scan backwards (to the left) to find the token that starts with that exact character; there is only one token that fulfills this condition.
4. This is your target token.
Output just the target token and absolutely nothing else. Do not include any explanations or formatting.

Example:
Input: "a4 b3 c2 d1 fc"
Output: c2

Input: "w9 t7 l5 g1 n4 l6 u9 k7 m5 p8 gk"
Output: k7

Input: "q5 r7 x4 t7 f4 k7 q4 u2 u3 r5 ex"
Output: x4

Task:
Input: "{sequence}"
Output:"""
```

### 8.2 run_llm_tests.py

**Propósito:** Ejecutar evaluación en LLMs (GPT 5.4, GPT 5.5, Claude Sonnet 3.7).

```python
def evaluate_llm_on_task(llm_client, task_type, sequence, prompt_template):
    """
    Evalúa un LLM en una instancia de la tarea.
    
    Args:
        llm_client: API client del LLM
        task_type: 'number' o 'letter'
        sequence: secuencia de tokens
        prompt_template: template de prompt
    
    Returns:
        answer: str, respuesta del modelo
        correct: bool, si la respuesta es correcta
    """
    prompt = prompt_template.format(sequence=sequence)
    
    response = llm_client.complete(
        prompt=prompt,
        max_tokens=1,  # Forzar respuesta de un token
        temperature=0.0,
        stop=None
    )
    
    answer = response.strip()
    correct_answer = compute_correct_answer(sequence, task_type)
    
    return answer, answer == correct_answer

def run_generalization_experiment(llm_clients, task_types, sequence_lengths,
                                  n_runs=5):
    """
    Ejecuta el experimento completo de generalización.
    
    Para cada modelo, tarea y longitud:
    - Generar n_runs datasets diferentes
    - Evaluar precisión
    - Reportar media y desviación estándar
    """
    results = {}
    
    for model_name, client in llm_clients.items():
        model_results = {'number': {}, 'letter': {}}
        
        for task_type in ['number', 'letter']:
            for length in sequence_lengths:
                accuracies = []
                
                for run in range(n_runs):
                    # Generar dataset para esta longitud
                    dataset = generate_dataset(task_type, length, seed=run)
                    
                    correct = 0
                    for sequence in dataset:
                        answer, is_correct = evaluate_llm_on_task(
                            client, task_type, sequence,
                            get_prompt_template(task_type)
                        )
                        if is_correct:
                            correct += 1
                    
                    accuracy = correct / len(dataset)
                    accuracies.append(accuracy)
                
                mean_acc = np.mean(accuracies)
                std_acc = np.std(accuracies)
                model_results[task_type][length] = (mean_acc, std_acc)
        
        results[model_name] = model_results
    
    return results
```

---

## 9. Flujo de Ejecución Completo

El pipeline completo del paper se ejecuta en el siguiente orden:

```bash
# 1. Generar datasets
python dataset/number_task.py --output data/number_task/
python dataset/letter_task.py --output data/letter_task/

# 2. Entrenar modelos
python training/train.py --config configs/number_task.yaml --output models/number_model/
python training/train.py --config configs/letter_task.yaml --output models/letter_model/

# 3. Calcular métricas (positional/symbolic scores)
python metrics/positional_symbolic_scores.py --model models/number_model/ --dataset data/number_task/
python metrics/positional_symbolic_scores.py --model models/letter_model/ --dataset data/letter_task/

# 4. Análisis de pureza de heads
python analysis/head_purity.py --scores results/scores_number.pkl
python analysis/head_purity.py --scores results/scores_letter.pkl

# 5. Análisis de frecuencias RoPE
python analysis/frequency_analysis.py --model models/number_model/
python analysis/frequency_analysis.py --model models/letter_model/

# 6. Pruebas de generalización
python analysis/generalization.py --model models/number_model/ --dataset data/number_task/
python analysis/generalization.py --model models/letter_model/ --dataset data/letter_task/

# 7. Cálculo de discrepancia teórica
python analysis/discrepancy.py --theta 0.8 --omega 0.785 --alphabet-size 8

# 8. Evaluación en LLMs
python llm_evaluation/run_llm_tests.py --models gpt-5.4,gpt-5.5,claude-sonnet-3.7

# 9. Generar visualizaciones
python visualization/learning_dynamics.py
python visualization/attention_patterns.py
python visualization/geometric_analysis.py
python visualization/generalization_plots.py
```

---

## 10. Dependencias (requirements.txt)

```
# Core
torch>=2.0.0
jax>=0.4.0           # Para GPT-J (mesh-transformer-jax)
flax>=0.7.0

# Data
numpy>=1.24.0
pandas>=2.0.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Analysis
scipy>=1.10.0
scikit-learn>=1.2.0

# LLM evaluation
openai>=1.0.0
anthropic>=0.30.0

# Utilities
tqdm>=4.65.0
pyyaml>=6.0
wandb>=0.15.0         # Logging (opcional)
```

---

## 11. Notas sobre la Reconstrucción

El código descrito en este informe es una **reconstrucción detallada** basada en la descripción del paper. El repositorio original en `anonymous.4open.science/r/number-and-letter-neurips26-C754` no fue accesible directamente debido a restricciones de red, pero el paper proporciona información suficientemente detallada para reconstruir la implementación completa:

1. **Arquitectura del modelo** (Appendix C): descripción completa del GPT-J con single-head.
2. **Generación de datos** (Appendix B): definiciones formales y detalles de implementación.
3. **Métricas** (Appendix D): definiciones formales de positional y symbolic scores.
4. **Construcciones teóricas** (Appendix F): pseudo-código y demostraciones completas.
5. **Evaluación en LLMs** (Appendix H): prompts utilizados.
6. **Configuración experimental** (Appendix I): recursos de cómputo.
