"""
Positional y Symbolic Scores para clasificar attention heads.

Implementación basada en Urrutia et al. (2025) y Appendix D del paper.

Definiciones:
- Un head es POSICIONAL si su atención es INVARIANTE ante permutaciones
  de los key vectors (atiende a posiciones, no a símbolos).
- Un head es SIMBÓLICO si su atención es EQUIVARIANTE ante permutaciones
  de los key vectors (atiende a símbolos, no a posiciones).

Los scores se calculan como promedios de similitud coseno sobre
permutaciones por pares (swaps), ponderadas por la masa de atención
que cada swap mueve.
"""

import math
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


def compute_attention_entropy(attention_weights):
    """
    Calcula la entropía normalizada de la distribución de atención.
    
    Entropía de Shannon normalizada:
        H = -Σ p_i · log(p_i) / log(n)
    
    Rango: [0, 1]
    0 = distribución determinista (un token domina)
    1 = distribución uniforme (todos los tokens igualmente atendidos)
    
    Args:
        attention_weights: (batch, seq_len) distribución de atención
    
    Returns:
        entropy: float
    """
    # Pequeño épsilon para evitar log(0)
    eps = 1e-10
    probs = attention_weights + eps
    entropy = -torch.sum(probs * torch.log(probs))
    n = attention_weights.shape[-1]
    normalized = entropy / math.log(n)
    return normalized.item()


def get_swap_permutation(i, j, n):
    """
    Crea una permutación que intercambia las posiciones i, j.
    """
    perm = list(range(n))
    perm[i], perm[j] = perm[j], perm[i]
    return perm


def compute_swap_weight(attention_probs, i, j, temperature=1.0):
    """
    Calcula el peso α(π) para el swap entre posiciones i, j.
    
    El peso depende de cuánta masa de atención se mueve:
        α(π) = softmax((d_i - d_j) / τ)
    
    donde d_i es la atención en la posición i y τ es la temperatura.
    
    Args:
        attention_probs: distribución de atención (seq_len,)
        i, j: índices a intercambiar
        temperature: temperatura τ (default: 1.0)
    
    Returns:
        weight: float
    """
    di = attention_probs[i].item()
    dj = attention_probs[j].item()
    
    # Softmax sobre la diferencia
    logits = torch.tensor([di - dj]) / temperature
    weight = F.softmax(logits, dim=0)[0].item()
    
    return weight


def compute_positional_score_for_sequence(
    attention_last_token, hidden_states, temperature=1.0
):
    """
    Calcula el positional score para una secuencia.
    
    El positional score mide cuán INVARIANTE es la atención del último
    token ante permutaciones de los key vectors.
    
    Para cada par (i,j):
        1. Atención original: v_ij = (a_i, a_j)
        2. Permutar hidden states en i, j
        3. Re-estimar atención: v_ij(π(x))
        4. positional contribución = cos_sim(v_ij(π(x)), v_ij)
    
    Score = Σ α(π) · cos_sim(π) / Σ α(π)
    
    Args:
        attention_last_token: (seq_len,) atención desde el último token
        hidden_states: (seq_len, d_model) representaciones ocultas
        temperature: temperatura para pesos
    
    Returns:
        pos_score: float entre 0 y 1
    """
    n = len(attention_last_token)
    total_weight = 0.0
    weighted_sum = 0.0
    
    for i in range(n - 1):
        for j in range(i + 1, n):
            # Atención original en posiciones i, j
            v_ij = torch.stack([
                attention_last_token[i], attention_last_token[j]
            ])
            
            # Peso de esta permutación
            alpha = compute_swap_weight(
                attention_last_token, i, j, temperature
            )
            
            # Nota: en la práctica, para obtener la atención con
            # estados permutados, se necesita re-ejecutar el forward
            # pass del modelo. Esto es computacionalmente costoso.
            # 
            # El paper de Urrutia et al. (2025) usa una aproximación
            # basada en las activaciones originales y los logits.
            # 
            # Aquí presentamos la versión conceptual; la implementación
            # exacta requiere acceso a las activaciones intermedias.
            attention_permuted = _reestimate_attention(
                hidden_states, i, j
            )
            v_ij_perm = torch.stack([
                attention_permuted[i], attention_permuted[j]
            ])
            
            # Similitud coseno
            cos_sim = F.cosine_similarity(
                v_ij.unsqueeze(0), v_ij_perm.unsqueeze(0)
            ).item()
            
            weighted_sum += alpha * cos_sim
            total_weight += alpha
    
    if total_weight > 0:
        return weighted_sum / total_weight
    return 0.0


def compute_symbolic_score_for_sequence(
    attention_last_token, hidden_states, temperature=1.0
):
    """
    Calcula el symbolic score para una secuencia.
    
    El symbolic score mide cuán EQUIVARIANTE es la atención del último
    token ante permutaciones de los key vectors.
    
    Bajo equivariancia: v_ij(π(x)) = v_ji(x)
    donde v_ji = (a_j, a_i) intercambiados.
    
    Args:
        attention_last_token: (seq_len,) atención desde el último token
        hidden_states: (seq_len, d_model)
        temperature: temperatura para pesos
    
    Returns:
        sym_score: float entre 0 y 1
    """
    n = len(attention_last_token)
    total_weight = 0.0
    weighted_sum = 0.0
    
    for i in range(n - 1):
        for j in range(i + 1, n):
            # Atención original en (i, j)
            v_ij = torch.stack([
                attention_last_token[i], attention_last_token[j]
            ])
            
            # Bajo equivariancia: después de permutar i↔j,
            # la atención debe ser v_ji = (a_j, a_i)
            v_ji = torch.stack([
                attention_last_token[j], attention_last_token[i]
            ])
            
            alpha = compute_swap_weight(
                attention_last_token, i, j, temperature
            )
            
            cos_sim = F.cosine_similarity(
                v_ij.unsqueeze(0), v_ji.unsqueeze(0)
            ).item()
            
            weighted_sum += alpha * cos_sim
            total_weight += alpha
    
    if total_weight > 0:
        return weighted_sum / total_weight
    return 0.0


def _reestimate_attention(hidden_states, i, j):
    """
    Re-estima la atención después de permutar posiciones i, j.
    
    En la implementación real, esto requeriría:
    1. Permutar los hidden states en posiciones i, j
    2. Re-ejecutar el forward de la capa de atención
    3. Obtener la nueva distribución
    
    Aquí usamos una aproximación basada en los logits originales.
    """
    # Versión simplificada para ilustración
    # La implementación real requiere acceso al modelo
    return hidden_states  # placeholder


def compute_scores_for_model(model, dataloader, device='cpu'):
    """
    Calcula positional y symbolic scores para todas las capas
    sobre un dataset completo.
    
    Args:
        model: GPTJModel
        dataloader: DataLoader con el dataset
        device: dispositivo
    
    Returns:
        pos_scores: list[float] — promedio por capa
        sym_scores: list[float] — promedio por capa
        entropy: list[float] — entropía promedio por capa
    """
    model.eval()
    n_layers = model.config['layers']
    
    layer_pos = [[] for _ in range(n_layers)]
    layer_sym = [[] for _ in range(n_layers)]
    layer_entropy = [[] for _ in range(n_layers)]
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing scores"):
            input_ids = batch['input_ids'].to(device)
            
            # Forward pass completo
            _, attention_maps = model(input_ids)
            
            # Para cada capa
            for layer_idx in range(n_layers):
                attn = attention_maps[layer_idx]  # (batch, 1, seq_len, seq_len)
                
                for b in range(input_ids.shape[0]):
                    # Atención del último token
                    last_attn = attn[b, 0, -1, :]  # (seq_len,)
                    
                    # Hidden states en esta capa
                    hidden = model.get_hidden_states(
                        input_ids[b:b+1], layer_idx
                    )[0]  # (seq_len, d_model)
                    
                    # Scores
                    pos = compute_positional_score_for_sequence(
                        last_attn, hidden
                    )
                    sym = compute_symbolic_score_for_sequence(
                        last_attn, hidden
                    )
                    ent = compute_attention_entropy(last_attn)
                    
                    layer_pos[layer_idx].append(pos)
                    layer_sym[layer_idx].append(sym)
                    layer_entropy[layer_idx].append(ent)
    
    # Promediar
    pos_scores = [np.mean(l) for l in layer_pos]
    sym_scores = [np.mean(l) for l in layer_sym]
    entropy = [np.mean(l) for l in layer_entropy]
    
    return pos_scores, sym_scores, entropy


# =============================================================================
# HEAD PURITY ANALYSIS
# =============================================================================

def classify_head_purity(pos_score, sym_score, gamma=0.1):
    """
    Clasifica un head como puro o mixto.
    
    Un head es:
    - 'positional': si pos_score > 1-γ y sym_score < γ
    - 'symbolic': si sym_score > 1-γ y pos_score < γ
    - 'mixed': en otro caso
    
    Args:
        pos_score: positional score (0-1)
        sym_score: symbolic score (0-1)
        gamma: umbral de tolerancia (default: 0.1)
    
    Returns:
        classification: str
    """
    if pos_score > (1 - gamma) and sym_score < gamma:
        return 'positional'
    elif sym_score > (1 - gamma) and pos_score < gamma:
        return 'symbolic'
    else:
        return 'mixed'


def compute_purity_over_time(model, model_checkpoints, dataset, gammas=[0.1, 0.05]):
    """
    Computa la evolución de la pureza de heads durante el entrenamiento.
    
    Para cada checkpoint, clasifica cada head y cuenta
    cuántos son posicionales puros y cuántos simbólicos puros.
    
    Args:
        model_checkpoints: list de estados del modelo
        dataset: dataset de evaluación
        gammas: lista de umbrales a probar
    
    Returns:
        purity_timeline: {
            step: {
                gamma: {
                    'positional': count,
                    'symbolic': count,
                    'mixed': count,
                    'layer_labels': [clases por capa]
                }
            }
        }
    """
    purity_timeline = {}
    
    for step, checkpoint in enumerate(model_checkpoints):
        model.load_state_dict(checkpoint)
        pos_scores, sym_scores, _ = compute_scores_for_model(
            model, dataset
        )
        
        step_results = {}
        for gamma in gammas:
            pos_count = 0
            sym_count = 0
            mixed_count = 0
            layer_labels = []
            
            for pos, sym in zip(pos_scores, sym_scores):
                label = classify_head_purity(pos, sym, gamma)
                layer_labels.append(label)
                if label == 'positional':
                    pos_count += 1
                elif label == 'symbolic':
                    sym_count += 1
                else:
                    mixed_count += 1
            
            step_results[gamma] = {
                'positional': pos_count,
                'symbolic': sym_count,
                'mixed': mixed_count,
                'layer_labels': layer_labels,
                'pos_scores': pos_scores,
                'sym_scores': sym_scores,
            }
        
        purity_timeline[step] = step_results
    
    return purity_timeline
