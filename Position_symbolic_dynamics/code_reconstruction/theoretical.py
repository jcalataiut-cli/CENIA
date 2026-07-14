"""
Construcciones teóricas y análisis de discrepancia.

Implementación de:
- Theorem 3.1: Construcción de T_Index y T_Retrieval
- Theorem 3.2: Cotas de discrepancia
- Remark 3.1: Separación posicional vs simbólico
"""

import math
import torch
import numpy as np


# =============================================================================
# ROTATION MATRIX
# =============================================================================

def rotation_matrix(theta):
    """
    Matriz de rotación 2D:
        ρ(θ) = [[cos(θ), -sin(θ)],
                [sin(θ),  cos(θ)]]
    """
    return torch.tensor([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta), math.cos(theta)]
    ])


# =============================================================================
# THEOREM 3.1: CONSTRUCCIÓN DE T_Index Y T_Retrieval
# =============================================================================

class TheoreticalIndexTransformer:
    """
    Transformer T_Index que implementa la función Selective Index.
    
    Construcción (Appendix F.2.1):
    - Embedding: one-hot + bias (última coordenada = 1)
    - W_Q ∈ ℝ²ˣᵈ: 
      * Para letras: W_Q·E(σ) = v
      * Para números: W_Q·E(σ) = ρ(θ)^(-σ)·v
    - W_K ∈ ℝ²ˣᵈ: W_K·E(σ) = v para todos los tokens
    - W_V ∈ ℝᵈˣᵈ: diagonal, α>1 para coordenadas de letras, 0 para el resto
    - RoPE: un solo ángulo θ
    - Residual stream: F(x, a) = x + a
    
    Mecanismo:
    - Si el query es una LETRA: atiende a sí mismo (posición n)
    - Si el query es un NÚMERO con valor σ: atiende a posición (n - σ)
    - Value projection copia la letra y pone 0 para números
    """
    
    def __init__(self, alphabet_size, integer_size, theta=0.1):
        """
        Args:
            alphabet_size: |ALPH| — número de letras
            integer_size: |INT| — número de enteros (max hop value)
            theta: ángulo RoPE
        """
        self.alphabet_size = alphabet_size
        self.integer_size = integer_size
        self.theta = theta
        
        # Dimensión de embedding: |ALPH| + |INT| + 1 (bias)
        self.d = alphabet_size + integer_size + 1
        
        # Vector v no nulo (2D)
        self.v = torch.tensor([0.0, 1.0])
        
        # Pre-computar matrices Q, K, V
        self._build_matrices()
    
    def _build_matrices(self):
        """Construye las matrices W_Q, W_K, W_V."""
        d = self.d
        v = self.v
        theta = self.theta
        n_letters = self.alphabet_size
        
        # W_Q ∈ ℝ²ˣᵈ
        W_Q = torch.zeros(2, d)
        
        # Para letras (coordenadas 0..n_letters-1): W_Q·E(σ) = v
        for idx in range(n_letters):
            W_Q[:, idx] = v
        
        # Para números (coordenadas n_letters..d-2): W_Q·E(σ) = ρ(θ)^(-σ)·v
        for idx in range(n_letters, d - 1):
            sigma = idx - n_letters + 1  # valor del entero (1, 2, ...)
            rot = rotation_matrix(-sigma * theta)
            W_Q[:, idx] = rot @ v
        
        # Última columna (bias): cero
        W_Q[:, -1] = torch.zeros(2)
        
        # W_K ∈ ℝ²ˣᵈ: todas las columnas = v
        W_K = v.unsqueeze(1).expand(2, d).contiguous()
        
        # W_V ∈ ℝᵈˣᵈ: diagonal, α=2 para letras, 0 para números y bias
        W_V = torch.zeros(d, d)
        for idx in range(n_letters):
            W_V[idx, idx] = 2.0  # α = 2
        
        self.W_Q = W_Q
        self.W_K = W_K
        self.W_V = W_V
    
    def compute_logit(self, query_token_is_letter, query_value, 
                      key_pos, query_pos):
        """
        Computa el logit de atención.
        
        Si query es letra:
            L = cos((n - j)θ)
        Si query es número con valor σ:
            L = cos((n - j - σ)θ)
        
        Args:
            query_token_is_letter: True si el token query es letra
            query_value: valor del token query (σ si es número)
            key_pos: posición del key
            query_pos: posición del query
        
        Returns:
            logit: float
        """
        diff = query_pos - key_pos
        if query_token_is_letter:
            angle = diff * self.theta
        else:
            angle = (diff - query_value) * self.theta
        
        return math.cos(angle)
    
    def forward(self, sequence):
        """
        Ejecuta T_Index sobre una secuencia.
        
        Args:
            sequence: list de tuplas (token_type, value)
              token_type: 'letter' o 'integer'
              value: el valor (string o int)
        
        Returns:
            output_sequence: secuencia de salida (aplicando Index)
        """
        n = len(sequence)
        output = list(sequence)
        
        for pos in range(n):
            token_type, value = sequence[pos]
            
            if token_type == 'integer':
                sigma = value
                target_pos = pos - sigma
                
                if target_pos >= 0:
                    target_type, target_val = sequence[target_pos]
                    if target_type == 'letter':
                        output[pos] = ('letter', target_val)
        
        return output
    
    @property
    def discrepancy_upper_bound(self):
        """
        Cota superior de discrepancia (Theorem 3.2):
            Δ_Index(n) ≤ min{1 - cos(θ), π²/(2n²)}
        """
        def bound_for_length(n):
            b1 = 1 - math.cos(self.theta)
            b2 = math.pi**2 / (2 * n**2)
            return min(b1, b2)
        return bound_for_length


class TheoreticalRetrievalTransformer:
    """
    Transformer T_Retrieval que implementa la función Retrieval.
    
    Construcción (Appendix F.2.2):
    - Embedding: one-hot sobre pares (L, R)
    - W_Q: codifica la letra derecha (R) o izquierda (L) según tipo
    - W_K: codifica la letra izquierda (L)
    - W_V: identity (copia el valor completo)
    - RoPE: ángulo θ, con 0 < θ < ω/2n₀
    - Residual stream: F(x, a) = a (attention-only)
    
    Mecanismo:
    - Para query (L_q, R_q) con L_q, R_q ∈ ALPH:
      * Máxima atención al key (L_k, R_k) donde L_k = R_q
      * La posición exacta no importa (simbólico)
    - Para query (L_q, N_q) con N_q ∈ INT:
      * Atiende a sí mismo (es el target a recuperar)
    """
    
    def __init__(self, alphabet_size, integer_size, 
                 theta=0.0027, omega=None):
        """
        Args:
            alphabet_size: |ALPH|
            integer_size: |INT|
            theta: ángulo RoPE (pequeño, default de freq_id=42)
            omega: ángulo = 2π/|ALPH| para codificar símbolos
        """
        self.alphabet_size = alphabet_size
        self.integer_size = integer_size
        self.theta = theta
        self.omega = omega if omega else 2 * math.pi / alphabet_size
        
        # Dimensión: |ALPH| × |ALPH| (pre-target) + |ALPH| × |INT| (target)
        self.pretarget_dim = alphabet_size * alphabet_size
        self.target_dim = alphabet_size * integer_size
        self.d = self.pretarget_dim + self.target_dim
        
        # Vector v no nulo
        self.v = torch.tensor([0.0, 1.0])
    
    def letter_to_index(self, letter):
        """Mapea letra a índice numérico (ϕ)."""
        return ord(letter) - ord('a')
    
    def compute_logit(self, query_left, query_right, 
                      key_left, key_right,
                      query_pos, key_pos,
                      query_is_target):
        """
        Computa el logit de atención para Retrieval.
        
        Para query en pre-target (L_q, R_q) con L_q, R_q ∈ ALPH:
            L = cos((ϕ(L_k) - ϕ(R_q))·ω + (n - j)·θ)
        
        donde la máxima atención ocurre cuando ϕ(L_k) = ϕ(R_q).
        """
        phi_k = self.letter_to_index(key_left)
        phi_q = self.letter_to_index(query_right)
        
        angle = (phi_k - phi_q) * self.omega + (query_pos - key_pos) * self.theta
        return math.cos(angle)
    
    def forward(self, sequence):
        """
        Ejecuta T_Retrieval sobre una secuencia.
        
        Args:
            sequence: list de tuplas (left, right)
              donde left ∈ ALPH y right ∈ ALPH ∪ INT
        
        Returns:
            output_sequence: aplicando Retrieval
        """
        n = len(sequence)
        output = list(sequence)
        
        for pos in range(n):
            left, right = sequence[pos]
            
            # Si el token actual es (letra, número) → es target, no cambiar
            if isinstance(right, int):
                continue
            
            # Si es (letra, letra) → buscar match
            # Buscar hacia atrás el par cuya primera letra = right
            for j in range(pos - 1, -1, -1):
                lj, rj = sequence[j]
                if lj == right:
                    output[pos] = sequence[j]
                    break
        
        return output


# =============================================================================
# THEOREM 3.2: COTAS DE DISCREPANCIA
# =============================================================================

def index_discrepancy_upper_bound(n, theta):
    """
    Cota superior de discrepancia para Index (Theorem 3.2).
    
    Δ_Index(n) ≤ min{1 - cos(θ), π²/(2n²)}
    
    Args:
        n: longitud de la secuencia
        theta: ángulo RoPE usado
    
    Returns:
        bound: float
    """
    bound1 = 1 - math.cos(theta)
    bound2 = math.pi**2 / (2 * n**2)
    return min(bound1, bound2)


def retrieval_discrepancy_upper_bound(n, theta, alphabet_size):
    """
    Cota superior de discrepancia para Retrieval (Theorem 3.2).
    
    Δ_Retrieval(n) ≤ 1 - cos(ω - n·θ)
    donde ω = 2π/|ALPH|
    
    Args:
        n: longitud de la secuencia
        theta: ángulo RoPE
        alphabet_size: |ALPH|
    
    Returns:
        bound: float
    """
    omega = 2 * math.pi / alphabet_size
    bound = 1 - math.cos(omega - n * theta)
    return bound


def compute_discrepancy_curves(max_len=300, theta_index=0.8, 
                                 theta_retrieval=0.0027, 
                                 alphabet_size=8):
    """
    Computa las curvas de discrepancia teórica para el rango de longitudes.
    
    Args:
        max_len: longitud máxima
        theta_index: ángulo para Index (θ₂ = 0.8, Table 1)
        theta_retrieval: ángulo para Retrieval (θ₄₂ = 0.0027)
        alphabet_size: |ALPH| = 8 para Letter Task
    
    Returns:
        index_bounds: list[float]
        retrieval_bounds: list[float]
        lengths: list[int]
    """
    lengths = list(range(1, max_len + 1))
    
    index_bounds = [
        index_discrepancy_upper_bound(n, theta_index)
        for n in lengths
    ]
    
    retrieval_bounds = [
        retrieval_discrepancy_upper_bound(n, theta_retrieval, alphabet_size)
        for n in lengths
    ]
    
    return index_bounds, retrieval_bounds, lengths


# =============================================================================
# REMARK 3.1: SEPARACIÓN POSICIONAL vs SIMBÓLICO
# =============================================================================

def prove_remark_3_1():
    """
    Demostración del Remark 3.1:
    
    1. Index NO puede implementarse con un head puramente simbólico.
       - Si el head es puramente simbólico, su salida es invariante
         ante permutaciones de tokens (Lemma 1, Urrutia et al., 2025).
       - Intercambiando dos letras en una secuencia, la respuesta
         cambia, pero un head simbólico daría la misma salida.
       → Contradicción.
    
    2. Retrieval NO puede implementarse con un head puramente posicional.
       - Un head posicional solo ve posiciones, no símbolos.
       - Retrieval requiere identificar qué token tiene la letra
         izquierda coincidente, lo que es inherentemente simbólico.
       - Theorem 4 de Urrutia et al. (2025) para Information Retrieval Task.
    
    Returns:
        proof_summary: str
    """
    return """
    Remark 3.1 — Separación fundamental:
    
    (a) Index es INHERENTEMENTE POSICIONAL:
        - Requiere atender a una posición específica (n - σ_n).
        - Un head simbólico no puede distinguir posiciones.
        - Prueba: intercambiar dos letras en la target window produce
          la misma respuesta en un modelo simbólico, pero la respuesta
          correcta cambia.
    
    (b) Retrieval es INHERENTEMENTE SIMBÓLICO:
        - Requiere atender al token cuya primera letra coincide con
          la segunda letra del query.
        - Un head posicional no puede identificar símbolos.
        - Prueba: mover el token objetivo a otra posición requiere
          atender a la nueva posición, pero un head posicional
          atendería a la vieja posición.
    
    Conclusión: La separación entre mecanismos posicionales y
    simbólicos no es solo una observación empírica, sino una
    consecuencia de la arquitectura de atención con RoPE.
    """
