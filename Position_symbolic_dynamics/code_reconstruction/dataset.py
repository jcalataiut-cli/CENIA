"""
Reconstrucción del código del paper:
"Positional versus Symbolic Attention Heads" (Urrutia et al., 2026)

Este módulo implementa la generación de datos para las tareas
Number Task y Letter Task, según las definiciones del Appendix B.
"""

import random
import torch
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# VOCABULARY
# =============================================================================

class NumberTaskVocab:
    """Vocabulario para Number Task (136 tokens)."""
    
    def __init__(self):
        # Letras: a-z (26) + bigramas aa-dp (94) = 120
        self.letters = [chr(ord('a') + i) for i in range(26)]
        self.bigrams = []
        for i in range(26):
            for j in range(26):
                bg = chr(ord('a') + i) + chr(ord('a') + j)
                self.bigrams.append(bg)
                if len(self.bigrams) >= 94:
                    break
            if len(self.bigrams) >= 94:
                break
        self.ALPH = self.letters + self.bigrams  # 120 símbolos
        
        # Enteros: {1, 2, ..., 16}
        self.INT = list(range(1, 17))
        
        # Vocabulario completo
        self.vocab = self.ALPH + [str(i) for i in self.INT]
        self.token_to_id = {t: i for i, t in enumerate(self.vocab)}
        self.id_to_token = {i: t for i, t in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)  # 136
    
    def __len__(self):
        return self.vocab_size


class LetterTaskVocab:
    """Vocabulario para Letter Task (128 tokens)."""
    
    def __init__(self):
        # 8 letras para el alfabeto reducido
        self.ALPH = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        # 8 enteros
        self.INT = list(range(1, 9))
        
        # Target window: pares (letra, número) → tokens atómicos
        self.target_tokens = [
            f"{l}{i}" for l in self.ALPH for i in self.INT
        ]  # 64 tokens
        
        # Pre-target window: pares (letra, letra) → tokens atómicos
        self.pretarget_tokens = [
            f"{l1}{l2}" for l1 in self.ALPH for l2 in self.ALPH
        ]  # 64 tokens
        
        # Vocabulario completo
        self.vocab = self.target_tokens + self.pretarget_tokens
        self.token_to_id = {t: i for i, t in enumerate(self.vocab)}
        self.id_to_token = {i: t for i, t in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)  # 128
    
    def __len__(self):
        return self.vocab_size


# =============================================================================
# DATASET GENERATION — NUMBER TASK
# =============================================================================

def generate_number_sequence(hops, vocab, target_letter=None, target_pos=None):
    """
    Genera una secuencia válida para Number Task.
    
    Definición formal (Appendix B.1):
    - Input: σ = s₁...s_{n₁} j₁...j_{n₂}
    - s_i ∈ ALPH (letras)
    - j_i ∈ INT (enteros)
    - Existe secuencia de hops j_{k₁}...j_{k_{HL}} ⊆ {j_{n₂}, ..., j₁}
      tal que:
      1. j_{k₁} = j_{n₂} (primer hop = último entero)
      2. n₁ - j_{k_{HL}} + 1 ∈ {1, ..., n₁} (último hop apunta a letra)
      3. k_i - k_{i+1} = j_{k_i} (cada hop define el siguiente)
    
    Args:
        hops: número de hops (1-4)
        vocab: instancia de NumberTaskVocab
        target_letter: letra solución (opcional)
        target_pos: posición en target window (opcional)
    
    Returns:
        sequence: lista de tokens (str)
        answer: token solución
        hop_count: número de hops
    """
    n1 = 8  # target window length
    n2 = 8  # pre-target length
    
    # Seleccionar letra y posición objetivo
    if target_letter is None:
        target_letter = random.choice(vocab.ALPH)
    if target_pos is None:
        target_pos = random.randint(0, n1 - 1)
    
    # Construir target window con la letra solución en target_pos
    target_window = []
    for i in range(n1):
        if i == target_pos:
            target_window.append(target_letter)
        else:
            # Letras aleatorias (distintas de la solución para evitar ambigüedad)
            letter = random.choice([l for l in vocab.ALPH if l != target_letter])
            target_window.append(letter)
    
    # Construir la secuencia de hops
    # Los hops se resuelven desde el final hacia atrás
    # hop 1: j_{k₁} = último entero
    # hop 2: j_{k₂} en posición n₂ - j_{k₁}
    # ...
    
    # Posiciones en el pre-target window
    hop_sequence = []
    current_pos = n2 - 1  # empezar desde el último
    
    for h in range(hops):
        # Valor del hop: cuántas posiciones retroceder
        if h == hops - 1:
            # Último hop: debe apuntar a target_pos
            hop_value = target_pos + 1  # +1 porque es 1-indexado
        else:
            hop_value = random.randint(1, min(8, current_pos))
        
        hop_sequence.append((current_pos, hop_value))
        
        # Siguiente posición
        if h < hops - 1:
            current_pos = current_pos - hop_value
    
    # Verificar que la secuencia de hops es válida
    # (la posición final debe ser alcanzable)
    final_pos = n2 - sum(h for _, h in hop_sequence)
    if not (0 <= final_pos < n2):
        # Reintentar si no es válida
        return generate_number_sequence(hops, vocab)
    
    # Construir pre-target window
    pre_target = []
    for i in range(n2):
        # Asignar valor de hop si esta posición corresponde
        matched = False
        for pos, val in hop_sequence:
            if pos == i:
                pre_target.append(str(val))
                matched = True
                break
        if not matched:
            # Números aleatorios
            pre_target.append(str(random.randint(1, 8)))
    
    # Secuencia completa
    sequence = target_window + pre_target
    answer = target_letter
    
    return sequence, answer, hops


class NumberTaskDataset(Dataset):
    """Dataset para Number Task."""
    
    def __init__(self, vocab, num_sequences=480000, split='train'):
        self.vocab = vocab
        self.split = split
        
        # Generar datos
        sequences = []
        answers = []
        hop_counts = []
        
        num_per_hop = num_sequences // 4
        for hops in [1, 2, 3, 4]:
            for _ in range(num_per_hop):
                seq, ans, h = generate_number_sequence(hops, vocab)
                sequences.append(seq)
                answers.append(ans)
                hop_counts.append(h)
        
        # Split train/val/test (90/0.5/9.5)
        n = len(sequences)
        if split == 'train':
            self.sequences = sequences[:int(n * 0.9)]
            self.answers = answers[:int(n * 0.9)]
            self.hop_counts = hop_counts[:int(n * 0.9)]
        elif split == 'val':
            start = int(n * 0.9)
            end = int(n * 0.905)
            self.sequences = sequences[start:end]
            self.answers = answers[start:end]
            self.hop_counts = hop_counts[start:end]
        else:  # test
            self.sequences = sequences[int(n * 0.905):]
            self.answers = answers[int(n * 0.905):]
            self.hop_counts = hop_counts[int(n * 0.905):]
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        ans = self.answers[idx]
        hops = self.hop_counts[idx]
        
        # Tokenizar
        input_ids = [self.vocab.token_to_id[t] for t in seq]
        
        # Labels: -100 para todo excepto el último token
        labels = [-100] * (len(input_ids) - 1) + [self.vocab.token_to_id[ans]]
        
        return {
            'input_ids': torch.tensor(input_ids),
            'labels': torch.tensor(labels),
            'hop_count': torch.tensor(hops),
        }


# =============================================================================
# DATASET GENERATION — LETTER TASK
# =============================================================================

def generate_letter_sequence(hops, vocab):
    """
    Genera una secuencia válida para Letter Task.
    
    Definición formal (Appendix B.2):
    - Input: (s₁,w₁)...(s_{n₁},w_{n₁}) (x₁,y₁)...(x_{n₂},y_{n₂})
    - s_i, x_i, y_i ∈ ALPH (letras)
    - w_i ∈ INT (enteros)
    - Existe secuencia (x_{k₁},y_{k₁})...(x_{k_{HL}},y_{k_{HL}}) ⊆ pre-target
      tal que:
      1. (x_{k₁},y_{k₁}) = (x_{n₂},y_{n₂}) (último par del pre-target)
      2. y_{k_H} = s_{i*} (último hop apunta a target)
      3. y_{k_i} = x_{k_{i+1}} (encadenamiento: segunda letra = primera del siguiente)
    
    Args:
        hops: número de hops (1-4)
        vocab: instancia de LetterTaskVocab
    
    Returns:
        sequence: lista de tokens (str)
        answer: token solución
        hop_count: número de hops
    """
    n1 = 8  # target window length
    n2 = 8  # pre-target length
    
    # Elegir letra y número solución
    solution_letter = random.choice(vocab.ALPH)
    solution_number = random.randint(1, 8)
    solution_token = f"{solution_letter}{solution_number}"
    
    # Construir pre-target window con la cadena de hops
    # Cada par (x,y) donde x es la primera letra e y es la segunda
    # Regla: y_{hop_i} = x_{hop_{i+1}}
    
    # Generar la cadena de hops desde el final
    chain = []
    current_letter = solution_letter
    
    for h in range(hops):
        first_letter = random.choice(
            [l for l in vocab.ALPH if l != current_letter]
        )
        chain.append((first_letter, current_letter))
        current_letter = first_letter
    
    # Invertir para orden correcto
    chain = chain[::-1]
    
    # Construir pre-target window
    pre_target = []
    chain_idx = 0
    for i in range(n2):
        if i >= n2 - len(chain):
            # Usar los pares de la cadena
            x, y = chain[chain_idx]
            pre_target.append(f"{x}{y}")
            chain_idx += 1
        else:
            # Pares aleatorios
            x = random.choice(vocab.ALPH)
            y = random.choice(vocab.ALPH)
            pre_target.append(f"{x}{y}")
    
    # Construir target window
    target_window = [
        f"{random.choice(vocab.ALPH)}{random.randint(1, 8)}"
        for _ in range(n1)
    ]
    # Poner la solución en una posición aleatoria
    target_pos = random.randint(0, n1 - 1)
    target_window[target_pos] = solution_token
    
    # Secuencia completa
    sequence = target_window + pre_target + [pre_target[-1]]
    # Nota: el paper usa 17 tokens (8+8+1), el último es el query
    
    answer = solution_token
    
    return sequence, answer, hops


class LetterTaskDataset(Dataset):
    """Dataset para Letter Task."""
    
    def __init__(self, vocab, num_sequences=480000, split='train'):
        self.vocab = vocab
        self.split = split
        
        sequences = []
        answers = []
        hop_counts = []
        
        num_per_hop = num_sequences // 4
        for hops in [1, 2, 3, 4]:
            for _ in range(num_per_hop):
                seq, ans, h = generate_letter_sequence(hops, vocab)
                sequences.append(seq)
                answers.append(ans)
                hop_counts.append(h)
        
        n = len(sequences)
        if split == 'train':
            self.sequences = sequences[:int(n * 0.9)]
            self.answers = answers[:int(n * 0.9)]
            self.hop_counts = hop_counts[:int(n * 0.9)]
        elif split == 'val':
            start = int(n * 0.9)
            end = int(n * 0.905)
            self.sequences = sequences[start:end]
            self.answers = answers[start:end]
            self.hop_counts = hop_counts[start:end]
        else:
            self.sequences = sequences[int(n * 0.905):]
            self.answers = answers[int(n * 0.905):]
            self.hop_counts = hop_counts[int(n * 0.905):]
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        ans = self.answers[idx]
        hops = self.hop_counts[idx]
        
        input_ids = [self.vocab.token_to_id[t] for t in seq]
        labels = [-100] * (len(input_ids) - 1) + [self.vocab.token_to_id[ans]]
        
        return {
            'input_ids': torch.tensor(input_ids),
            'labels': torch.tensor(labels),
            'hop_count': torch.tensor(hops),
        }
