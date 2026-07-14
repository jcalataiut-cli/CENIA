"""
Implementación del modelo GPT-J con single-head attention y RoPE.

Basado en la descripción del Appendix C del paper:
- 12 layers, single head per layer
- Hidden dimension: 128
- RoPE positional encoding
- GELU activations
- Last-token prediction objective
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# RoPE (Rotary Positional Encoding)
# =============================================================================

class RotaryPositionalEncoding(nn.Module):
    """
    Rotary Positional Encoding (RoPE) - Su et al. (2024).
    
    RoPE rota los vectores query y key según su posición,
    de modo que el producto punto capture información relativa.
    
    Para dimensión d, se definen d/2 frecuencias:
        θ_k = base^(-2k/d)  para k = 0, ..., d/2 - 1
    
    La rotación para el par (x_{2k}, x_{2k+1}) en posición pos es:
        RoPE(x, pos)_{2k}   = x_{2k}·cos(pos·θ_k) - x_{2k+1}·sin(pos·θ_k)
        RoPE(x, pos)_{2k+1} = x_{2k}·sin(pos·θ_k) + x_{2k+1}·cos(pos·θ_k)
    """
    
    def __init__(self, d_model, base=10000.0):
        super().__init__()
        self.d_model = d_model
        self.base = base
        
        # Pre-computar frecuencias
        inv_freq = 1.0 / (
            base ** (torch.arange(0, d_model, 2).float() / d_model)
        )
        self.register_buffer('inv_freq', inv_freq)
    
    def _compute_rotate_cache(self, max_seq_len, device):
        """
        Pre-computa los valores cos y sin para posiciones 0..max_seq_len-1.
        
        Returns:
            cos, sin: (max_seq_len, d_model/2)
        """
        t = torch.arange(max_seq_len, device=device).float()
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)  # (seq_len, d/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, d)
        
        cos = emb.cos()
        sin = emb.sin()
        
        return cos, sin
    
    def forward(self, x, positions=None):
        """
        Aplica RoPE a un tensor.
        
        Args:
            x: (batch, seq_len, d_model) o (..., d_model)
            positions: (seq_len,) índices de posición (default: 0..seq_len-1)
        
        Returns:
            x_rotated: misma forma que x
        """
        if positions is None:
            seq_len = x.shape[-2]
            positions = torch.arange(seq_len, device=x.device)
        
        cos, sin = self._compute_rotate_cache(
            positions.max().item() + 1, x.device
        )
        
        # Seleccionar cos/sin para las posiciones dadas
        cos = cos[positions]  # (seq_len, d_model)
        sin = sin[positions]  # (seq_len, d_model)
        
        # Aplicar rotación por pares
        x_rotated = torch.zeros_like(x)
        x_rotated[..., 0::2] = (
            x[..., 0::2] * cos[..., 0::2] - x[..., 1::2] * sin[..., 0::2]
        )
        x_rotated[..., 1::2] = (
            x[..., 0::2] * sin[..., 1::2] + x[..., 1::2] * cos[..., 1::2]
        )
        
        return x_rotated


# =============================================================================
# SINGLE HEAD ATTENTION with RoPE
# =============================================================================

class SingleHeadAttention(nn.Module):
    """
    Atención single-head con RoPE.
    
    El logit de atención entre query en posición i y key en posición j es:
        L(q_i, k_j) = (R_j · k_j)^T · (R_i · q_i) = k_j^T · R^{i-j} · q_i
    
    donde R_i es la matriz de rotación RoPE para la posición i.
    
    Esta es la arquitectura clave del paper, ya que permite:
    - Atención posicional: cuando los key vectors son iguales y solo
      varían por RoPE según su posición.
    - Atención simbólica: cuando los key vectors codifican la identidad
      del símbolo y RoPE tiene poco efecto (frecuencias bajas).
    """
    
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_model  # single head
        
        # Proyecciones lineales Q, K, V
        self.q_proj = nn.Linear(d_model, self.d_head, bias=False)
        self.k_proj = nn.Linear(d_model, self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, self.d_head, bias=False)
        self.out_proj = nn.Linear(self.d_head, d_model, bias=False)
        
        # RoPE
        self.rope = RotaryPositionalEncoding(self.d_head)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None, return_attention=True):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len, seq_len) — causal mask
            return_attention: si True, devuelve pesos de atención
        
        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: (batch, seq_len, seq_len) [opcional]
        """
        batch, seq_len, _ = x.shape
        
        # Proyecciones Q, K, V
        q = self.q_proj(x)  # (batch, seq_len, d_head)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Aplicar RoPE a Q y K
        positions = torch.arange(seq_len, device=x.device)
        q = self.rope(q, positions)
        k = self.rope(k, positions)
        
        # Scaled dot-product attention
        # scores = q · k^T / sqrt(d_head)
        scores = torch.matmul(
            q, k.transpose(-2, -1)
        ) / math.sqrt(self.d_head)
        
        # Causal mask (decoder-only)
        if mask is None:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device) * float('-inf'),
                diagonal=1
            )
        scores = scores + mask
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Weighted sum of values
        output = torch.matmul(attention_weights, v)
        
        # Output projection
        output = self.out_proj(output)
        
        if return_attention:
            return output, attention_weights
        return output


# =============================================================================
# TRANSFORMER BLOCK
# =============================================================================

class TransformerBlock(nn.Module):
    """
    Bloque Transformer con pre-norm y single-head attention.
    
    Cada bloque:
    1. LayerNorm → Attention → Residual
    2. LayerNorm → MLP (GELU) → Residual
    """
    
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        
        self.ln1 = nn.LayerNorm(d_model)
        self.attention = SingleHeadAttention(d_model, dropout)
        
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: causal mask
        
        Returns:
            x: (batch, seq_len, d_model)
            attn_weights: (batch, seq_len, seq_len)
        """
        # Attention with pre-norm
        attn_out, attn_weights = self.attention(self.ln1(x), mask)
        x = x + attn_out
        
        # MLP with pre-norm
        x = x + self.mlp(self.ln2(x))
        
        return x, attn_weights


# =============================================================================
# GPT-J MODEL (Decoder-only, 12 layers, single head)
# =============================================================================

class GPTJModel(nn.Module):
    """
    Decoder-only Transformer basado en GPT-J (Wang, 2021).
    
    Arquitectura específica del paper:
    - 12 capas
    - 1 attention head por capa (single-head)
    - Dimensión oculta: 128
    - RoPE para codificación posicional
    - GELU activations
    - Dropout: 0.1
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.embedding = nn.Embedding(
            config['vocab_size'], config['d_model']
        )
        self.dropout = nn.Dropout(config.get('dropout_rate', 0.1))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config['d_model'],
                dropout=config.get('dropout_rate', 0.1),
            ) for _ in range(config['layers'])
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config['d_model'])
        
        # LM Head (weight tying con embedding)
        self.lm_head = nn.Linear(
            config['d_model'], config['vocab_size'], bias=False
        )
        self.lm_head.weight = self.embedding.weight
    
    def forward(self, input_ids, return_attention=True):
        """
        Forward pass completo.
        
        Args:
            input_ids: (batch, seq_len) — índices de tokens
            return_attention: si True, captura matrices de atención
        
        Returns:
            logits: (batch, seq_len, vocab_size)
            attention_maps: list[(batch, seq_len, seq_len)] por capa
        """
        batch, seq_len = input_ids.shape
        device = input_ids.device
        
        # Embedding
        x = self.embedding(input_ids)
        x = self.dropout(x)
        
        # Causal mask
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float('-inf'),
            diagonal=1
        )
        
        # Pass through transformer blocks
        attention_maps = []
        for block in self.blocks:
            x, attn = block(x, mask)
            if return_attention:
                attention_maps.append(attn)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # LM Head
        logits = self.lm_head(x)
        
        return logits, attention_maps
    
    def forward_until(self, input_ids, layer_idx):
        """
        Forward hasta una capa específica (para extraer representaciones intermedias).
        
        Args:
            input_ids: (batch, seq_len)
            layer_idx: capa hasta la cual forward
        
        Returns:
            hidden_states: (batch, seq_len, d_model)
            attentions: list de atención hasta layer_idx
        """
        batch, seq_len = input_ids.shape
        device = input_ids.device
        
        x = self.embedding(input_ids)
        x = self.dropout(x)
        
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float('-inf'),
            diagonal=1
        )
        
        attentions = []
        for i in range(layer_idx + 1):
            x, attn = self.blocks[i](x, mask)
            attentions.append(attn)
        
        return x, attentions
    
    def get_hidden_states(self, input_ids, layer_idx):
        """
        Obtiene los hidden states en una capa específica.
        """
        batch, seq_len = input_ids.shape
        device = input_ids.device
        
        x = self.embedding(input_ids)
        x = self.dropout(x)
        
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float('-inf'),
            diagonal=1
        )
        
        for i in range(layer_idx + 1):
            x, _ = self.blocks[i](x, mask)
        
        return x


# =============================================================================
# CONFIGURACIONES
# =============================================================================

def get_number_task_config():
    """Configuración para Number Task."""
    return {
        'layers': 12,
        'd_model': 128,
        'd_head': 128,
        'd_ff': 512,
        'vocab_size': 136,  # 120 letras + 16 enteros
        'max_seq_len': 17,
        'dropout_rate': 0.1,
        'activation': 'gelu',
        'use_rope': True,
        'rope_base': 10000.0,
    }

def get_letter_task_config():
    """Configuración para Letter Task."""
    return {
        'layers': 12,
        'd_model': 128,
        'd_head': 128,
        'd_ff': 512,
        'vocab_size': 128,  # 64 (letra,núm) + 64 (letra,letra)
        'max_seq_len': 17,
        'dropout_rate': 0.1,
        'activation': 'gelu',
        'use_rope': True,
        'rope_base': 10000.0,
    }


def create_model(config):
    """Crea una instancia del modelo GPT-J."""
    return GPTJModel(config)
