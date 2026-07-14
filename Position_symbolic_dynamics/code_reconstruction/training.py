"""
Entrenamiento y evaluación del modelo GPT-J con last-token prediction.

El objetivo es minimizar la negative log-likelihood SOLO en el último token:
    L(θ) = -log P_θ(x_n | x_1, ..., x_{n-1})

Todos los tokens anteriores tienen ignore_index = -100.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(model, dataloader, optimizer, device='cpu'):
    """
    Entrena el modelo por una época.
    
    La pérdida solo se calcula en el último token (last-token prediction).
    
    Args:
        model: GPTJModel
        dataloader: DataLoader
        optimizer: optimizador
        device: dispositivo
    
    Returns:
        avg_loss: pérdida promedio
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        logits, _ = model(input_ids)
        
        # Pérdida solo en el último token
        # logits: (batch, seq_len, vocab_size)
        # labels: (batch, seq_len) con -100 en posiciones ignoradas
        loss = nn.CrossEntropyLoss()(
            logits[:, -1, :],  # Solo último paso de tiempo
            labels[:, -1]       # Solo etiqueta del último token
        )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def train_model(model, train_loader, val_loader, config, device='cpu'):
    """
    Entrenamiento completo con logging.
    
    Args:
        model: GPTJModel
        train_loader: DataLoader de entrenamiento
        val_loader: DataLoader de validación
        config: dict con hiperparámetros de entrenamiento
        device: dispositivo
    
    Returns:
        history: dict con historial de pérdidas
        checkpoints: list de estados del modelo
    """
    learning_rate = config.get('learning_rate', 1e-4)
    num_epochs = config.get('num_epochs', 10)
    save_every = config.get('save_every', 500)  # pasos
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=config.get('weight_decay', 0.01)
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )
    
    history = {'train_loss': [], 'val_loss': []}
    checkpoints = []
    global_step = 0
    
    for epoch in range(num_epochs):
        # Training
        train_loss = train_epoch(model, train_loader, optimizer, device)
        history['train_loss'].append(train_loss)
        
        # Validation
        val_loss = evaluate_loss(model, val_loader, device)
        history['val_loss'].append(val_loss)
        
        # Scheduler step
        scheduler.step()
        
        # Logging
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            checkpoints.append(model.state_dict())
    
    return history, checkpoints


def evaluate_loss(model, dataloader, device='cpu'):
    """
    Evalúa la pérdida en un dataset.
    
    Args:
        model: GPTJModel
        dataloader: DataLoader
        device: dispositivo
    
    Returns:
        avg_loss: pérdida promedio
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            logits, _ = model(input_ids)
            
            loss = nn.CrossEntropyLoss()(
                logits[:, -1, :],
                labels[:, -1]
            )
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_accuracy(model, dataloader, device='cpu'):
    """
    Evalúa la precisión del modelo.
    
    Calcula:
    - Accuracy global (promedio sobre todos los hops)
    - Accuracy por hop (1, 2, 3, 4)
    
    Args:
        model: GPTJModel
        dataloader: DataLoader
        device: dispositivo
    
    Returns:
        accuracy: float global
        hop_accuracies: dict {hop: accuracy}
    """
    model.eval()
    correct = 0
    total = 0
    hop_correct = {1: 0, 2: 0, 3: 0, 4: 0}
    hop_total = {1: 0, 2: 0, 3: 0, 4: 0}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating accuracy"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            hop_counts = batch['hop_count'].to(device)
            
            logits, _ = model(input_ids)
            
            # Predicción en el último token
            predictions = logits[:, -1, :].argmax(dim=-1)
            
            # Comparar con etiquetas (último token)
            targets = labels[:, -1]
            
            for i in range(len(predictions)):
                pred = predictions[i].item()
                target = targets[i].item()
                hops = hop_counts[i].item()
                
                if pred == target:
                    correct += 1
                    if hops in hop_correct:
                        hop_correct[hops] += 1
                
                total += 1
                if hops in hop_total:
                    hop_total[hops] += 1
    
    accuracy = correct / total if total > 0 else 0.0
    hop_accuracies = {
        h: hop_correct[h] / hop_total[h]
        for h in range(1, 5) if hop_total[h] > 0
    }
    
    return accuracy, hop_accuracies


def evaluate_generalization(model, dataloader, extended_lengths, 
                            vocab, device='cpu'):
    """
    Evalúa la generalización a secuencias más largas.
    
    Se extiende la target window añadiendo tokens extra antes de la
    ventana que contiene la respuesta. La dificultad de la tarea
    no cambia porque el modelo no necesita mirar los tokens extra.
    
    Args:
        model: GPTJModel
        dataloader: DataLoader base (longitud original)
        extended_lengths: list de longitudes a probar
        vocab: vocabulario
        device: dispositivo
    
    Returns:
        results: dict {length: accuracy}
    """
    results = {}
    
    for length in extended_lengths:
        # Crear dataset con longitud extendida
        # (añadir tokens neutros antes de la target window)
        extended_loader = create_extended_dataloader(
            dataloader.dataset, length, vocab, batch_size=dataloader.batch_size
        )
        
        accuracy, _ = evaluate_accuracy(model, extended_loader, device)
        results[length] = accuracy
        
        print(f"Length {length}: Accuracy {accuracy:.4f}")
    
    return results


def create_extended_dataloader(base_dataset, new_length, vocab, batch_size=32):
    """
    Crea un DataLoader con secuencias de longitud extendida.
    
    Se añaden tokens de padding/palabras neutras antes de la target window,
    manteniendo la respuesta en la misma posición relativa.
    """
    # Placeholder: en la práctica, se genera un nuevo dataset
    # con la longitud deseada y se devuelve el DataLoader
    pass
