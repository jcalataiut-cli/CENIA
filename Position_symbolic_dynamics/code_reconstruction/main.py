#!/usr/bin/env python3
"""
Punto de entrada principal para la reconstrucción del código.

Ejecuta el pipeline completo:
1. Generar datasets
2. Entrenar modelos
3. Evaluar
4. Calcular métricas
5. Generar visualizaciones
"""

import os
import sys
import yaml
import torch
import argparse

# Añadir directorio al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import (
    NumberTaskVocab, LetterTaskVocab,
    NumberTaskDataset, LetterTaskDataset
)
from model import GPTJModel, get_number_task_config, get_letter_task_config
from training import train_model, evaluate_accuracy
from metrics import compute_scores_for_model


def main():
    parser = argparse.ArgumentParser(
        description="Positional vs Symbolic Attention Heads"
    )
    parser.add_argument('--task', type=str, choices=['number', 'letter'],
                       required=True, help='Task to run')
    parser.add_argument('--mode', type=str, 
                       choices=['train', 'eval', 'metrics', 'all'],
                       default='all', help='Pipeline stage')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output-dir', type=str, default='./output')
    args = parser.parse_args()
    
    # Configuración
    if args.task == 'number':
        vocab = NumberTaskVocab()
        config = get_number_task_config()
        DatasetClass = NumberTaskDataset
    else:
        vocab = LetterTaskVocab()
        config = get_letter_task_config()
        DatasetClass = LetterTaskDataset
    
    config['learning_rate'] = args.lr
    config['num_epochs'] = args.epochs
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Crear modelo
    model = GPTJModel(config).to(device)
    print(f"Model created: {sum(p.numel() for p in model.parameters()):,} params")
    
    if args.mode in ['train', 'all']:
        # Datasets
        train_dataset = DatasetClass(vocab, split='train')
        val_dataset = DatasetClass(vocab, split='val')
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size
        )
        
        # Entrenar
        print(f"Training on {args.task} task...")
        history, checkpoints = train_model(
            model, train_loader, val_loader, config, device
        )
        
        # Guardar modelo
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(model.state_dict(), 
                  os.path.join(args.output_dir, f'model_{args.task}.pt'))
        print("Model saved.")
    
    if args.mode in ['eval', 'all']:
        # Cargar modelo guardado
        model_path = os.path.join(args.output_dir, f'model_{args.task}.pt')
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
        
        # Evaluar
        test_dataset = DatasetClass(vocab, split='test')
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size
        )
        
        accuracy, hop_acc = evaluate_accuracy(model, test_loader, device)
        print(f"Test accuracy: {accuracy:.4f}")
        for h, acc in hop_acc.items():
            print(f"  Hop {h}: {acc:.4f}")
    
    if args.mode in ['metrics', 'all']:
        test_dataset = DatasetClass(vocab, split='test')
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=32
        )
        
        print("Computing positional/symbolic scores...")
        pos_scores, sym_scores, entropy = compute_scores_for_model(
            model, test_loader, device
        )
        
        for layer, (pos, sym, ent) in enumerate(
            zip(pos_scores, sym_scores, entropy)
        ):
            print(f"Layer {layer}: Pos={pos:.3f}, Sym={sym:.3f}, Ent={ent:.3f}")


if __name__ == '__main__':
    main()
