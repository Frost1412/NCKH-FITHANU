"""
Training script for emotion recognition models.
"""
import os
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import load_dataset, Audio
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import wandb


# Emotion label mapping
EMOTION_LABELS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
LABEL2ID = {label: idx for idx, label in enumerate(EMOTION_LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}


def load_metadata(metadata_path: str) -> pd.DataFrame:
    """Load dataset metadata from CSV."""
    return pd.read_csv(metadata_path)


def prepare_dataset(metadata_path: str, processor: Wav2Vec2Processor):
    """
    Prepare dataset from metadata CSV.
    
    Args:
        metadata_path: Path to CSV with file_path and label columns
        processor: Wav2Vec2 processor
        
    Returns:
        HuggingFace Dataset
    """
    # Load metadata
    df = load_metadata(metadata_path)
    
    # Create dataset dict
    dataset_dict = {
        'audio': df['file_path'].tolist(),
        'label': [LABEL2ID[label] for label in df['label'].tolist()]
    }
    
    # Create HuggingFace dataset
    from datasets import Dataset
    dataset = Dataset.from_dict(dataset_dict)
    
    # Cast audio column
    dataset = dataset.cast_column('audio', Audio(sampling_rate=16000))
    
    # Preprocess function
    def preprocess_function(examples):
        audio_arrays = [x['array'] for x in examples['audio']]
        inputs = processor(
            audio_arrays,
            sampling_rate=16000,
            return_tensors='pt',
            padding=True,
            max_length=16000 * 10,  # Max 10 seconds
            truncation=True
        )
        inputs['labels'] = examples['label']
        return inputs
    
    # Apply preprocessing
    dataset = dataset.map(
        preprocess_function,
        batched=True,
        batch_size=8,
        remove_columns=['audio']
    )
    
    return dataset


def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_weighted = f1_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }


def train_wav2vec2(
    train_dataset,
    val_dataset,
    output_dir: str,
    model_name: str = "facebook/wav2vec2-base",
    num_epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 3e-5,
    warmup_steps: int = 500,
    weight_decay: float = 0.01,
    use_wandb: bool = False
):
    """
    Train Wav2Vec2 model for emotion recognition.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        output_dir: Directory to save model checkpoints
        model_name: Pretrained model name
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        warmup_steps: Number of warmup steps
        weight_decay: Weight decay
        use_wandb: Whether to use Weights & Biases
    """
    print(f"\n{'=' * 60}")
    print(f"Training Wav2Vec2 Model")
    print(f"{'=' * 60}")
    print(f"Model: {model_name}")
    print(f"Output directory: {output_dir}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"{'=' * 60}\n")
    
    # Load processor and model
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(EMOTION_LABELS),
        label2id=LABEL2ID,
        id2label=ID2LABEL
    )
    
    # Freeze feature extractor
    model.freeze_feature_extractor()
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=3,
        report_to="wandb" if use_wandb else "none",
        run_name=f"wav2vec2-emotion-{datetime.now().strftime('%Y%m%d_%H%M%S')}" if use_wandb else None
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save best model
    best_model_path = Path(output_dir) / "best_model"
    trainer.save_model(best_model_path)
    processor.save_pretrained(best_model_path)
    print(f"\nBest model saved to: {best_model_path}")
    
    return trainer, model, processor


def evaluate_model(trainer, test_dataset, output_dir: str):
    """
    Evaluate model on test set.
    
    Args:
        trainer: Trained model trainer
        test_dataset: Test dataset
        output_dir: Directory to save results
    """
    print(f"\n{'=' * 60}")
    print("Evaluating on Test Set")
    print(f"{'=' * 60}\n")
    
    # Predict
    predictions = trainer.predict(test_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids
    
    # Metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    f1_macro = f1_score(true_labels, pred_labels, average='macro')
    f1_weighted = f1_score(true_labels, pred_labels, average='weighted')
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1 (Macro): {f1_macro:.4f}")
    print(f"Test F1 (Weighted): {f1_weighted:.4f}")
    
    # Classification report
    print(f"\n{'=' * 60}")
    print("Classification Report")
    print(f"{'=' * 60}\n")
    report = classification_report(
        true_labels,
        pred_labels,
        target_names=EMOTION_LABELS,
        digits=4
    )
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    print(f"\n{'=' * 60}")
    print("Confusion Matrix")
    print(f"{'=' * 60}\n")
    print("Predicted ->")
    print(f"{'True':12s}", end='')
    for label in EMOTION_LABELS:
        print(f"{label:12s}", end='')
    print()
    for i, label in enumerate(EMOTION_LABELS):
        print(f"{label:12s}", end='')
        for j in range(len(EMOTION_LABELS)):
            print(f"{cm[i][j]:12d}", end='')
        print()
    
    # Save results
    results = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'classification_report': classification_report(
            true_labels, pred_labels,
            target_names=EMOTION_LABELS,
            output_dict=True
        ),
        'confusion_matrix': cm.tolist()
    }
    
    results_path = Path(output_dir) / "test_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Train emotion recognition model")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=['ravdess', 'custom'],
        help="Dataset to use"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="wav2vec2",
        choices=['wav2vec2'],
        help="Model architecture"
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="data/processed",
        help="Directory with processed metadata"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: models/checkpoints/<dataset>_<timestamp>)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="facebook/wav2vec2-base",
        help="Pretrained model name"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Use Weights & Biases for logging"
    )
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = f"models/checkpoints/{args.dataset}_{timestamp}"
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(project="emotion-recognition", name=f"{args.dataset}_{args.model}")
    
    # Load processor
    processor = Wav2Vec2Processor.from_pretrained(args.model_name)
    
    # Prepare datasets
    print("Loading datasets...")
    train_dataset = prepare_dataset(
        f"{args.processed_dir}/{args.dataset}_train.csv",
        processor
    )
    val_dataset = prepare_dataset(
        f"{args.processed_dir}/{args.dataset}_val.csv",
        processor
    )
    test_dataset = prepare_dataset(
        f"{args.processed_dir}/{args.dataset}_test.csv",
        processor
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Train model
    trainer, model, processor = train_wav2vec2(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=args.output_dir,
        model_name=args.model_name,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_wandb=args.use_wandb
    )
    
    # Evaluate on test set
    evaluate_model(trainer, test_dataset, args.output_dir)
    
    print(f"\n{'=' * 60}")
    print("Training Complete!")
    print(f"{'=' * 60}")
    print(f"Model saved to: {args.output_dir}/best_model")
    print(f"To use this model, update backend/app/models/emotion_inference.py")


if __name__ == "__main__":
    main()
