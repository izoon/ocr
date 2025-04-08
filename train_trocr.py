import torch
from torch.utils.data import Dataset, DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainingArguments, Seq2SeqTrainer
from PIL import Image
import os
import json
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Optional
import evaluate

class ReceiptDataset(Dataset):
    def __init__(self, image_dir: str, json_file: str, processor: TrOCRProcessor, max_length: int = 128):
        self.image_dir = image_dir
        self.processor = processor
        self.max_length = max_length
        
        # Load annotations
        with open(json_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Filter out invalid entries
        self.annotations = [
            ann for ann in self.annotations 
            if os.path.exists(os.path.join(image_dir, ann['image']))
        ]
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        item = self.annotations[idx]
        image = Image.open(os.path.join(self.image_dir, item['image'])).convert("RGB")
        
        # Process image
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()
        
        # Process text
        text = item['text']
        labels = self.processor.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze()
        
        return {
            "pixel_values": pixel_values,
            "labels": labels
        }

def train_trocr(
    model_name: str = "microsoft/trocr-base-printed",
    train_data_dir: str = "receipts/train",
    val_data_dir: str = "receipts/val",
    train_json: str = "receipts/train.json",
    val_json: str = "receipts/val.json",
    output_dir: str = "trocr_receipt_model",
    num_train_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    print(f"Using device: {device}")
    
    # Initialize processor and model
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    
    # Move model to device
    model = model.to(device)
    
    # Create datasets
    train_dataset = ReceiptDataset(train_data_dir, train_json, processor)
    val_dataset = ReceiptDataset(val_data_dir, val_json, processor)
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        report_to="tensorboard",
    )
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=lambda pred: compute_metrics(pred, processor),
    )
    
    # Train the model
    trainer.train()
    
    # Save the model and processor
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    
    print(f"Model saved to {output_dir}")

def compute_metrics(pred, processor):
    # Decode predictions and labels
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    
    # Convert to text
    pred_text = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_text = processor.batch_decode(label_ids, skip_special_tokens=True)
    
    # Compute metrics
    cer_metric = evaluate.load("cer")
    wer_metric = evaluate.load("wer")
    
    cer = cer_metric.compute(predictions=pred_text, references=label_text)
    wer = wer_metric.compute(predictions=pred_text, references=label_text)
    
    return {
        "cer": cer,
        "wer": wer
    }

if __name__ == "__main__":
    # Example usage
    train_trocr(
        model_name="microsoft/trocr-base-printed",
        train_data_dir="receipts/train",
        val_data_dir="receipts/val",
        train_json="receipts/train.json",
        val_json="receipts/val.json",
        output_dir="trocr_receipt_model",
        num_train_epochs=3,
        batch_size=8,
        learning_rate=5e-5
    ) 