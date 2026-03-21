#!/usr/bin/env python
# coding: utf-8
import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

import time
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from transformers import (
    AutoTokenizer, AutoFeatureExtractor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    default_data_collator, TrainerCallback
)
from evaluate import load

# === CONFIGURATION ===
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_CACHE"] = "NUL"
os.environ["HF_DATASETS_CACHE"] = "NUL"
os.environ["WANDB_DISABLED"] = "true"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class config:
    ENCODER = "google/vit-base-patch16-224-in21k"
    DECODER = "gpt2"
    TRAIN_BATCH_SIZE = 8
    VAL_BATCH_SIZE = 8
    GRAD_ACC_STEPS = 2
    LR = 5e-5
    EPOCHS = 1
    MAX_LEN = 128
    LABEL_MASK = -100
    ROOT_IMG_DIR = "flickr30k_images"
    MODEL_OUT = "vit-base-patch16-224-in21k_gpt2"

# === TOKENIZER & FEATURE EXTRACTOR ===
def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
    return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]

AutoTokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens

tokenizer = AutoTokenizer.from_pretrained(config.DECODER)
tokenizer.pad_token = tokenizer.unk_token
feature_extractor = AutoFeatureExtractor.from_pretrained(config.ENCODER)
rouge = load("rouge")

# === MÉTRIQUES D'ÉVALUATION ===
epochs, training_loss, validation_loss = [], [], []
rouge2_precision, rouge2_recall, rouge2_fmeasure = [], [], []

class TrainingLoggerCallback(TrainerCallback):
    def __init__(self, trainer_ref=None):
        self.trainer_ref = trainer_ref

    def on_epoch_end(self, args, state, control, **kwargs):
        trainer = self.trainer_ref or kwargs.get("trainer", None)
        if trainer is None: return control

        epoch = int(state.epoch)
        epochs.append(epoch)

        for log in reversed(state.log_history):
            if 'loss' in log:
                training_loss.append(log['loss'])
                break

        metrics = trainer.evaluate()
        validation_loss.append(metrics.get('eval_loss', 0))
        rouge2_precision.append(metrics.get('eval_rouge2_precision', 0))
        rouge2_recall.append(metrics.get('eval_rouge2_recall', 0))
        rouge2_fmeasure.append(metrics.get('eval_rouge2_fmeasure', 0))

        print(f"[INFO] Epoch {epoch} — Loss={log['loss']:.4f} — ROUGE2-F={rouge2_fmeasure[-1]:.4f}")
        return control

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == config.LABEL_MASK] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    result = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])
    rouge2 = result.get("rouge2", {})

    if isinstance(rouge2, dict):
        precision = rouge2.get("precision", 0)
        recall = rouge2.get("recall", 0)
        fmeasure = rouge2.get("fmeasure", 0)
    else:
        precision = recall = fmeasure = float(rouge2)  # fallback

    return {
        "rouge2_precision": round(precision, 4),
        "rouge2_recall": round(recall, 4),
        "rouge2_fmeasure": round(fmeasure, 4),
    }


# === DATASET ===
class ImgDataset(Dataset):
    def __init__(self, df, root_dir, tokenizer, feature_extractor):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.root_dir = root_dir
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        caption = self.df.caption.iloc[idx]
        image_path = os.path.join(self.root_dir, self.df.image.iloc[idx])
        img = Image.open(image_path).convert("RGB").resize((640, 640))
        img = self.transform(img)
        pixel_values = self.feature_extractor(images=img, return_tensors="pt", do_rescale=False).pixel_values
        tokenized = self.tokenizer(caption, padding='max_length', max_length=config.MAX_LEN, truncation=True).input_ids
        tokenized = [t if t != tokenizer.pad_token_id else config.LABEL_MASK for t in tokenized]
        return {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(tokenized)}

# === CHARGEMENT DES DONNÉES ===
df = pd.read_csv("results.csv", delimiter="|").sample(n=2000, random_state=42)
df.columns = ["image", "comment_number", "caption"]
df.drop(columns=["comment_number"], inplace=True)
train_df, val_df = train_test_split(df, test_size=0.2)

train_dataset = ImgDataset(train_df, config.ROOT_IMG_DIR, tokenizer, feature_extractor)
val_dataset = ImgDataset(val_df, config.ROOT_IMG_DIR, tokenizer, feature_extractor)

# === MODÈLE ===
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(config.ENCODER, config.DECODER)
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id or tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size
model.config.max_length = config.MAX_LEN
model.config.length_penalty = 2.0
model.config.no_repeat_ngram_size = 3
model.config.early_stopping = True
model.config.num_beams = 4
model = model.to(device)

# === ARGUMENTS D'ENTRAÎNEMENT ===
training_args = Seq2SeqTrainingArguments(
    output_dir=config.MODEL_OUT,
    logging_strategy="epoch",
    save_strategy="epoch",
    evaluation_strategy="epoch",
    per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=config.VAL_BATCH_SIZE,
    predict_with_generate=True,
    do_train=True,
    do_eval=True,
    learning_rate=config.LR,
    num_train_epochs=config.EPOCHS,
    overwrite_output_dir=True,
    fp16=torch.cuda.is_available()
)

trainer = Seq2SeqTrainer(
    tokenizer=tokenizer,
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=default_data_collator
)

trainer.add_callback(TrainingLoggerCallback(trainer_ref=trainer))

# === ENTRAÎNEMENT ===
start_time = time.time()
#checkpoint_dir = os.path.join(config.MODEL_OUT, "checkpoint-300")
#trainer.train(resume_from_checkpoint=checkpoint_dir if os.path.exists(checkpoint_dir) else None)
# Remplace par ceci pour forcer un nouveau début :
trainer.train()
trainer.save_model(config.MODEL_OUT)

end_time = time.time()
print(f"\n[INFO] Durée totale de l'entraînement : {int(end_time - start_time) // 60} min {int(end_time - start_time) % 60} sec")

# === COURBES METRIQUES ===
min_len = min(len(epochs), len(training_loss), len(validation_loss), len(rouge2_precision), len(rouge2_recall), len(rouge2_fmeasure))
epochs = epochs[:min_len]
training_loss = training_loss[:min_len]
validation_loss = validation_loss[:min_len]
rouge2_precision = rouge2_precision[:min_len]
rouge2_recall = rouge2_recall[:min_len]
rouge2_fmeasure = rouge2_fmeasure[:min_len]

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs, training_loss, label='Training Loss', marker='o')
plt.plot(epochs, validation_loss, label='Validation Loss', marker='o')
plt.title('Training & Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, rouge2_precision, label='ROUGE-2 Precision', marker='o')
plt.plot(epochs, rouge2_recall, label='ROUGE-2 Recall', marker='o')
plt.plot(epochs, rouge2_fmeasure, label='ROUGE-2 Fmeasure', marker='o')
plt.title('ROUGE-2 Scores')
plt.xlabel('Epochs')
plt.ylabel('Score')
plt.legend()

plt.tight_layout()
plt.savefig("training_metrics.png")
plt.show()

# === GÉNÉRATION DE LÉGENDES SUR IMAGES DE TEST ===
sample_imgs = ["36979.jpg", "65567.jpg"]
with open("generated_captions.txt", "w", encoding="utf-8") as f:
    for img_name in sample_imgs:
        print(f"\n[INFO] Génération pour : {img_name}")
        img_path = os.path.join(config.ROOT_IMG_DIR, img_name)
        img = Image.open(img_path).convert("RGB").resize((640, 640))
        pixel_values = feature_extractor(img, return_tensors="pt", do_rescale=False).pixel_values.to(device)

        output = model.generate(
            pixel_values,
            attention_mask=torch.ones(pixel_values.shape[:-1], dtype=torch.long).to(device),
            pad_token_id=tokenizer.pad_token_id
        )

        caption = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Caption : {caption}")
        f.write(f"{img_name} : {caption}\n")