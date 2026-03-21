# image_captioning_vit_gpt2.py

import os
import time
import torch
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoFeatureExtractor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    default_data_collator
)
from evaluate import load
from torchvision import transforms
import matplotlib.pyplot as plt

# --------------------------
# 1. CONFIGURATION
# --------------------------
class Config:
    ENCODER = "google/vit-base-patch16-224-in21k"
    DECODER = "gpt2"
    TRAIN_BATCH_SIZE = 8
    VAL_BATCH_SIZE = 8
    EPOCHS = 8
    LR = 5e-5
    MAX_LEN = 128
    IMG_SIZE = (224, 224)
    LABEL_MASK = -100
    SEED = 42
    OUTPUT_DIR = "vit-base-patch16-224-in21k_gpt2_Medium"
    NUM_WORKERS = os.cpu_count()

config = Config()

# Fix TF conflicts
os.environ["USE_TF"] = "0"
os.environ["WANDB_DISABLED"] = "true"

# --------------------------
# 2. DEVICE SETUP
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# --------------------------
# 3. LOAD TOKENIZER + EXTRACTOR
# --------------------------
tokenizer = AutoTokenizer.from_pretrained(config.DECODER)
tokenizer.pad_token = tokenizer.eos_token
feature_extractor = AutoFeatureExtractor.from_pretrained(config.ENCODER)

# --------------------------
# 4. METRICS FUNCTION
# --------------------------
rouge = load("rouge")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    scores = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])
    rouge2 = scores["rouge2"]

    if isinstance(rouge2, dict):
        return {
            "rouge2_precision": round(rouge2["precision"], 4),
            "rouge2_recall": round(rouge2["recall"], 4),
            "rouge2_fmeasure": round(rouge2["fmeasure"], 4),
        }
    else:
        return {
            "rouge2_fmeasure": round(rouge2, 4)
        }



# --------------------------
# 5. LOAD DATASET
# --------------------------
df = pd.read_csv("results.csv", delimiter="|").sample(n=2000, random_state=config.SEED)
df.columns = ["image", "comment_number", "caption"]
df.drop(columns=["comment_number"], inplace=True)
train_df, val_df = train_test_split(df, test_size=0.2)

# --------------------------
# 6. DATASET CLASS
# --------------------------
class ImgDataset(Dataset):
    def __init__(self, df, root_dir, tokenizer, feature_extractor):
        self.df = df
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row.image)
        img = Image.open(img_path).convert("RGB")
        pixel_values = self.feature_extractor(images=img, return_tensors="pt", do_rescale=False).pixel_values.squeeze()
        tokens = self.tokenizer(
            row.caption,
            padding='max_length',
            max_length=50,
            truncation=True
        ).input_ids
        labels = [tok if tok != tokenizer.pad_token_id else config.LABEL_MASK for tok in tokens]
        return {"pixel_values": pixel_values, "labels": torch.tensor(labels)}

# --------------------------
# 7. DATA LOADERS
# --------------------------
train_dataset = ImgDataset(train_df, "flickr30k_images", tokenizer, feature_extractor)
val_dataset = ImgDataset(val_df, "flickr30k_images", tokenizer, feature_extractor)

# --------------------------
# 8. LOAD MODEL
# --------------------------
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(config.ENCODER, config.DECODER)
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size
model.config.eos_token_id = tokenizer.eos_token_id
model.config.max_length = config.MAX_LEN
model.config.length_penalty = 2.0
model.config.num_beams = 4
model.to(device)

# --------------------------
# 9. TRAINING ARGUMENTS
# --------------------------
training_args = Seq2SeqTrainingArguments(
    output_dir=config.OUTPUT_DIR,
    per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=config.VAL_BATCH_SIZE,
    predict_with_generate=True,
    do_train=True,
    do_eval=True,
    logging_steps=500,
    evaluation_strategy="epoch",
    learning_rate=config.LR,
    num_train_epochs=config.EPOCHS,
    overwrite_output_dir=True,
    fp16=torch.cuda.is_available(),
    report_to="none"
)

# --------------------------
# 10. TRAINING
# --------------------------
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=default_data_collator,
)

start = time.time()
train_result = trainer.train()
end = time.time()

print(f"\nTraining time: {(end - start)/60:.2f} minutes")

# Save training loss metrics as graph
def save_loss_plots(trainer):
    logs = trainer.state.log_history
    train_loss = [log['loss'] for log in logs if 'loss' in log]
    eval_loss = [log['eval_loss'] for log in logs if 'eval_loss' in log]

    plt.figure(figsize=(10, 5))

    if train_loss:
        train_steps = list(range(1, len(train_loss) + 1))
        plt.plot(train_steps, train_loss, label='Training Loss')

    if eval_loss:
        eval_steps = list(range(1, len(eval_loss) + 1))
        plt.plot(eval_steps, eval_loss, label='Validation Loss')

    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(config.OUTPUT_DIR, "loss_plot.png"))
    plt.close()


save_loss_plots(trainer)

# --------------------------
# 11. SAVE MODEL
# --------------------------
trainer.save_model(config.OUTPUT_DIR)
print("✅ Modèle sauvegardé dans :", config.OUTPUT_DIR)
print("📁 Contenu du dossier :", os.listdir(config.OUTPUT_DIR))
# --------------------------
# 12. INFERENCE FUNCTION
# --------------------------
def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    pixel_values = feature_extractor(image, return_tensors="pt", do_rescale=False).pixel_values.to(device)
    output_ids = model.generate(pixel_values)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# --------------------------
# 13. PLOT TRAINING METRICS
# --------------------------
import os
import matplotlib.pyplot as plt

import os
import matplotlib.pyplot as plt

def plot_training_metrics_separately(train_loss, val_loss, rouge2_scores, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    epochs = list(range(1, len(train_loss) + 1))

    # Plot 1: Training & Validation Loss
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    loss_path = os.path.join(output_dir, "training_validation_loss.png")
    plt.savefig(loss_path)
    print(f"✅ Saved: {loss_path}")
    plt.close()

    # Plot 2: ROUGE-2 F1 Score
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, rouge2_scores, label='ROUGE-2 F1')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('ROUGE-2 F1 Score')
    plt.legend()
    rouge_path = os.path.join(output_dir, "rouge2_score.png")
    plt.savefig(rouge_path)
    print(f"✅ Saved: {rouge_path}")
    plt.close()



# --------------------------
# 14. BATCH INFERENCE
# --------------------------
def evaluate_multiple_images(image_paths):
    results = {}
    for img_path in image_paths:
        try:
            caption = generate_caption(img_path)
            results[img_path] = caption
        except Exception as e:
            results[img_path] = f"Error: {str(e)}"
    return results

# Example:
example_path = "flickr30k_images/36979.jpg"
print("Generated caption:", generate_caption(example_path))

# Optional batch test
# image_files = ["flickr30k_images/image1.jpg", "flickr30k_images/image2.jpg"]
# print(evaluate_multiple_images(image_files))