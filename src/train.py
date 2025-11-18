import os
import math
from transformers import AutoTokenizer, AutoConfig
from datasets import load_dataset, DatasetDict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import wandb
from torchvision import transforms
from tqdm.auto import tqdm
## Custom imports
from transformers import AutoModel, Trainer, TrainingArguments
from itertools import islice
import random

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from config.model_config import (
    MODEL_NAME, CKPT_PATH_DOCQA, BATCH_SIZE, GRAD_ACC_STEPS,
    get_t5_config
)
from utils.auth import setup_authentication
from models.tilt_model import TiLTDocQATransformer
from data.dataset import DocVQADataset
from data.collate import DocVQACollateFn
from training.optimizer import create_optimizer
from training.scheduler import get_arctic_tilt_scheduler
from training.callbacks import ValidationMetricsCallback
from metrics import compute_docqa_metrics

setup_authentication()
device = "cuda" if torch.cuda.is_available() else "cpu"

hf_ds = load_dataset("hxlinh/SP-DocVQA")

train_split = hf_ds['train'].train_test_split(test_size=0.1, seed=42)

docvqa_dataset = DatasetDict({
    'train': train_split['train'],
    'validation': train_split['test'],
    'test': hf_ds['val']
})

print(" THỐNG KÊ DATASET:")
print(f"   • Train: {len(docvqa_dataset['train'])} samples")
print(f"   • Validation: {len(docvqa_dataset['validation'])} samples")
print(f"   • Test: {len(docvqa_dataset['test'])} samples")
print(f"   • Tổng: {len(docvqa_dataset['train']) + len(docvqa_dataset['validation']) + len(docvqa_dataset['test'])} samples")

t5_config_docvqa = get_t5_config()

transform_docqa = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 2 * x - 1)
])

tokenizer_docqa = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True,
    model_max_length=t5_config_docvqa.model_max_length
)

if tokenizer_docqa.pad_token is None:
    tokenizer_docqa.pad_token = tokenizer_docqa.eos_token

print(f" Tokenizer info:")
print(f"   Vocab size: {tokenizer_docqa.vocab_size}")
print(f"   Pad token: '{tokenizer_docqa.pad_token}' (id: {tokenizer_docqa.pad_token_id})")
print(f"   EOS token: '{tokenizer_docqa.eos_token}' (id: {tokenizer_docqa.eos_token_id})")
print(f"   BOS token: '{tokenizer_docqa.bos_token}' (id: {tokenizer_docqa.bos_token_id})")

test_text = "What is the answer?"
test_encoded = tokenizer_docqa.encode(test_text)
test_decoded = tokenizer_docqa.decode(test_encoded)
print(f"Tokenizer test: '{test_text}' -> {test_encoded} -> '{test_decoded}'")

# Tạo datasets
train_ds_docqa = DocVQADataset(
    docvqa_dataset['train'],
    tokenizer=tokenizer_docqa,
    max_source_length=t5_config_docvqa.model_max_length,
    max_target_length=t5_config_docvqa.max_target_length,
    transform=transform_docqa,
    language='en',
    use_chunked_processing=t5_config_docvqa.use_chunked_processing,
    # use_case_augmentation=t5_config_docvqa.use_case_augmentation
)

val_ds_docqa = DocVQADataset(
    docvqa_dataset['validation'],
    tokenizer=tokenizer_docqa,
    max_source_length=t5_config_docvqa.model_max_length,
    max_target_length=t5_config_docvqa.max_target_length,
    transform=transform_docqa,
    language='en',
    use_chunked_processing=t5_config_docvqa.use_chunked_processing
)

test_ds_docqa = DocVQADataset(
    docvqa_dataset['test'],
    tokenizer=tokenizer_docqa,
    max_source_length=t5_config_docvqa.model_max_length,
    max_target_length=t5_config_docvqa.max_target_length,
    transform=transform_docqa,
    language='en',
    use_chunked_processing=t5_config_docvqa.use_chunked_processing
)

# Collate function với chunked processing
collate_fn_docqa = DocVQACollateFn(
    tokenizer_docqa,
    max_source_length=t5_config_docvqa.model_max_length,
    max_target_length=t5_config_docvqa.max_target_length,
    use_chunked_processing=t5_config_docvqa.use_chunked_processing
)

print(f"\n DATASET LOADERS:")
print(f"   • Train dataset: {len(train_ds_docqa)} samples")
print(f"   • Validation dataset: {len(val_ds_docqa)} samples")
print(f"   • Test dataset: {len(test_ds_docqa)} samples")
    
import os

run_id_file = os.path.join(CKPT_PATH_DOCQA, "wandb_run_id.txt")

if os.path.exists(run_id_file):
    # Resume case
    with open(run_id_file, 'r') as f:
        run_id = f.read().strip()
    print(f" Resuming wandb run: {run_id}")

    wandb.init(
        config=t5_config_docvqa,
        project="Arctic-TILT on DocVQA",
        id=run_id,
        resume="must",
        entity="xuanlinh-work-dut"
    )
else:
    # New training case
    print(f" Starting new wandb run")

    wandb.init(
        config=t5_config_docvqa,
        project="Arctic-TILT on DocVQA",
        entity="xuanlinh-work-dut"
    )

    # Save run_id
    os.makedirs(CKPT_PATH_DOCQA, exist_ok=True)
    with open(run_id_file, 'w') as f:
        f.write(wandb.run.id)

print(" Initializing Arctic-TILT model...")
model = TiLTDocQATransformer(t5_config_docvqa)

# Verify config
print(f"\n Model Configuration:")
print(f"   • Model: {MODEL_NAME}")
print(f"   • Max sequence length: {t5_config_docvqa.model_max_length}")
print(f"   • Chunk length: {t5_config_docvqa.core_chunk_length}")
print(f"   • Use chunked processing: {t5_config_docvqa.use_chunked_processing}")
print(f"   • Use post-fusion: {t5_config_docvqa.use_post_fusion}")

total_train_samples = len(train_ds_docqa)
batch_size = BATCH_SIZE
grad_acc_steps = GRAD_ACC_STEPS
num_epochs = t5_config_docvqa.max_epochs
total_steps = math.ceil(total_train_samples / batch_size / grad_acc_steps) * num_epochs

warmup_steps = max(1, int(0.01 * total_steps))
linear_steps = max(1, int(0.89 * total_steps))
cosine_steps = total_steps - warmup_steps - linear_steps

print(f"\n Training Schedule:")
print(f"   • Total steps: {total_steps}")
print(f"   • Warmup (1%): {warmup_steps} steps")
print(f"   • Linear decay (89%): {linear_steps} steps")
print(f"   • Cosine decay (10%): {cosine_steps} steps")

training_args = TrainingArguments(
    # Output
    output_dir=CKPT_PATH_DOCQA,
    overwrite_output_dir=True,

    # Training
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=grad_acc_steps,
    gradient_checkpointing=True,
    max_grad_norm = 1,

    # Optimization (Arctic-TILT settings)
    learning_rate=t5_config_docvqa.lr,  # 1e-3
    weight_decay=t5_config_docvqa.weight_decay,  # 1e-5

    # Mixed Precision
    bf16=True,
    # fp16=True,

    # Logging
    logging_dir=os.path.join(CKPT_PATH_DOCQA, "logs"),
    logging_strategy="steps",
    logging_steps=5,
    logging_first_step=True,

    # Evaluation
    eval_strategy="epoch",
    eval_delay=0,

    # Saving
    save_strategy="epoch",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="anls",
    greater_is_better=True,

    # Hardware
    dataloader_num_workers=8,
    dataloader_pin_memory=True,

    # Format
    remove_unused_columns=False,
    save_safetensors=False,

    # Reporting
    report_to=["wandb"],

    seed=42,
    data_seed=42,
)

print("\n Setting up Trainer...")

# Create custom optimizer
optimizer = create_optimizer(
    model,
    learning_rate=t5_config_docvqa.lr,
    weight_decay=t5_config_docvqa.weight_decay
)

# Create scheduler
lr_scheduler = get_arctic_tilt_scheduler(optimizer, total_steps)

# Initialize callbacks
validation_callback = ValidationMetricsCallback(
    tokenizer=tokenizer_docqa,
    compute_metrics_fn=compute_docqa_metrics
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds_docqa,
    eval_dataset=val_ds_docqa,
    data_collator=collate_fn_docqa,
    optimizers=(optimizer, lr_scheduler),
    callbacks=[validation_callback],
)

print(" Trainer initialized successfully!")

print("\n Starting training...")
print("="*60)

# Resume from checkpoint if exists
resume_from_checkpoint = None

if os.path.exists(CKPT_PATH_DOCQA):
    checkpoints = [d for d in os.listdir(CKPT_PATH_DOCQA)
                  if d.startswith("checkpoint-")]
    if checkpoints:
        latest_checkpoint = max(checkpoints,
                              key=lambda x: int(x.split("-")[-1]))
        resume_from_checkpoint = os.path.join(CKPT_PATH_DOCQA, latest_checkpoint)
        print(f" Resuming from checkpoint: {resume_from_checkpoint}")

# Train
result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

print("\n Training completed!")

final_model_dir = os.path.join(CKPT_PATH_DOCQA, "final_model")
print(f"\n Saving final model to {final_model_dir}...")
trainer.save_model(final_model_dir)
tokenizer_docqa.save_pretrained(final_model_dir)
print(f" Model saved successfully!")

summary_path = os.path.join(final_model_dir, "training_summary.json")
with open(summary_path, 'w') as f:
    json.dump({
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_val_anls": trainer.state.best_metric,
        "final_loss": result.training_loss,
        "total_steps": result.global_step,
        "num_epochs": num_epochs,
    }, f, indent=2)
