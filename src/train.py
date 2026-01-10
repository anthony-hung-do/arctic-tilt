import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.6"

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
    MODEL_NAME, CKPT_PATH_DOCQA, BATCH_SIZE, GRAD_ACC_STEPS, PRETRAIN_MAX_EPOCHS, PRETRAIN_BATCH_SIZE, PRETRAIN_GRAD_ACC_STEPS,
    get_t5_config, get_pretrain_config
)
from utils.auth import setup_authentication
from models.tilt_model import TiLTDocQATransformer
from data.dataset import DocVQADataset, PretrainDataset
from data.collate import DocVQACollateFn, PretrainCollateFn
from training.optimizer import create_optimizer
from training.scheduler import get_arctic_tilt_scheduler
from training.callbacks import ValidationMetricsCallback, ClearCacheCallback, MemoryMonitorCallback
from metrics import compute_docqa_metrics

setup_authentication()
device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================
# PRETRAIN FUNCTION
# ============================================
def pretrain_arctic_tilt(
    data_root: str,
    output_dir: str,
    config,
    tokenizer,
    num_epochs: int = 1,
    batch_size: int = 1,
    learning_rate: float = 1e-3,
    save_steps: int = 300,
    resume_from_checkpoint: str = None
):
    """
    Main pretraining function for Arctic-TILT using MLM
    """
    # Initialize wandb for pretraining
    run_id_file = os.path.join(output_dir, "wandb_pretrain_run_id.txt")
    if os.path.exists(run_id_file):
        with open(run_id_file, 'r') as f:
            run_id = f.read().strip()
        print(f"📊 Resuming wandb pretrain run: {run_id}")
        wandb.init(
            project="Arctic-TILT Pretrain MLM",
            id=run_id,
            resume="must",
            entity="xuanlinh-work-dut",
            config=config.__dict__ if hasattr(config, '__dict__') else config
        )
    else:
        print(f"📊 Starting new wandb pretrain run")
        wandb.init(
            project="Arctic-TILT Pretrain MLM",
            entity="xuanlinh-work-dut",
            config=config.__dict__ if hasattr(config, '__dict__') else config
        )
        os.makedirs(output_dir, exist_ok=True)
        with open(run_id_file, 'w') as f:
            f.write(wandb.run.id)

    print("="*80)
    print("🚀 ARCTIC-TILT PRETRAINING")
    print("="*80)
    
    # Add mask token if not exists
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({'mask_token': '<mask>'})
    
    print(f"\n📚 Tokenizer initialized:")
    print(f"   • Vocab size: {len(tokenizer)}")
    print(f"   • Mask token: '{tokenizer.mask_token}' (id: {tokenizer.mask_token_id})")
    
    # Create dataset
    print(f"\n📂 Loading pretrain dataset from {data_root}...")
    pretrain_dataset = PretrainDataset(
        data_root=data_root,
        tokenizer=tokenizer,
        max_length=config.model_max_length,
        mlm_probability=0.15,
        use_chunked_processing=config.use_chunked_processing,
        max_tokens_limit=5000
    )
    
    # Create data collator
    collate_fn = PretrainCollateFn(
        tokenizer=tokenizer,
        use_chunked_processing=config.use_chunked_processing
    )
    
    # Initialize model
    print(f"\n🏗️ Initializing Arctic-TILT model for pretraining...")
    model = TiLTDocQATransformer(config)
    
    print(f"   • Original vocab size: {model.t5_model.config.vocab_size}")
    print(f"   • Resizing to match tokenizer: {len(tokenizer)}")
    
    # Resize embeddings for T5 model
    model.t5_model.resize_token_embeddings(len(tokenizer))
    
    print(f"   • New vocab size: {model.t5_model.config.vocab_size}")
    print(f"   • Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Calculate training steps
    total_steps = math.ceil((len(pretrain_dataset) / batch_size / PRETRAIN_GRAD_ACC_STEPS) * num_epochs)
    # warmup_steps = int(0.01 * total_steps)
    
    print(f"\n📊 Pretraining configuration:")
    print(f"   • Total samples: {len(pretrain_dataset)}")
    print(f"   • Batch size: {batch_size}")
    print(f"   • Gradient accumulation steps: {PRETRAIN_GRAD_ACC_STEPS}")
    print(f"   • Epochs: {num_epochs}")
    print(f"   • Total steps: {total_steps}")
    print(f"   • Save checkpoint every: {save_steps} steps")
    print(f"   • Learning rate: {learning_rate}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=False,
        
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=PRETRAIN_GRAD_ACC_STEPS,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        
        learning_rate=learning_rate,
        weight_decay=1e-5,
        
        bf16=True,
        # fp16=True,

        logging_dir=os.path.join(output_dir, "logs"),
        logging_strategy="steps",
        logging_steps=10,
        logging_first_step=True,
        
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        
        remove_unused_columns=False,
        save_safetensors=False,
        
        report_to=["wandb"],
        
        seed=42,
        data_seed=42,
    )
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(
        model,
        learning_rate=learning_rate,
        weight_decay=1e-5
    )
    
    lr_scheduler = get_arctic_tilt_scheduler(optimizer, total_steps)

    memory_callback = MemoryMonitorCallback()
    clear_cache_callback = ClearCacheCallback()
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=pretrain_dataset,
        data_collator=collate_fn,
        optimizers=(optimizer, lr_scheduler),
        # callbacks=[memory_callback],
        callbacks=[memory_callback, clear_cache_callback],
    )
    
    # Start training
    print("\n" + "="*80)
    print("🎯 Starting pretraining...")
    print("="*80 + "\n")
    
    result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Save final model
    final_model_dir = os.path.join(output_dir, "final_model")
    print(f"\n💾 Saving pretrained model to {final_model_dir}...")
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    
    wandb.log({
        "pretrain/final_loss": result.training_loss,
        "pretrain/total_steps": result.global_step
    })

    wandb.finish()
    # Save training summary
    summary_path = os.path.join(final_model_dir, "pretrain_summary.json")
    with open(summary_path, 'w') as f:
        json.dump({
            "final_loss": result.training_loss,
            "total_steps": result.global_step,
            "num_epochs": num_epochs,
            "num_samples": len(pretrain_dataset),
        }, f, indent=2)
    
    print(f"\n✅ Pretraining completed!")
    print(f"   • Final loss: {result.training_loss:.4f}")
    print(f"   • Total steps: {result.global_step}")
    print(f"   • Model saved to: {final_model_dir}")
    
    return trainer, model, final_model_dir

hf_ds = load_dataset("nielsr/docvqa_1200_examples")

train_split = hf_ds['train'].train_test_split(test_size=0.1, seed=42)

docvqa_dataset = DatasetDict({
    'train': train_split['train'],
    'validation': train_split['test'],
    'test': hf_ds['test']
    # 'test': hf_ds['val']
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

# ============================================
# STEP 1: PRETRAINING (Optional)
# ============================================
ENABLE_PRETRAIN = os.getenv("ENABLE_PRETRAIN", "false").lower() == "true"
PRETRAIN_DATA_PATH = os.getenv("PRETRAIN_DATA_PATH", "../data")

if ENABLE_PRETRAIN and os.path.exists(PRETRAIN_DATA_PATH):
    print("\n" + "="*80)
    print("STEP 1: PRETRAINING WITH MLM")
    print("="*80)
    
    pretrain_output_dir = os.path.join(CKPT_PATH_DOCQA, "pretrain_output")
    pretrain_config = get_pretrain_config()

    # Check for existing checkpoints
    resume_pretrain = None
    if os.path.exists(pretrain_output_dir):
        pretrain_checkpoints = [d for d in os.listdir(pretrain_output_dir)
                                if d.startswith("checkpoint-")]
        if pretrain_checkpoints:
            latest_pretrain_checkpoint = max(pretrain_checkpoints,
                                            key=lambda x: int(x.split("-")[-1]))
            resume_pretrain = os.path.join(pretrain_output_dir, latest_pretrain_checkpoint)
            print(f"🔄 Resuming pretraining from checkpoint: {resume_pretrain}")

    # Run pretraining
    _, _, pretrained_model_path = pretrain_arctic_tilt(
        data_root=PRETRAIN_DATA_PATH,
        output_dir=pretrain_output_dir,
        # config=pretrain_config,
        config=t5_config_docvqa,
        tokenizer=tokenizer_docqa,
        num_epochs=PRETRAIN_MAX_EPOCHS,
        batch_size=PRETRAIN_BATCH_SIZE,
        learning_rate=1e-3,
        save_steps=100,
        resume_from_checkpoint=resume_pretrain
    )

    print(f"\n Loading pretrained weights for fine-tuning...")
    # Load pretrained model for fine-tuning
    t5_config_docvqa.update(dict(
        load_weights=False  # Disable auto-load for pretraining
    ))
    model = TiLTDocQATransformer(t5_config_docvqa)
    model.t5_model.resize_token_embeddings(len(tokenizer_docqa))
    
    pretrained_state = torch.load(
        os.path.join(pretrained_model_path, "pytorch_model.bin"),
        map_location=device
    )
    model.load_state_dict(pretrained_state, strict=False)
    print(f"   • Pretrained weights loaded successfully!")
else:
    # print("\n Pretraining disabled, using randomly initialized model")
    # model = TiLTDocQATransformer(t5_config_docvqa)
    pretrained_model_path = os.path.join(CKPT_PATH_DOCQA, "pretrain_output/checkpoint-100")
    tokenizer_docqa = AutoTokenizer.from_pretrained(pretrained_model_path)
    print("\n Pretraining disabled, Loading pretrained weights from ckpt model")
    t5_config_docvqa.update(dict(
        load_weights=False  # Disable auto-load for pretraining
    ))
    model = TiLTDocQATransformer(t5_config_docvqa)
    model.t5_model.resize_token_embeddings(len(tokenizer_docqa))
    
    pretrained_state = torch.load(
        os.path.join(pretrained_model_path, "pytorch_model.bin"),
        map_location=device
    )
    model.load_state_dict(pretrained_state, strict=False)
    print(f"   • Pretrained weights loaded successfully!")

# ============================================
# STEP 2: FINE-TUNING ON DocVQA
# ============================================
print("\n" + "="*80)
print("STEP 2: FINE-TUNING ON DocVQA")
print("="*80)
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

# print(" Initializing Arctic-TILT model...")
# model = TiLTDocQATransformer(t5_config_docvqa)

# # Verify config
# print(f"\n Model Configuration:")
# print(f"   • Model: {MODEL_NAME}")
# print(f"   • Max sequence length: {t5_config_docvqa.model_max_length}")
# print(f"   • Chunk length: {t5_config_docvqa.core_chunk_length}")
# print(f"   • Use chunked processing: {t5_config_docvqa.use_chunked_processing}")
# print(f"   • Use post-fusion: {t5_config_docvqa.use_post_fusion}")

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
print("\n🎉 All training completed!")