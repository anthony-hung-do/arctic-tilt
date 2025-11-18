from transformers import Trainer, TrainingArguments, TrainerCallback
import torch
from torch.utils.data import DataLoader
import wandb

class ValidationMetricsCallback(TrainerCallback):
    def __init__(self, tokenizer, compute_metrics_fn):
        self.tokenizer = tokenizer
        self.compute_metrics_fn = compute_metrics_fn

    def on_evaluate(self, args, state, control, model, eval_dataloader, metrics=None, **kwargs):
        if state.is_world_process_zero:
            print(f"\n VALIDATION METRICS (Epoch {state.epoch:.0f}):")

            model.eval()
            all_predictions = []
            all_references = []
            all_questions = []
            total_loss = 0.0
            num_batches = 0

            device = next(model.parameters()).device

            with torch.no_grad():
                for batch_idx, batch in enumerate(eval_dataloader):
                    # Move to device
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()}

                    outputs = model(batch)
                    total_loss += outputs.loss.item()
                    num_batches += 1

                    # Generate predictions cho metrics
                    generated_ids = model.generate(
                        batch,
                        max_length=128,
                        num_beams=4,
                        early_stopping=True
                    )

                    predictions = self.tokenizer.batch_decode(
                        generated_ids,
                        skip_special_tokens=True
                    )

                    all_predictions.extend(predictions)
                    all_references.extend([[answer] for answer in batch['answers']])
                    all_questions.extend(batch['questions'])

            # Compute eval_loss
            eval_loss = total_loss / num_batches

            # Compute custom metrics
            custom_metrics = self.compute_metrics_fn(all_predictions, all_references)

            # Print all metrics
            print(f"   • Eval Loss: {eval_loss:.4f}")
            print(f"   • Exact Match: {custom_metrics['exact_match']:.2f}%")
            print(f"   • F1 Score: {custom_metrics['f1']:.2f}%")
            print(f"   • ANLS: {custom_metrics['anls']:.4f}")

            #  Update metrics dict với cả eval_loss và custom metrics
            if metrics is not None:
                metrics.update({
                    "eval_loss": eval_loss,
                    "eval_exact_match": custom_metrics['exact_match'],
                    "eval_f1": custom_metrics['f1'],
                    "eval_anls": custom_metrics['anls']
                })

            # Print sample predictions
            print(f"\n SAMPLE VALIDATION PREDICTIONS (Epoch {state.epoch:.0f}):")
            print("="*80)
            num_samples = min(5, len(all_predictions))
            for i in range(num_samples):
                print(f"Sample {i+1}:")
                print(f"  Question: {all_questions[i]}")
                print(f"  Ground Truth: {all_references[i][0]}")
                print(f"  Prediction: {all_predictions[i]}")
                print("-"*80)

            # Log to wandb
            if args.report_to and "wandb" in args.report_to:
                wandb.log({
                    "val_loss": eval_loss,
                    "val_exact_match": custom_metrics['exact_match'],
                    "val_f1": custom_metrics['f1'],
                    "val_anls": custom_metrics['anls'],
                    "epoch": state.epoch
                })