import torch.nn as nn
from .visual_embedding import VisualEmbedding
from .t5_custom import T5ForConditionalGeneration

# Import Transformer Engine for FP8 autocast
try:
    import transformer_engine.pytorch as te
    TE_AVAILABLE = True
except ImportError:
    TE_AVAILABLE = False

class TiLTDocQATransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.visual_embedding_extractor = VisualEmbedding(config)
        self.t5_model = T5ForConditionalGeneration(config)
        
        # Store FP8 config
        self.use_fp8 = getattr(config, 'use_fp8', False) and TE_AVAILABLE

    def common_step(self, batch):
        ## Visual embedding
        visual_embedding = self.visual_embedding_extractor(
            pixel_values=batch['pixel_values'],
            bboxes=batch['bboxes']
        )

        ## Semantic embedding from t5_model's embedding layer
        semantic_embedding = self.t5_model.shared(batch['input_ids'])

        # Arctic-TILT: Return both embeddings separately for fusion in each transformer block
        return semantic_embedding, visual_embedding

    def forward(self, batch=None, **kwargs):  # ← THÊM **kwargs
        """
        Forward pass with compatibility for both dict-based and HF Trainer calls

        Args:
            batch: Dict containing all inputs (for your custom code)
            **kwargs: Individual arguments (for HF Trainer compatibility)
        """
        # ============================================
        # COMPATIBILITY LAYER: Handle both input formats
        # ============================================
        if batch is None:
            # Called by HF Trainer with individual arguments
            # Reconstruct batch dict from kwargs
            batch = {
                'input_ids': kwargs.pop('input_ids', None),
                'attention_mask': kwargs.pop('attention_mask', None),
                'labels': kwargs.pop('labels', None),
                'pixel_values': kwargs.pop('pixel_values', None),
                'bboxes': kwargs.pop('bboxes', None),
            }

            # Remove any remaining kwargs that aren't needed
            kwargs.pop('past_key_values', None)
            kwargs.pop('cache_position', None)
            kwargs.pop('return_dict', None)
            kwargs.pop('use_cache', None)

        # Now proceed with normal forward logic
        embeddings, vision_embeddings = self.common_step(batch)

        # Arctic-TILT: Pass vision embeddings to be used in each transformer block
        # Wrap forward pass with FP8 autocast if enabled
        if self.use_fp8 and self.training:
            with te.fp8_autocast(enabled=True):
                final_output = self.t5_model(
                    attention_mask=batch['attention_mask'],
                    inputs_embeds=embeddings,
                    labels=batch['labels'],
                    vision_embeddings=vision_embeddings
                )
        else:
            final_output = self.t5_model(
                attention_mask=batch['attention_mask'],
                inputs_embeds=embeddings,
                labels=batch['labels'],
                vision_embeddings=vision_embeddings
            )

        return final_output

    def generate(self, batch, max_length=128, num_beams=4, early_stopping=True):
        """Generate answers for inference"""
        embeddings, vision_embeddings = self.common_step(batch)

        # Generate using T5's generate method
        generated_ids = self.t5_model.generate(
            attention_mask=batch['attention_mask'],
            inputs_embeds=embeddings,
            vision_embeddings=vision_embeddings,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=early_stopping,
            pad_token_id=self.t5_model.config.pad_token_id,
            eos_token_id=self.t5_model.config.eos_token_id,
            decoder_start_token_id=self.t5_model.config.pad_token_id,
            do_sample=False,  #  Tắt sampling để deterministic
            repetition_penalty=1.2,  #  Tránh lặp lại
            length_penalty=1.0,
            no_repeat_ngram_size=2  #  Tránh repeat ngrams
        )

        return generated_ids

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for the T5 model."""
        if hasattr(self.t5_model, 'gradient_checkpointing_enable'):
            self.t5_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        else:
            print("Warning: T5 model does not support gradient checkpointing.")