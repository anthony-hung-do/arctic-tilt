import os
import torch
import logging
from transformers import Trainer
from transformers.utils.import_utils import is_peft_available
from transformers.modeling_utils import PreTrainedModel

logger = logging.getLogger(__name__)

class CustomTrainer(Trainer):
    """
    Custom Trainer để tắt save_safetensors trong transformers 5.0.0+
    """
    
    # def _save(self, output_dir: str = None, state_dict=None):
    #     """
    #     Override save method để force .bin format
    #     """
    #     output_dir = output_dir if output_dir is not None else self.args.output_dir
    #     os.makedirs(output_dir, exist_ok=True)
        
    #     # Save model state dict as .bin
    #     if state_dict is None:
    #         state_dict = self.model.state_dict()
        
    #     torch.save(
    #         state_dict,
    #         os.path.join(output_dir, "pytorch_model.bin")
    #     )
    #     # if self.tokenizer is not None:
    #     #     self.tokenizer.save_pretrained(output_dir)
    #     # Save config
    #     if hasattr(self.model, "config"):
    #         self.model.config.save_pretrained(output_dir)
        
    #     # Save training args
    #     torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def _save(self, output_dir: str | None = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, supported_classes):
            if state_dict is None:
                state_dict = self.model.state_dict()

            if isinstance(self.accelerator.unwrap_model(self.model, keep_torch_compile=False), supported_classes):
                self.accelerator.unwrap_model(self.model, keep_torch_compile=False).save_pretrained(
                    output_dir, state_dict=state_dict
                )
            else:
                torch.save(
                        state_dict,
                        os.path.join(output_dir, "pytorch_model.bin")
                    )
        else:
            self.model.save_pretrained(output_dir, state_dict=state_dict)

        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        elif (
            self.data_collator is not None
            and hasattr(self.data_collator, "tokenizer")
            and self.data_collator.tokenizer is not None
        ):
            logger.info("Saving Trainer.data_collator.tokenizer by default as Trainer.processing_class is `None`")
            self.data_collator.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))