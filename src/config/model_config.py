# config/model_config.py
from transformers import AutoConfig

# Model constants
MODEL_NAME = "t5-large"
CKPT_PATH_DOCQA = "../models"

# Visual Embedding parameters
IN_CHANNELS = 3
NUM_POOL_LAYERS = 3
CHANNELS = 16
SAMPLING_RATIO = 2
SPATIAL_SCALE = 48 / 384
OUTPUT_SIZE = (3, 3)

# Training parameters
DROPOUT_RATE = 0.1
LOAD_WEIGHTS = True
MODEL_MAX_LENGTH = 150000
MAX_TARGET_LENGTH = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
FUSION_DROPOUT = 0.1
CORE_CHUNK_LENGTH = 1024
USE_POST_FUSION = True
CHUNK_OVERLAP = 0
USE_CHUNKED_PROCESSING = True
PREFIX_LENGTH = 8

MAX_EPOCHS = 30
BATCH_SIZE = 2
GRAD_ACC_STEPS = 64

def get_t5_config():
    """Return configured T5 config"""
    config = AutoConfig.from_pretrained(MODEL_NAME)
    config.update(dict(
        dropout_rate=DROPOUT_RATE,
        in_channels=IN_CHANNELS,
        num_pool_layers=NUM_POOL_LAYERS,
        channels=CHANNELS,
        model_max_length=MODEL_MAX_LENGTH,
        max_target_length=MAX_TARGET_LENGTH,
        output_size=OUTPUT_SIZE,
        spatial_scale=SPATIAL_SCALE,
        sampling_ratio=SAMPLING_RATIO,
        use_cache=False,
        load_weights=LOAD_WEIGHTS,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        max_epochs=MAX_EPOCHS,
        use_post_fusion=USE_POST_FUSION,
        fusion_dropout=FUSION_DROPOUT,
        core_chunk_length=CORE_CHUNK_LENGTH,
        chunk_overlap=CHUNK_OVERLAP,
        use_chunked_processing=USE_CHUNKED_PROCESSING,
        prefix_length=PREFIX_LENGTH,
        task_type="document_qa",
    ))
    return config