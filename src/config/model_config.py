# config/model_config.py
from transformers import AutoConfig

# Model constants
MODEL_NAME = "t5-base"
CKPT_PATH_DOCQA = "../ckpt"

# Visual Embedding parameters
IN_CHANNELS = 3
NUM_POOL_LAYERS = 3
CHANNELS = 16
SAMPLING_RATIO = 2
SPATIAL_SCALE = 48 / 384
OUTPUT_SIZE = (3, 3)

# Training parameters
DROPOUT_RATE = 0.2
LOAD_WEIGHTS = True
MODEL_MAX_LENGTH = 150000
PRETRAIN_MAX_LENGTH = 8000
MAX_TARGET_LENGTH = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
FUSION_DROPOUT = 0.1
CORE_CHUNK_LENGTH = 1024
USE_POST_FUSION = True
CHUNK_OVERLAP = 0
USE_CHUNKED_PROCESSING = True
PREFIX_LENGTH = 8

USE_FP8 = True
PAD_TO_MULTIPLE_OF = 16

ENABLE_CHUNK_DISCARD = False
CHUNK_DISCARD_RATIO = 0.6
ENABLE_CHUNK_DISCARD_V2 = False

PRETRAIN_MAX_EPOCHS = 1
PRETRAIN_BATCH_SIZE = 2
PRETRAIN_GRAD_ACC_STEPS = 64
    
MAX_EPOCHS = 10
BATCH_SIZE = 4
GRAD_ACC_STEPS = 2

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
        pad_to_multiple_of=PAD_TO_MULTIPLE_OF,
        task_type="document_qa",
        use_fp8=USE_FP8, 
        # use_case_augmentation = False,
        # randomly discard chunks
        enable_chunk_discard = ENABLE_CHUNK_DISCARD,
        chunk_discard_ratio = CHUNK_DISCARD_RATIO,
        # discard chunks v2
        enable_chunk_discard_v2 = ENABLE_CHUNK_DISCARD_V2,
    ))
    return config

def get_pretrain_config():
    """Return config for pretraining with lower max_length"""
    config = get_t5_config()
    # config.model_max_length = PRETRAIN_MAX_LENGTH
    config.enable_chunk_discard = False
    config.enable_chunk_discard_v2 = False
    return config