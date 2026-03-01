import torch.nn as nn
from .visual_backbone import Unet_encoder, RoIPool
from .t5_custom import LinearLayer

class VisualEmbedding(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.unet_encoder = Unet_encoder(in_channels = config.in_channels, channels = config.channels, num_pool_layers = config.num_pool_layers)
    self.roi_pool = RoIPool(output_size = config.output_size, spatial_scale = config.spatial_scale)
    # self.proj = nn.Linear(in_features = 128 * 3 * 3, out_features = config.d_model)
    self.proj = LinearLayer(in_features = 128 * 3 * 3, out_features = config.d_model)
    self.config = config

  def forward(self, pixel_values, bboxes):
    image_embedding = self.unet_encoder(pixel_values)
    feature_maps_bboxes = self.roi_pool(image_embedding, bboxes).flatten(2)
    projection = self.proj(feature_maps_bboxes)
    return projection