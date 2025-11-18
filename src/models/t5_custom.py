import torch
import torch.nn as nn
import math
import copy
from transformers.models import t5
from transformers import AutoModel
import math
from typing import Optional, Tuple, List
import torch
import torch.nn as nn

class ChunkedProcessor:
    def __init__(self, core_chunk_length: int = 1024, chunk_overlap: int = 0, debug: bool = False):
        self.core_chunk_length = core_chunk_length
        self.chunk_overlap = chunk_overlap
        self.debug = debug
        self.sample_count = 0

    def create_chunks(self,
                input_ids: torch.Tensor = None,
                attention_mask: torch.Tensor = None,
                inputs_embeds: torch.Tensor = None,
                prefix_length: int = 1) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[int]]:

        if inputs_embeds is not None:
            batch_size, total_length, embed_dim = inputs_embeds.shape
        elif input_ids is not None:
            batch_size, total_length = input_ids.shape
        elif attention_mask is not None:
            batch_size, total_length = attention_mask.shape
        else:
            raise ValueError("At least one of input_ids, inputs_embeds, or attention_mask must be provided")

        # Debug info
        if self.debug and self.sample_count < 5:
            print(f"\n SAMPLE {self.sample_count + 1} - CREATE CHUNKS:")
            print(f"    Original sequence length: {total_length}")
            print(f"    Chunk length limit: {self.core_chunk_length}")
            print(f"    Prefix length: {prefix_length}")

        # If sequence is shorter than chunk length, return as single chunk
        if total_length <= self.core_chunk_length:
            if self.debug and self.sample_count < 5:
                print(f"    No chunking needed - sequence fits in one chunk")
                print(f"    Output: 1 chunk of length {total_length}")

            chunked_input_ids = [input_ids] if input_ids is not None else [None]
            chunked_embeds = [inputs_embeds] if inputs_embeds is not None else [None]
            return chunked_input_ids, [attention_mask], chunked_embeds, [0]

        chunked_input_ids = []
        chunked_attention_masks = []
        chunked_embeds = []
        chunk_info = []

        # Extract prefix for all modalities
        if prefix_length > 0:
            prefix_ids = input_ids[:, :prefix_length] if input_ids is not None else None
            prefix_mask = attention_mask[:, :prefix_length]
            prefix_embeds = inputs_embeds[:, :prefix_length, :] if inputs_embeds is not None else None

            # Content (after prefix)
            input_content = input_ids[:, prefix_length:] if input_ids is not None else None
            mask_content = attention_mask[:, prefix_length:]
            embeds_content = inputs_embeds[:, prefix_length:, :] if inputs_embeds is not None else None
        else:
            prefix_ids = None
            prefix_mask = None
            prefix_embeds = None
            input_content = input_ids
            mask_content = attention_mask
            embeds_content = inputs_embeds

        content_length = mask_content.shape[1]
        current_pos = 0
        chunk_count = 0

        if self.debug and self.sample_count < 5:
            print(f"    Content length (after prefix): {content_length}")
            print(f"    Will create chunks:")

        while current_pos < content_length:
            # Calculate chunk boundaries (excluding prefix length)
            chunk_size = self.core_chunk_length - prefix_length if prefix_length > 0 else self.core_chunk_length
            chunk_end = min(current_pos + chunk_size, content_length)

            # Extract content chunks
            mask_chunk = mask_content[:, current_pos:chunk_end]

            if input_content is not None:
                content_chunk = input_content[:, current_pos:chunk_end]
            else:
                content_chunk = None

            if embeds_content is not None:
                embeds_chunk = embeds_content[:, current_pos:chunk_end, :]
            else:
                embeds_chunk = None

            # Prepend prefix to chunks
            if prefix_mask is not None:
                chunk_attention_mask = torch.cat([prefix_mask, mask_chunk], dim=1)
            else:
                chunk_attention_mask = mask_chunk

            if prefix_ids is not None and content_chunk is not None:
                chunk_input_ids = torch.cat([prefix_ids, content_chunk], dim=1)
            else:
                chunk_input_ids = content_chunk

            if prefix_embeds is not None and embeds_chunk is not None:
                chunk_embeds = torch.cat([prefix_embeds, embeds_chunk], dim=1)
            else:
                chunk_embeds = embeds_chunk

            final_chunk_length = chunk_embeds.shape[1] if chunk_embeds is not None else chunk_attention_mask.shape[1]

            if self.debug and self.sample_count < 5:
                print(f"       Chunk {chunk_count}: pos[{current_pos}:{chunk_end}] + prefix({prefix_length}) = {final_chunk_length} tokens")

            chunked_input_ids.append(chunk_input_ids)
            chunked_attention_masks.append(chunk_attention_mask)
            chunked_embeds.append(chunk_embeds)
            chunk_info.append(current_pos)

            # Move to next chunk
            if self.chunk_overlap > 0:
                current_pos = chunk_end - self.chunk_overlap
            else:
                current_pos = chunk_end

            chunk_count += 1

            if chunk_end >= content_length:
                break

        if self.debug and self.sample_count < 5:
            print(f"    Total chunks created: {len(chunked_embeds)}")

        return chunked_input_ids, chunked_attention_masks, chunked_embeds, chunk_info

    def align_embeddings_for_chunk(self, text_chunk: torch.Tensor, vision_embeddings: torch.Tensor, attention_mask: torch.Tensor, chunk_idx: int = 0):
        """
        Align text chunk với vision embeddings và track alignment strategy
        """
        batch_size = text_chunk.shape[0]
        text_seq_len = text_chunk.shape[1]
        vision_seq_len = vision_embeddings.shape[1]
        d_model = text_chunk.shape[2]
        device = text_chunk.device

        # Track what actually happened
        alignment_info = {
            'original_text_length': text_seq_len,
            'original_vision_length': vision_seq_len,
            'final_length': max(text_seq_len, vision_seq_len),
            'text_was_padded': False,
            'vision_was_padded': False
        }

        if self.debug and self.sample_count < 5:
            print(f"       ALIGNMENT Chunk {chunk_idx}:")
            print(f"          Text length: {text_seq_len}")
            print(f"           Vision length: {vision_seq_len}")

        if text_seq_len == vision_seq_len:
            if self.debug and self.sample_count < 5:
                print(f"          Lengths match - no alignment needed")
            return text_chunk, vision_embeddings, attention_mask, alignment_info

        if text_seq_len > vision_seq_len:
            # Pad vision embeddings to match text chunk length
            pad_length = text_seq_len - vision_seq_len
            vision_padding = torch.zeros(batch_size, pad_length, d_model,
                                      device=device, dtype=vision_embeddings.dtype)
            aligned_vision_chunk = torch.cat([vision_embeddings, vision_padding], dim=1)
            aligned_text_chunk = text_chunk
            aligned_attention_mask = attention_mask

            alignment_info['vision_was_padded'] = True
            alignment_info['final_length'] = text_seq_len

            if self.debug and self.sample_count < 5:
                print(f"          VISION PADDED: {vision_seq_len} → {text_seq_len} (+{pad_length})")

        else:
            # Pad text chunk and attention mask to match vision length
            pad_length = vision_seq_len - text_seq_len
            text_padding = torch.zeros(batch_size, pad_length, d_model,
                                    device=device, dtype=text_chunk.dtype)
            mask_padding = torch.zeros(batch_size, pad_length, device=device, dtype=attention_mask.dtype)

            aligned_text_chunk = torch.cat([text_chunk, text_padding], dim=1)
            aligned_vision_chunk = vision_embeddings
            aligned_attention_mask = torch.cat([attention_mask, mask_padding], dim=1)

            alignment_info['text_was_padded'] = True
            alignment_info['final_length'] = vision_seq_len

            if self.debug and self.sample_count < 5:
                print(f"          TEXT PADDED: {text_seq_len} → {vision_seq_len} (+{pad_length})")

        return aligned_text_chunk, aligned_vision_chunk, aligned_attention_mask, alignment_info

    def process_long_sequence(self,
                        encoder_func,
                        input_ids: torch.Tensor = None,
                        attention_mask: torch.Tensor = None,
                        inputs_embeds: torch.Tensor = None,
                        vision_embeddings: torch.Tensor = None,
                        prefix_length: int = 1,
                        **encoder_kwargs) -> torch.Tensor:
        """
        Process long sequence using chunked processing with alignment between text chunks and vision embeddings.
        """

        if self.debug and self.sample_count < 5:
            print(f"\n SAMPLE {self.sample_count + 1} - PROCESS LONG SEQUENCE:")
            if inputs_embeds is not None:
                print(f"    Input text embeddings shape: {inputs_embeds.shape}")
            if vision_embeddings is not None:
                print(f"     Input vision embeddings shape: {vision_embeddings.shape}")

        # Create chunks for TEXT only
        chunked_input_ids, chunked_masks, chunked_embeds, chunk_info = self.create_chunks(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            prefix_length=prefix_length
        )

        # Process each chunk with aligned embeddings
        chunk_outputs = []
        for chunk_idx, (chunk_input, chunk_mask, chunk_embeds) in enumerate(
            zip(chunked_input_ids, chunked_masks, chunked_embeds)
        ):
            if self.debug and self.sample_count < 5:
                print(f"\n    PROCESSING CHUNK {chunk_idx}:")

            # Prepare encoder arguments for this chunk
            encoder_kwargs_copy = encoder_kwargs.copy()

            alignment_info = None

            if chunk_embeds is not None and vision_embeddings is not None:
                aligned_text_chunk, aligned_vision_chunk, aligned_attention_mask, alignment_info = self.align_embeddings_for_chunk(
                    text_chunk=chunk_embeds, vision_embeddings=vision_embeddings, attention_mask=chunk_mask, chunk_idx=chunk_idx
                )

                encoder_kwargs_copy.update({
                    'inputs_embeds': aligned_text_chunk,
                    'attention_mask': aligned_attention_mask,
                    'vision_embeddings': aligned_vision_chunk,
                    'input_ids': None
                })

                if self.debug and self.sample_count < 5:
                    print(f"       Input to encoder: text={aligned_text_chunk.shape[1]}, vision={aligned_vision_chunk.shape[1]}")
            else:
                # No alignment needed
                encoder_kwargs_copy.update({
                    'input_ids': chunk_input,
                    'attention_mask': chunk_mask,
                    'vision_embeddings': vision_embeddings
                })

            # Process chunk with aligned embeddings
            chunk_output = encoder_func(**encoder_kwargs_copy)
            chunk_hidden = chunk_output.last_hidden_state if hasattr(chunk_output, 'last_hidden_state') else chunk_output

            if self.debug and self.sample_count < 5:
                print(f"       Raw encoder output shape: {chunk_hidden.shape}")

            # SMART TRIMMING: Only trim if TEXT was padded
            if alignment_info is not None and alignment_info['text_was_padded']:
                # TEXT được pad, nên cần trim về độ dài text gốc
                original_text_length = alignment_info['original_text_length']
                chunk_hidden = chunk_hidden[:, :original_text_length, :]

                if self.debug and self.sample_count < 5:
                    print(f"        TRIMMED (text was padded): {alignment_info['final_length']} → {original_text_length}")

            elif alignment_info is not None and alignment_info['vision_was_padded']:
                # VISION được pad, giữ nguyên output length = text length
                if self.debug and self.sample_count < 5:
                    print(f"       KEPT (vision was padded): {chunk_hidden.shape[1]} tokens")
                pass
            else:
                if self.debug and self.sample_count < 5:
                    print(f"       KEPT (no alignment): {chunk_hidden.shape[1]} tokens")

            if self.debug and self.sample_count < 5:
                print(f"       Final chunk output shape: {chunk_hidden.shape}")

            # No alignment case - keep as is
            chunk_outputs.append(chunk_hidden)

            del chunk_hidden
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

        # Recombine chunks
        if self.debug and self.sample_count < 5:
            print(f"\n    RECOMBINING CHUNKS:")
            for i, chunk in enumerate(chunk_outputs):
                print(f"       Chunk {i} shape: {chunk.shape}")

        final_output = self.recombine_chunk_outputs(chunk_outputs, prefix_length)

        if self.debug and self.sample_count < 5:
            print(f"    FINAL OUTPUT SHAPE: {final_output.shape}")
            print(f"   " + "="*60)
            self.sample_count += 1

        return final_output

    def recombine_chunk_outputs(self,
                              chunk_outputs: List[torch.Tensor],
                              prefix_length: int = 1) -> torch.Tensor:
        """
        Recombine chunk outputs by removing prefix tokens from all but the first chunk
        and concatenating the results, as described in Arctic-TILT paper.
        """
        if len(chunk_outputs) == 1:
            if self.debug and self.sample_count < 5:
                print(f"       Single chunk - no recombination needed")
            return chunk_outputs[0]

        combined_chunks = []

        for i, chunk_output in enumerate(chunk_outputs):
            if i == 0:
                # Keep first chunk completely
                combined_chunks.append(chunk_output)
                if self.debug and self.sample_count < 5:
                    print(f"       Chunk 0: KEEP ALL {chunk_output.shape[1]} tokens")
            else:
                # Remove prefix embeddings from subsequent chunks
                if prefix_length > 0:
                    chunk_without_prefix = chunk_output[:, prefix_length:, :]
                    combined_chunks.append(chunk_without_prefix)
                    if self.debug and self.sample_count < 5:
                        print(f"       Chunk {i}: REMOVE PREFIX {chunk_output.shape[1]} → {chunk_without_prefix.shape[1]} tokens")
                else:
                    combined_chunks.append(chunk_output)
                    if self.debug and self.sample_count < 5:
                        print(f"       Chunk {i}: KEEP ALL {chunk_output.shape[1]} tokens")

        # Concatenate all chunks along sequence dimension
        combined_output = torch.cat(combined_chunks, dim=1)

        if self.debug and self.sample_count < 5:
            total_tokens = sum(chunk.shape[1] for chunk in combined_chunks)
            print(f"       Concatenated: {len(combined_chunks)} chunks → {total_tokens} tokens")

        return combined_output

class TiltLayerNorm(nn.Module):
    """
    This is essentially the T5 modification of layer norm, referred to as RMS norm.

    Args:
        dim: the dimension of vectors to be normalized, i.e. the last dimension of
             the input tensor
        eps: small positive value added to computed second moment for numerical
             stability
    """
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.init_weights()

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        dtype = inp.dtype
        x = inp.to(torch.float32)
        squared_norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(squared_norm + self.eps)
        return self.w * x.to(dtype)

    def init_weights(self, factor: float = 1.0) -> None:
        self.w.data.fill_(factor * 1.0)

class TiltPostFusionModule(nn.Module):
    """
    Introduced in the Arctic-TILT paper.

    Args:
        d_model: dimension of input vectors
        dropout: probability of dropout applied to input embeddings
        layer_norm: the module responsible for input embeddings
    """

    def __init__(self, d_model: int, dropout: float, layer_norm: TiltLayerNorm):
        super().__init__()
        self.layer_norm = layer_norm
        self.to_v = nn.Linear(d_model, d_model, bias=False)
        self.to_out = nn.Linear(d_model, d_model, bias=False)
        self.to_r = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_queries: torch.Tensor, image_queries: torch.Tensor) -> torch.Tensor:
        """
        Compute module’s forward pass.

        Args:
            text_queries (Tensor): Tensor representing the primary input in the
                                  fusion, which is text-based, or mixed.
            image_queries (Tensor): Tensor representing the secondary input in the
                                   fusion, which is image-based.
        """
        bs, text_len, d = text_queries.shape
        _, img_len, _ = image_queries.shape

        # Handle sequence length mismatch
        if text_len != img_len:
            raise ValueError(f"Embeddings should be aligned before PostFusion! text={text_len}, vision={img_len}")

        bs, l, d = text_queries.shape
        inputs = torch.stack([text_queries, image_queries], dim=-2)
        inputs = inputs.view(bs * l, 2, d)
        normed_inputs = self.dropout(self.layer_norm(inputs))
        normed_primary_input = normed_inputs[:, 0]
        out: Tensor = self.to_v(normed_inputs.sum(-2))
        out = out + out * self.to_r(normed_primary_input)
        out = self.to_out(out)
        out = out.view(bs, l, d)
        return text_queries + out

class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 Style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):

        # T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32

        variance = hidden_states.to(torch.float32).pow(
            2).mean(-1, keepdim=True)
        hidden_states = hidden_states * \
            torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class T5DenseActDense(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = nn.ReLU()

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if hidden_states.dtype != self.wo.weight.dtype and self.wo.weight.dtype != torch.int8:
            hidden_states = hidden_states.to(self.wo.weight.dtype)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5DenseGatedActDense(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = nn.ReLU()

    def forward(self, hidden_states):
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        if hidden_states.dtype != self.wo.weight.dtype and self.wo.weight.dtype != torch.int8:
            hidden_states = hidden_states.to(self.wo.weight.dtype)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5LayerFF(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.is_gated_act:
            self.DenseReluDense = T5DenseGatedActDense(config)
        else:
            self.DenseReluDense = T5DenseActDense(config)

        self.layer_norm = T5LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class T5Attention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        '''
        Here is where the change lies, i.e adding the relative_horizontal_bias as well as the relative_vertical_bias
        '''
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(
                self.relative_attention_num_buckets, self.n_heads)
            self.relative_horizontal_bias = nn.Embedding(
                self.relative_attention_num_buckets, self.n_heads)
            self.relative_vertical_bias = nn.Embedding(
                self.relative_attention_num_buckets, self.n_heads)

        self.gradient_checkpointing = False

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on
        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer
        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position
                                 > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = - \
                torch.min(relative_position,
                          torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(
                relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small,
                                        relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias_1d(self, query_length, key_length, device=None):
        """Compute binned relative position bias"""
        if device is None:
            device = self.relative_attention_bias.weight.device
        context_position = torch.arange(
            query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(
            key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - \
            context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        # shape (query_length, key_length, num_heads)
        values = self.relative_attention_bias(relative_position_bucket)
        # shape (1, num_heads, query_length, key_length)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        return values

    def compute_vertical_horizontal_bias(self, total_boxes: int = 512, device=None):

        denominator_to_divide = total_boxes // self.relative_attention_num_buckets

        """Compute the vertical and horizontal bias"""
        if device is None:
            device = self.relative_attention_bias.weight.device
        indices = torch.arange(total_boxes, dtype=torch.long, device=device)
        h_distances = (indices % self.relative_attention_num_buckets)[
            :, None] - (indices % self.relative_attention_num_buckets)[None, :]
        v_distances = (
            indices // denominator_to_divide)[:, None] - (indices // denominator_to_divide)[None, :]

        h_distances_bucket = self._relative_position_bucket(
            h_distances,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )

        ## It has to be like this : https://github.com/microsoft/i-Code/blob/d933ae53eb9dec057e605fa4c89ea701629c5b9d/i-Code-Doc/core/models/embedding/relative/relative.py#L175
        ## so change is needed here
        v_distances_bucket = self._relative_position_bucket(
            v_distances,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )

        h_distances_values = self.relative_horizontal_bias(
            h_distances_bucket)  # shape (query_length, key_length, num_heads)
        h_distances_values = h_distances_values.permute([2, 0, 1]).unsqueeze(
            0)  # shape (1, num_heads, query_length, key_length)

        v_distances_values = self.relative_vertical_bias(
            v_distances_bucket)  # shape (query_length, key_length, num_heads)
        v_distances_values = v_distances_values.permute([2, 0, 1]).unsqueeze(
            0)  # shape (1, num_heads, query_length, key_length)

        return h_distances_values, v_distances_values

    def forward(self, hidden_states, mask=None, key_value_states=None, position_bias=None, past_key_value=None, layer_head_mask=None, query_length=None,
                use_cache=False, output_attentions=False):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)

        past_key_value = None
        use_cache = False

        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            assert(len(past_key_value)
                   == 2), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
            real_seq_length += past_key_value[0].shape[2] if query_length is None else key_value_states.shape[1]

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[
            1]

        def shape(states):
            "projection"
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """project hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat(
                        [past_key_value, hidden_states], dim=2)
                elif past_key_value.shape[2] != key_value_states.shape[1]:
                    # checking that the `sequence_length` of the `past_key_value` is the same as
                    # the provided `key_value_states` to support prefix tuning
                    # cross-attn
                    # (batch_size, n_heads, seq_length, dim_per_head)
                    hidden_states = shape(proj_layer(key_value_states))
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(self.q(hidden_states))

        # get key/value states
        key_states = project(hidden_states, self.k, key_value_states,
                             past_key_value[0] if past_key_value is not None else None)
        value_states = project(hidden_states, self.v, key_value_states,
                               past_key_value[0] if past_key_value is not None else None)

        # compute score
        # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9
        scores = torch.matmul(query_states, key_states.transpose(3, 2))

        # Sequential Part
        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias_1d(
                    real_seq_length, key_length, device=scores.device)
                h_distances_values, v_distances_values = self.compute_vertical_horizontal_bias(
                    total_boxes=real_seq_length, device=scores.device)
                position_bias = position_bias + h_distances_values + v_distances_values

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1):, :]

            if mask is not None:

                # Handle different mask formats properly
                if mask.dim() == 2:
                    # Standard attention mask: (batch_size, seq_len)
                    # Convert to 4D: (batch_size, 1, 1, seq_len)
                    extended_mask = mask.unsqueeze(1).unsqueeze(1)
                elif mask.dim() == 3:
                    # 3D mask: (batch_size, 1, seq_len) or (batch_size, seq_len, seq_len)
                    if mask.shape[1] == 1:
                        # (batch_size, 1, seq_len) -> (batch_size, 1, 1, seq_len)
                        extended_mask = mask.unsqueeze(1)
                    else:
                        # (batch_size, seq_len, seq_len) -> (batch_size, 1, seq_len, seq_len)
                        extended_mask = mask.unsqueeze(1)
                elif mask.dim() == 4:
                    # Already 4D format
                    extended_mask = mask
                elif mask.dim() == 5:
                    # Remove extra dimensions: (batch, 1, ?, heads, seq, seq) -> (batch, heads, seq, seq)
                    extended_mask = mask.squeeze(1).squeeze(1) if mask.shape[1] == 1 and mask.shape[2] == 1 else mask
                    if extended_mask.dim() == 4 and extended_mask.shape[1] == 1:
                        extended_mask = extended_mask.expand(-1, self.n_heads, -1, -1)
                elif mask.dim() == 6:
                    # 6D mask: (batch, 1, 1, 1, seq, seq)
                    # Squeeze all singleton dimensions except batch and last two (seq dims)
                    extended_mask = mask.squeeze(1).squeeze(1).squeeze(1)  # -> (batch, seq, seq)
                    # Add head dimension: (batch, seq, seq) -> (batch, 1, seq, seq)
                    extended_mask = extended_mask.unsqueeze(1)
                    # Expand to all heads if needed
                    if extended_mask.shape[1] == 1 and self.n_heads > 1:
                        extended_mask = extended_mask.expand(-1, self.n_heads, -1, -1)
                else:
                    raise ValueError(f"Unsupported mask dimension: {mask.dim()}, shape: {mask.shape}")

                # Ensure extended_mask matches position_bias shape
                target_shape = (position_bias.shape[0], position_bias.shape[1], position_bias.shape[2], position_bias.shape[3])

                # Expand to match target shape if needed
                if extended_mask.shape != target_shape:
                    # Handle batch dimension
                    if extended_mask.shape[0] == 1 and target_shape[0] != 1:
                        extended_mask = extended_mask.expand(target_shape[0], -1, -1, -1)
                    elif extended_mask.shape[0] != target_shape[0]:
                        extended_mask = extended_mask[:target_shape[0]]

                    # Handle head dimension
                    if extended_mask.shape[1] == 1 and target_shape[1] != 1:
                        extended_mask = extended_mask.expand(-1, target_shape[1], -1, -1)

                    # Handle sequence dimensions
                    if extended_mask.shape[2] != target_shape[2] or extended_mask.shape[3] != target_shape[3]:
                        extended_mask = extended_mask[..., :target_shape[2], :target_shape[3]]

                # Ensure position_bias has correct batch size
                if position_bias.shape[0] == 1 and target_shape[0] != 1:
                    position_bias = position_bias.expand(target_shape[0], -1, -1, -1)

                # Add mask to position bias
                position_bias = position_bias + extended_mask

        position_bias_masked = position_bias  # No pruning right now

        if scores.shape != position_bias_masked.shape:
            # Ensure position_bias matches scores shape (batch_size, num_heads, seq_len, seq_len)
            target_shape = scores.shape

            # Expand batch dimension if needed
            if position_bias_masked.shape[0] == 1 and target_shape[0] != 1:
                position_bias_masked = position_bias_masked.expand(target_shape[0], -1, -1, -1)

            # Expand head dimension if needed
            if position_bias_masked.shape[1] == 1 and target_shape[1] != 1:
                position_bias_masked = position_bias_masked.expand(-1, target_shape[1], -1, -1)

            # Trim sequence dimensions if needed
            if position_bias_masked.shape[2] != target_shape[2] or position_bias_masked.shape[3] != target_shape[3]:
                position_bias_masked = position_bias_masked[:, :, :target_shape[2], :target_shape[3]]

        scores += position_bias_masked
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        # (batch_size, seq_length, dim)
        attn_output = unshape(torch.matmul(attn_weights, value_states))
        attn_output = self.o(attn_output)

        present_key_value_state = (key_states, value_states) if (
            self.is_decoder and use_cache) else None
        outputs = (attn_output,) + \
            (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs



class T5LayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.SelfAttention = T5Attention(
            config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, attention_mask=None, position_bias=None, layer_head_mask=None, past_key_value=None, use_cache=False, output_attentions=False):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(normed_hidden_states, mask=attention_mask, position_bias=position_bias,
                                              layer_head_mask=layer_head_mask, past_key_value=past_key_value, use_cache=use_cache, output_attentions=output_attentions,)
        hidden_states = hidden_states + self.dropout(attention_output[0])
        # add attentions if we output them
        outputs = (hidden_states,) + attention_output[1:]
        return outputs


class T5LayerCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.EncDecAttention = T5Attention(
            config, has_relative_attention_bias=False)
        self.layer_norm = T5LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, key_value_states, attention_mask=None, position_bias=None, layer_head_mask=None, past_key_value=None, use_cache=False, query_length=None, output_attentions=False, ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(normed_hidden_states, mask=attention_mask,
                                                key_value_states=key_value_states, position_bias=position_bias,
                                                layer_head_mask=layer_head_mask,
                                                past_key_value=past_key_value,
                                                use_cache=use_cache,
                                                query_length=query_length,
                                                output_attentions=output_attentions,)
        layer_output = hidden_states + self.dropout(attention_output[0])
        # add attention if we output them
        outputs = (layer_output, ) + attention_output[1:]
        return outputs


import torch.utils.checkpoint as checkpoint

class T5Block(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False, layer_index=0):  # add layer_index parameter
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer_index = layer_index  # Store layer index
        self.total_layers = getattr(config, 'num_layers', 12)

        self.layer = nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(
            config, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config))

        self.layer.append(T5LayerFF(config))

        # Arctic-TILT fusion module
        fusion_layer_norm = TiltLayerNorm(config.d_model)
        fusion_dropout = getattr(config, 'fusion_dropout', 0.1)
        self.post_fusion = TiltPostFusionModule(
            d_model=config.d_model,
            dropout=fusion_dropout,
            layer_norm=fusion_layer_norm
        )

    def forward(self, hidden_states, attention_mask=None, position_bias=None, encoder_hidden_states=None,
                encoder_attention_mask=None, encoder_decoder_position_bias=None, layer_head_mask=None, cross_attn_layer_head_mask=None,
                past_key_value=None, past_key_values=None, use_cache=False, output_attentions=False, return_dict=True, vision_embeddings=None,cache_position=None):

        if past_key_values is not None and past_key_value is None:
            past_key_value = past_key_values


        past_key_value = None
        use_cache = False
        # NESTED STACK CHECKPOINTING:
        use_checkpointing = (
            not self.is_decoder and  # Chỉ trong encoder
            self.layer_index < self.total_layers - 1 and  # Not last layer
            self.training and  # training
            past_key_value is None
        )

        if use_checkpointing:
            # Checkpoint if not last layers: Don't save activations
            return self._forward_with_checkpoint(
                hidden_states, attention_mask, position_bias, encoder_hidden_states,
                encoder_attention_mask, encoder_decoder_position_bias, layer_head_mask,
                cross_attn_layer_head_mask, past_key_value, use_cache, output_attentions,
                return_dict, vision_embeddings
            )
        else:
            # Last layer or decoder: Save activations
            return self._forward_normal(
                hidden_states, attention_mask, position_bias, encoder_hidden_states,
                encoder_attention_mask, encoder_decoder_position_bias, layer_head_mask,
                cross_attn_layer_head_mask, past_key_value, use_cache, output_attentions,
                return_dict, vision_embeddings
            )

    def _forward_with_checkpoint(self, hidden_states, attention_mask, position_bias, encoder_hidden_states,
                               encoder_attention_mask, encoder_decoder_position_bias, layer_head_mask,
                               cross_attn_layer_head_mask, past_key_value, use_cache, output_attentions,
                               return_dict, vision_embeddings):
        """Forward with checkpoint - save memory"""

        def create_custom_forward():
            def custom_forward(*inputs):
                return self._forward_normal(*inputs)
            return custom_forward

        # Sử dụng checkpoint cho forward pass
        return checkpoint.checkpoint(
            create_custom_forward(),
            hidden_states, attention_mask, position_bias, encoder_hidden_states,
            encoder_attention_mask, encoder_decoder_position_bias, layer_head_mask,
            cross_attn_layer_head_mask, past_key_value, use_cache, output_attentions,
            return_dict, vision_embeddings,
            use_reentrant=False
        )

    def _forward_normal(self, hidden_states, attention_mask=None, position_bias=None, encoder_hidden_states=None,
                       encoder_attention_mask=None, encoder_decoder_position_bias=None, layer_head_mask=None,
                       cross_attn_layer_head_mask=None, past_key_value=None, use_cache=False, output_attentions=False,
                       return_dict=True, vision_embeddings=None):
        """Forward bình thường"""

        past_key_value = None
        use_cache = False

        if past_key_value is not None:
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4
            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"Got {len(past_key_value)} past key / value states"
                )
            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]

        # Clamp inf values
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            query_length = present_key_value_state[0].shape[2] if present_key_value_state is not None else None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]

            if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # Arctic-TILT: Apply post-fusion when vision embeddings are provided
        if vision_embeddings is not None:
            hidden_states = self.post_fusion(hidden_states, vision_embeddings)

        # Clamp inf values
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return outputs


class T5Stack(t5.modeling_t5.T5Stack):
    def __init__(self, config, embed_tokens=None):
        '''Arctic-TILT implementation with custom T5Block and chunked processing'''
        super().__init__(config=config, embed_tokens=embed_tokens)
        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0), layer_index=i)
             for i in range(config.num_layers)]
        )

        # Initialize chunked processor for Arctic-TILT
        if getattr(config, 'use_chunked_processing', False):
            self.chunked_processor = ChunkedProcessor(
                core_chunk_length=getattr(config, 'core_chunk_length', 1024),
                chunk_overlap=getattr(config, 'chunk_overlap', 0),
                debug = False
            )
        else:
            self.chunked_processor = None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        vision_embeddings=None,
        cache_position=None,
    ):
        use_cache = False
        past_key_values = None

        use_chunked = (
            self.chunked_processor is not None and
            not self.is_decoder  # Only for encoder
        )

        if use_chunked:
            # Use chunked processing in encoder
            return self._forward_chunked(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                vision_embeddings=vision_embeddings,
            )
        else:
            # Use standard processing
            return self._forward_standard(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                vision_embeddings=vision_embeddings,
            )

    def _forward_chunked(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        vision_embeddings=None,
    ):
        """
        Forward pass with chunked processing using Arctic-TILT approach.

        This implements the exact SLED approach described in the paper:
        - Divide sequence into chunks with prefix
        - Apply DENSE attention within each chunk
        - Recombine by removing prefix from all but first chunk
        """
        # Set defaults
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape

            # Create attention_mask if not provided
            if attention_mask is None:
                attention_mask = torch.ones((batch_size, seq_length), device=inputs_embeds.device)

        # Align vision and text embeddings
        # if vision_embeddings is not None and inputs_embeds is not None:
        #     if vision_embeddings.shape[1] != inputs_embeds.shape[1]:
        #         print(f"  Vision-text length mismatch: {vision_embeddings.shape[1]} vs {inputs_embeds.shape[1]}")
        #         # Align lengths
        #         min_len = min(vision_embeddings.shape[1], inputs_embeds.shape[1])
        #         vision_embeddings = vision_embeddings[:, :min_len, :]
        #         inputs_embeds = inputs_embeds[:, :min_len, :]
        #         if attention_mask is not None:
        #             attention_mask = attention_mask[:, :min_len]

        # Arctic-TILT: Use chunked processing with dense attention
        combined_output = self.chunked_processor.process_long_sequence(
            encoder_func=lambda **kwargs: self._forward_standard(**kwargs),
            input_ids=None,  # Force use inputs_embeds
            attention_mask=attention_mask,
            prefix_length=getattr(self.config, 'prefix_length', 1),
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            use_cache=False,  # Disable cache for chunked processing
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            vision_embeddings=vision_embeddings,
        )

        # Return in appropriate format
        if return_dict:
            from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
            return BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=combined_output,
                past_key_values=None,
                hidden_states=None,
                attentions=None,
                cross_attentions=None,
            )
        else:
            return (combined_output,)

    def _forward_standard(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        vision_embeddings=None,
    ):
        """
        Standard forward pass for T5Stack with vision embeddings fusion.
        """
        use_cache = False
        past_key_values = None
        # Arctic-TILT: Custom forward pass with vision_embeddings passed to each block
        if vision_embeddings is not None:
            # Prepare inputs similar to original T5Stack forward
            use_cache = use_cache if use_cache is not None else self.config.use_cache
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            if input_ids is not None and inputs_embeds is not None:
                raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
            elif input_ids is not None:
                input_shape = input_ids.size()
                input_ids = input_ids.view(-1, input_shape[-1])
            elif inputs_embeds is not None:
                input_shape = inputs_embeds.size()[:-1]
            else:
                raise ValueError("You have to specify either input_ids or inputs_embeds")

            if inputs_embeds is None:
                assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
                inputs_embeds = self.embed_tokens(input_ids)

            batch_size, seq_length = input_shape
            device = inputs_embeds.device

            # Prepare attention mask
            if attention_mask is None:
                attention_mask = torch.ones(batch_size, seq_length, device=device)

            if attention_mask is not None:
                # Convert to float and apply mask logic: 1 for tokens to attend, 0 for tokens to ignore
                extended_attention_mask = attention_mask[:, None, None, :]
                extended_attention_mask = extended_attention_mask.to(dtype=inputs_embeds.dtype)
                extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(inputs_embeds.dtype).min
            else:
                extended_attention_mask = None

            # Initialize past_key_values with empty states if not provided
            if past_key_values is None:
                past_key_values = [None] * len(self.block)

            # Prepare head mask if provided
            if head_mask is not None:
                if head_mask.dim() == 1:
                    head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                    head_mask = head_mask.expand(self.config.num_layers, -1, -1, -1, -1)
                elif head_mask.dim() == 2:
                    head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)

            present_key_value_states = () if use_cache else None
            all_hidden_states = () if output_hidden_states else None
            all_attentions = () if output_attentions else None
            all_cross_attentions = () if (output_attentions and self.is_decoder) else None
            position_bias = None
            encoder_decoder_position_bias = None

            hidden_states = inputs_embeds

            # Pass through each block with vision embeddings
            for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
                layer_head_mask = head_mask[i] if head_mask is not None else None
                cross_attn_layer_head_mask = None
                if self.is_decoder and encoder_hidden_states is not None:
                    cross_attn_layer_head_mask = head_mask[i] if head_mask is not None else None

                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                # Arctic-TILT: Pass vision_embeddings to each block for fusion
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    past_key_values=None,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    vision_embeddings=vision_embeddings,
                )

                # layer_outputs is a tuple: (hidden-states, key-value-states, attention_weights, position_bias, ...)
                if use_cache is False:
                    layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

                hidden_states, present_key_value_state = layer_outputs[:2]

                # Update position bias for next layer
                if i == 0:
                    position_bias = layer_outputs[2] if len(layer_outputs) > 2 else None
                    if self.is_decoder and encoder_hidden_states is not None:
                        encoder_decoder_position_bias = layer_outputs[3] if len(layer_outputs) > 3 else None

                if use_cache:
                    present_key_value_states = present_key_value_states + (present_key_value_state,)

                if output_attentions:
                    all_attentions = all_attentions + (layer_outputs[3] if len(layer_outputs) > 3 else layer_outputs[2],)
                    if self.is_decoder and encoder_hidden_states is not None:
                        all_cross_attentions = all_cross_attentions + (layer_outputs[4] if len(layer_outputs) > 4 else None,)

            hidden_states = self.final_layer_norm(hidden_states)
            hidden_states = self.dropout(hidden_states)

            # Add last hidden state
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if not return_dict:
                return tuple(
                    v
                    for v in [
                        hidden_states,
                        present_key_value_states,
                        all_hidden_states,
                        all_attentions,
                        all_cross_attentions,
                    ]
                    if v is not None
                )

            # Return appropriate output class
            from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
            return BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=hidden_states,
                past_key_values=present_key_value_states,
                hidden_states=all_hidden_states,
                attentions=all_attentions,
                cross_attentions=all_cross_attentions,
            )
        else:
            # Fallback to standard forward if no vision embeddings
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )


class T5Model(t5.modeling_t5.T5Model):
    def __init__(self, config):
        super().__init__(config=config)

        self.config = config
        encoder_config = copy.deepcopy(config)
        decoder_config = copy.deepcopy(config)
        decoder_config.update(dict(is_decoder=True))

        self.encoder = T5Stack(encoder_config, self.shared)
        self.decoder = T5Stack(decoder_config, self.shared)

        self.post_init()

    def forward(self, **kwargs):
        # Extract vision_embeddings if present
        vision_embeddings = kwargs.pop('vision_embeddings', None)

        # If we have vision embeddings, pass them to encoder
        if vision_embeddings is not None and 'encoder_outputs' not in kwargs:
            # Get encoder outputs with vision embeddings
            encoder_kwargs = {k: v for k, v in kwargs.items()
                            if k in ['input_ids', 'attention_mask', 'inputs_embeds', 'head_mask',
                                   'output_attentions', 'output_hidden_states', 'return_dict']}
            encoder_kwargs['vision_embeddings'] = vision_embeddings
            encoder_outputs = self.encoder(**encoder_kwargs)
            kwargs['encoder_outputs'] = encoder_outputs

        return super().forward(**kwargs)

    def load_weights(self):
        dummy_model = AutoModel.from_pretrained(self.config._name_or_path)
        self.load_state_dict(dummy_model.state_dict(), strict=False)
        print("Weights loaded successfully!")


class T5ForConditionalGeneration(t5.modeling_t5.T5ForConditionalGeneration):
    def __init__(self, config):
        '''
        It is similar to the T5ForConditionalGeneration described in the `hugging_face` repository, however I had to tweak it a bit,
        since there is an addition of the `relative_horizontal_bias` as well as `relative_vertical_bias` in the `T5Attention` class, and also
        the entire approach is generative in nature, so maybe it can be used in some other dataset, such as Question Answering
        '''

        super().__init__(config=config)

        self.config = config
        encoder_config = copy.deepcopy(config)
        decoder_config = copy.deepcopy(config)
        # In the pretrained version, the decoder config, the `is_decoder` option is True
        decoder_config.update(dict(is_decoder=True))

        self.encoder = T5Stack(encoder_config, self.shared)
        self.decoder = T5Stack(decoder_config, self.shared)

        if config.load_weights:
            self.load_weights()
        else:
            self.post_init()
            print("Initialization done without loading the weights")

    def generate(self,
                 inputs=None,
                 attention_mask=None,
                 inputs_embeds=None,
                 vision_embeddings=None,
                 **kwargs):
        """Override generate method to support vision_embeddings"""

        # Store vision_embeddings in a way that encoder can access
        if vision_embeddings is not None:
            # Get encoder outputs first with vision embeddings
            if inputs_embeds is not None:
                encoder_kwargs = {
                    'inputs_embeds': inputs_embeds,
                    'attention_mask': attention_mask,
                    'vision_embeddings': vision_embeddings,
                    'return_dict': True
                }
                encoder_outputs = self.encoder(**encoder_kwargs)
                kwargs['encoder_outputs'] = encoder_outputs

                # Remove conflicting parameters
                kwargs.pop('inputs_embeds', None)
                kwargs.pop('vision_embeddings', None)
                kwargs.pop("cache_position", None)

        # Call parent generate method
        return super().generate(
            inputs=inputs,
            attention_mask=attention_mask,
            **kwargs
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        encoder_outputs=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        vision_embeddings=None,
        cache_position=None,
        **kwargs
    ):

        if vision_embeddings is not None and encoder_outputs is None:
            encoder_kwargs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'inputs_embeds': inputs_embeds,
                'head_mask': head_mask,
                'output_attentions': output_attentions,
                'output_hidden_states': output_hidden_states,
                'return_dict': True,
                'vision_embeddings': vision_embeddings,
            }
            encoder_outputs = self.encoder(**encoder_kwargs)

            input_ids = None
            inputs_embeds = None

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def load_weights(self):
        '''
        Loads the weights from the pretrained model
        '''
        dummy_model = AutoModel.from_pretrained(self.config._name_or_path)
        self.load_state_dict(dummy_model.state_dict(), strict=False)
        print("Weights loaded successfully!")


class T5EncoderModel(t5.modeling_t5.T5ForConditionalGeneration):
    def __init__(self, config):
        '''
        It is similar to the T5EncoderModel described in the `hugging_face` repository, however I had to tweak it a bit,
        since there is an addition of the `relative_horizontal_bias` as well as `relative_vertical_bias` in the `T5Attention` class
        '''
        super().__init__(config=config)
        self.encoder = T5Stack(config, self.shared)
        self.post_init()

    def forward(self, **kwargs):
        '''Similar to the `T5EncoderModel` mentioned in the hugging face's t5 implementation'''
        return super().forward(**kwargs)


class T5ForConditionalGenerationAbstractive(t5.modeling_t5.T5ForConditionalGeneration):
    def __init__(self, config):
        '''
        T5ForConditionalGenerationAbstractive is a T5ForConditionalGeneration model with a linear layer on top of the decoder output,
        where the decoder output is the output of the last layer of the decoder, followed by a linear layer projection.

        It is similar to T5ForConditionalGeneration, however, it is based on a concept of generative answer, and this was what I did earlier,
        however, the authors have used an abstractive approach, and so I had to tweak somethinig, and essentially, it is the `self.lm_head`
        '''

        super().__init__(config=config)

        self.config = config
        encoder_config = copy.deepcopy(config)
        decoder_config = copy.deepcopy(config)
        # In the pretrained version, the decoder config, the `is_decoder` option is True
        decoder_config.update(dict(is_decoder=True))

        self.encoder = T5Stack(encoder_config, self.shared)
        self.decoder = T5Stack(decoder_config, self.shared)
        self.lm_head = nn.Linear(in_features=config.d_model,
                                 out_features=config.num_classes, bias=False)

        if config.load_weights:
            self.load_weights()
        else:
            self.post_init()
            print("Initialization done without loading the weights")

    def forward(self, **kwargs):
        '''
        Forward pass of T5ForConditionalGenerationAbstractive. It is similar to T5ForConditionalGeneration, however, it is based on a concept of generative answer,
        and this was what I did earlier,
        '''
        # Extract vision_embeddings if present
        vision_embeddings = kwargs.pop('vision_embeddings', None)

        # If we have vision embeddings, pass them to encoder
        if vision_embeddings is not None and 'encoder_outputs' not in kwargs:
            # Get encoder outputs with vision embeddings
            encoder_kwargs = {k: v for k, v in kwargs.items()
                            if k in ['input_ids', 'attention_mask', 'inputs_embeds', 'head_mask',
                                   'output_attentions', 'output_hidden_states', 'return_dict']}
            encoder_kwargs['vision_embeddings'] = vision_embeddings
            encoder_outputs = self.encoder(**encoder_kwargs)
            kwargs['encoder_outputs'] = encoder_outputs

        return super().forward(**kwargs)

    def load_weights(self):
        '''
        Load the weights of the T5ForConditionalGenerationAbstractive model
        It is adaptable to both the `t5-base` and `t5-large` configuration settings
        '''
        dummy_model = AutoModel.from_pretrained(self.config._name_or_path)
        self.load_state_dict(dummy_model.state_dict(), strict=False)
        print("Weights loaded successfully!")