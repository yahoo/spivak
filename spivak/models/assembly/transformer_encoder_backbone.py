# coding=utf-8
# Copyright 2023, Yahoo Inc.
# Copyright 2021 The Fairseq Authors and the HuggingFace Inc. team. All rights reserved.
# Copyright 2021 The Marian Team Authors and The HuggingFace Inc. team. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file contains pieces of code taken from the following set of files. At
# Yahoo Inc., the code was modified and new code was added.
# https://github.com/huggingface/transformers/blob/50a8ed3ee02a02550d7055d1539de5b12358cf26/src/transformers/models/hubert/modeling_tf_hubert.py
# https://github.com/huggingface/transformers/blob/50a8ed3ee02a02550d7055d1539de5b12358cf26/src/transformers/models/bert/modeling_tf_bert.py
# https://github.com/huggingface/transformers/blob/50a8ed3ee02a02550d7055d1539de5b12358cf26/src/transformers/models/marian/modeling_tf_marian.py

from typing import Optional, Tuple, List, Dict

import numpy as np
import tensorflow as tf
from tensorflow import Tensor

from spivak.models.assembly.huggingface_activations import gelu, gelu_new, \
    mish, gelu_fast
from spivak.models.assembly.layers import Nodes, NODE_PENULTIMATE

LARGE_NEGATIVE = -1e8
ACT2FN = {
    "gelu": gelu,
    "relu": tf.keras.activations.relu,
    "swish": tf.keras.activations.swish,
    "silu": tf.keras.activations.swish,
    "gelu_new": gelu_new,
    "mish": mish,
    "tanh": tf.keras.activations.tanh,
    "gelu_fast": gelu_fast,
}


def create_transformer_encoder_backbone(
        transformer_encoder_config: "TransformerEncoderConfig",
        input_reduction_out: Tensor) -> Nodes:
    # In order to comply with the standard Transformer format, we need to
    # adjust the dimensions. First we remove the extra singleton dimension
    # from the input, then run it through the Transformer encoder, then add
    # the singleton dimension back.
    input_reduction_out_squeezed = tf.squeeze(input_reduction_out, axis=2)
    encoder = TransformerEncoder(transformer_encoder_config)
    encoded = encoder(input_reduction_out_squeezed)
    expanded_encoded = tf.expand_dims(encoded, axis=2)
    return {NODE_PENULTIMATE: expanded_encoded}


class TransformerEncoderConfig:

    LEARNED_POSITIONAL_EMBEDDING = "learned"
    CONVOLUTIONAL_POSITIONAL_EMBEDDING = "convolutional"
    SINUSOIDAL_POSITIONAL_EMBEDDING = "sinusoidal"

    def __init__(
            self, embedding_size: int, num_chunk_frames: int,
            initializer_range: float, layer_dropout: float,
            layer_norm_eps: float, do_stable_layer_norm: bool,
            hidden_layers: int, hidden_groups: int, hidden_size: int,
            hidden_dropout: float, hidden_activation: str, attention_heads: int,
            attention_dropout: float, intermediate_size: int,
            activation_dropout: float, positional_embedding: str,
            convolutional_positional_embedding_kernel: int,
            convolutional_positional_embedding_groups: int,
            convolutional_positional_embedding_activation: str,
            name: str) -> None:
        self.embedding_size = embedding_size
        self.num_chunk_frames = num_chunk_frames
        self.initializer_range = initializer_range
        self.layer_dropout = layer_dropout
        self.layer_norm_eps = layer_norm_eps
        self.do_stable_layer_norm = do_stable_layer_norm
        self.hidden_layers = hidden_layers
        self.hidden_groups = hidden_groups
        self.hidden_size = hidden_size
        self.hidden_dropout = hidden_dropout
        self.hidden_activation = hidden_activation
        self.attention_heads = attention_heads
        self.attention_dropout = attention_dropout
        self.intermediate_size = intermediate_size
        self.activation_dropout = activation_dropout
        self.positional_embedding = positional_embedding
        self.convolutional_positional_embedding_kernel = \
            convolutional_positional_embedding_kernel
        self.convolutional_positional_embedding_groups = \
            convolutional_positional_embedding_groups
        self.convolutional_positional_embedding_activation = \
            convolutional_positional_embedding_activation
        self.name = name


# Copied from
# transformers.models.wav2vec2.modeling_tf_wav2vec2.TFWav2Vec2Encoder
class TransformerEncoder(tf.keras.layers.Layer):

    def __init__(self, config: TransformerEncoderConfig) -> None:
        # Due to the way layers works in Keras, we're supposed to be able to
        # serialize and deserialize a Layer object from a JSON using get_config
        # and from_config. We must thus store all the parameters from
        # dependencies here in the object, which makes it weird to try to do
        # dependency injection here.
        super().__init__(name=config.name)
        self.config = config
        self.layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="layer_norm")
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout)
        self.positional_embedding = \
            TransformerEncoder._create_positional_embedding(config)
        self.embedding_hidden_mapping_in = tf.keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=_get_initializer(config.initializer_range),
            name="embedding_hidden_mapping_in",
        )
        self.layer_groups = TransformerEncoder._create_layer_groups(config)

    def get_config(self) -> Dict:
        config_dict = super(TransformerEncoder, self).get_config()
        config_dict.update(self.config.__dict__)
        return config_dict

    @classmethod
    def from_config(cls, config_dict: Dict) -> "TransformerEncoder":
        return cls(TransformerEncoderConfig(**config_dict))

    def call(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        training: Optional[bool] = False,
    ) -> Tensor:
        # Masking was done here in original HUBERT code. I'm not sure why,
        # but will keep it here, at least initially, as it doesn't seem like
        # it can do any harm.
        if attention_mask is not None:
            hidden_states = hidden_states * tf.expand_dims(attention_mask, -1)
            attention_mask = _expand_mask(attention_mask)
        position_embeddings = self.positional_embedding(hidden_states)
        hidden_states = hidden_states + position_embeddings
        if not self.config.do_stable_layer_norm:
            hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        # Project dimension from initial hidden_states dimension (
        # embedding_size) to the desired hidden_size, which is used throughout
        # the rest of the encoder.
        hidden_states = self.embedding_hidden_mapping_in(inputs=hidden_states)
        for layer_index in range(self.config.hidden_layers):
            # LayerDrop is implemented here (see
            # https://arxiv.org/abs/1909.11556 for the description).
            dropout_probability = np.random.uniform(0, 1)
            if training and (dropout_probability < self.config.layer_dropout):
                # skip the layer
                continue
            group_index = int(
                layer_index /
                (self.config.hidden_layers / self.config.hidden_groups)
            )
            layer_group = self.layer_groups[group_index]
            hidden_states = layer_group(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                training=training
            )
        if self.config.do_stable_layer_norm:
            hidden_states = self.layer_norm(hidden_states)
        return hidden_states

    @staticmethod
    def _create_layer_groups(
            config: TransformerEncoderConfig) -> List[tf.keras.layers.Layer]:
        return [
            TransformerEncoder._create_encoder_layer(
                config, f"layer_groups.{i}")
            for i in range(config.hidden_groups)
        ]

    @staticmethod
    def _create_encoder_layer(
            config: TransformerEncoderConfig, name: str
    ) -> tf.keras.layers.Layer:
        attention = TransformerEncoder._create_attention(config)
        feed_forward = TransformerEncoder._create_feed_forward(config)
        if config.do_stable_layer_norm:
            return EncoderLayerStableLayerNorm(
                attention, feed_forward, config.hidden_dropout,
                config.layer_norm_eps, name=name)
        else:
            return EncoderLayer(
                attention, feed_forward, config.hidden_dropout,
                config.layer_norm_eps, name=name)

    @staticmethod
    def _create_positional_embedding(
            config: TransformerEncoderConfig) -> tf.keras.layers.Layer:
        if (config.positional_embedding ==
                TransformerEncoderConfig.CONVOLUTIONAL_POSITIONAL_EMBEDDING):
            conv = WeightNormConv1D(
                filters=config.embedding_size,
                kernel_size=config.convolutional_positional_embedding_kernel,
                groups=config.convolutional_positional_embedding_groups,
                explicit_padding=(
                        config.convolutional_positional_embedding_kernel // 2),
                name="pos_conv",
            )
            positional_embedding = ConvolutionalPositionalEmbedding(
                conv, config.convolutional_positional_embedding_kernel,
                config.convolutional_positional_embedding_activation,
                name="pos_conv_embed")
        elif (config.positional_embedding ==
              TransformerEncoderConfig.LEARNED_POSITIONAL_EMBEDDING):
            positional_embedding = LearnedPositionalEmbedding(
                config.embedding_size, config.num_chunk_frames,
                config.initializer_range, name="pos_learned_embed")
        elif (config.positional_embedding ==
              TransformerEncoderConfig.SINUSOIDAL_POSITIONAL_EMBEDDING):
            positional_embedding = SinusoidalPositionalEmbedding(
                config.num_chunk_frames, config.embedding_size,
                "pos_sinusoidal_embed")
        else:
            raise ValueError(f"Unknown positional embedding choice "
                             f"{config.positional_embedding}")
        return positional_embedding

    @staticmethod
    def _create_attention(config: TransformerEncoderConfig) -> "Attention":
        return Attention(
            embed_dim=config.hidden_size, num_heads=config.attention_heads,
            dropout=config.attention_dropout, is_decoder=False,
            name="attention")

    @staticmethod
    def _create_feed_forward(config: TransformerEncoderConfig) -> "FeedForward":
        return FeedForward(
            config.intermediate_size, config.hidden_size,
            config.activation_dropout, config.hidden_dropout,
            config.initializer_range, config.hidden_activation,
            name="feed_forward")


# Copied from
# transformers.models.wav2vec2.modeling_tf_wav2vec2.TFWav2Vec2EncoderLayer
class EncoderLayer(tf.keras.layers.Layer):

    def __init__(
            self, attention, feed_forward, hidden_dropout: float,
            layer_norm_eps: float, name: str):
        super().__init__(name=name)
        self.attention = attention
        self.dropout = tf.keras.layers.Dropout(hidden_dropout)
        self.layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=layer_norm_eps, name="layer_norm")
        self.feed_forward = feed_forward
        self.final_layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=layer_norm_eps, name="final_layer_norm"
        )

    def call(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        training: Optional[bool] = False,
    ) -> Tensor:
        attn_residual = hidden_states
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, training=training
        )
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = attn_residual + hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states


# Copied from
# transformers.models.wav2vec2.modeling_tf_wav2vec2.TFWav2Vec2EncoderLayerStableLayerNorm
class EncoderLayerStableLayerNorm(tf.keras.layers.Layer):

    def __init__(
            self, attention, feed_forward, hidden_dropout: float,
            layer_norm_eps: float, name: str) -> None:
        super().__init__(name=name)
        self.attention = attention
        self.dropout = tf.keras.layers.Dropout(hidden_dropout)
        self.layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=layer_norm_eps, name="layer_norm")
        self.feed_forward = feed_forward
        self.final_layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=layer_norm_eps, name="final_layer_norm"
        )

    def call(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        training: Optional[bool] = False,
    ) -> Tensor:
        attn_residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, training=training
        )
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = attn_residual + hidden_states
        hidden_states = hidden_states + self.feed_forward(
            self.final_layer_norm(hidden_states))
        return hidden_states


# Copied from transformers.models.bart.modeling_tf_bart.TFBartAttention
class Attention(tf.keras.layers.Layer):
    """Multi-headed attention from "Attention Is All You Need"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder

        self.k_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="k_proj")
        self.q_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="q_proj")
        self.v_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="v_proj")
        self.out_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="out_proj")

    def _shape(self, tensor: Tensor, seq_len: int, bsz: int):
        return tf.transpose(tf.reshape(tensor, (bsz, seq_len, self.num_heads, self.head_dim)), (0, 2, 1, 3))

    def call(
        self,
        hidden_states: Tensor,
        key_value_states: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[Tensor]]] = None,
        attention_mask: Optional[Tensor] = None,
        layer_head_mask: Optional[Tensor] = None,
        training=False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = _shape_list(hidden_states)
        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = tf.concat([past_key_value[0], key_states], axis=2)
            value_states = tf.concat([past_key_value[1], value_states], axis=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(Tensor, Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(Tensor, Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = tf.reshape(self._shape(query_states, tgt_len, bsz), proj_shape)
        key_states = tf.reshape(key_states, proj_shape)
        value_states = tf.reshape(value_states, proj_shape)

        src_len = _shape_list(key_states)[1]
        attn_weights = tf.matmul(query_states, key_states, transpose_b=True)

        # The tf.debugging asserts are not compliant with XLA then they
        # have to be disabled in other modes than eager.
        if tf.executing_eagerly():
            tf.debugging.assert_equal(
                _shape_list(attn_weights),
                [bsz * self.num_heads, tgt_len, src_len],
                message=f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {_shape_list(attn_weights)}",
            )

        if attention_mask is not None:
            # The tf.debugging asserts are not compliant with XLA then they
            # have to be disabled in other modes than eager.
            if tf.executing_eagerly():
                tf.debugging.assert_equal(
                    _shape_list(attention_mask),
                    [bsz, 1, tgt_len, src_len],
                    message=f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {_shape_list(attention_mask)}",
                )

            attention_mask = tf.cast(attention_mask, dtype=attn_weights.dtype)
            attn_weights = tf.reshape(attn_weights, (bsz, self.num_heads, tgt_len, src_len)) + attention_mask
            attn_weights = tf.reshape(attn_weights, (bsz * self.num_heads, tgt_len, src_len))

        attn_weights = tf.nn.softmax(attn_weights, axis=-1)

        if layer_head_mask is not None:
            # The tf.debugging asserts are not compliant with XLA then they
            # have to be disabled in other modes than eager.
            if tf.executing_eagerly():
                tf.debugging.assert_equal(
                    _shape_list(layer_head_mask),
                    [self.num_heads],
                    message=f"Head mask for a single layer should be of size {(self.num_heads)}, but is {_shape_list(layer_head_mask)}",
                )

            attn_weights = tf.reshape(layer_head_mask, (1, -1, 1, 1)) * tf.reshape(
                attn_weights, (bsz, self.num_heads, tgt_len, src_len)
            )
            attn_weights = tf.reshape(attn_weights, (bsz * self.num_heads, tgt_len, src_len))

        attn_probs = self.dropout(attn_weights, training=training)
        attn_output = tf.matmul(attn_probs, value_states)

        # The tf.debugging asserts are not compliant with XLA then they
        # have to be disabled in other modes than eager.
        if tf.executing_eagerly():
            tf.debugging.assert_equal(
                _shape_list(attn_output),
                [bsz * self.num_heads, tgt_len, self.head_dim],
                message=f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {_shape_list(attn_output)}",
            )

        attn_output = tf.transpose(
            tf.reshape(attn_output, (bsz, self.num_heads, tgt_len, self.head_dim)), (0, 2, 1, 3)
        )
        attn_output = tf.reshape(attn_output, (bsz, tgt_len, embed_dim))

        attn_output = self.out_proj(attn_output)
        attn_weights: Tensor = tf.reshape(attn_weights, (bsz, self.num_heads, tgt_len, src_len))

        return attn_output, attn_weights, past_key_value


# Copied from
# transformers.models.wav2vec2.modeling_tf_wav2vec2.TFWav2Vec2FeedForward
class FeedForward(tf.keras.layers.Layer):

    def __init__(
            self, intermediate_size: int, hidden_size: int,
            activation_dropout: float, hidden_dropout: float,
            initializer_range, hidden_act, name: str):
        super().__init__(name=name)
        self.intermediate_dropout = tf.keras.layers.Dropout(activation_dropout)
        self.intermediate_dense = tf.keras.layers.Dense(
            units=intermediate_size,
            kernel_initializer=_get_initializer(initializer_range),
            bias_initializer="zeros",
            name="intermediate_dense",
        )
        self.intermediate_act_fn = _get_tf_activation(hidden_act)
        self.output_dense = tf.keras.layers.Dense(
            units=hidden_size,
            kernel_initializer=_get_initializer(initializer_range),
            bias_initializer="zeros",
            name="output_dense",
        )
        self.output_dropout = tf.keras.layers.Dropout(hidden_dropout)

    def call(
            self, hidden_states: Tensor,
            training: bool = False) -> Tensor:
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(
            hidden_states, training=training)
        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states, training=training)
        return hidden_states


# Copied from
# transformers.models.wav2vec2.modeling_tf_wav2vec2.TFWav2Vec2PositionalConvEmbedding
class ConvolutionalPositionalEmbedding(tf.keras.layers.Layer):

    def __init__(
            self, conv, num_conv_pos_embeddings, feat_extract_activation,
            name: str) -> None:
        super().__init__(name=name)
        self.conv = conv
        self.padding = SamePadLayer(num_conv_pos_embeddings)
        self.activation = _get_tf_activation(feat_extract_activation)

    def call(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.conv(hidden_states)
        hidden_states = self.padding(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_tf_bert.TFBertEmbeddings.call
class LearnedPositionalEmbedding(tf.keras.layers.Layer):

    def __init__(
            self, embedding_size, max_position_embeddings, initializer_range,
            name: str):
        super().__init__(name=name)
        self.embedding_size = embedding_size
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range

    def build(self, input_shape: tf.TensorShape):
        with tf.name_scope("position_embeddings"):
            self.position_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.max_position_embeddings, self.embedding_size],
                initializer=_get_initializer(self.initializer_range),
            )
        super().build(input_shape)

    # Copied from
    # transformers.models.bert.modeling_tf_bert.TFBertEmbeddings.call
    def call(self, hidden_states: Tensor) -> Tensor:
        input_shape = _shape_list(hidden_states)[:-1]
        position_ids = tf.expand_dims(
            tf.range(start=0, limit=input_shape[1]), axis=0)
        position_embeds = tf.gather(
            params=self.position_embeddings, indices=position_ids)
        position_embeds = tf.tile(
            input=position_embeds, multiples=(input_shape[0], 1, 1))
        return position_embeds


# Copied from
# transformers.models.marian.modeling_tf_marian.TFMarianSinusoidalPositionalEmbedding
class SinusoidalPositionalEmbedding(tf.keras.layers.Layer):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(
            self, num_positions: int, embedding_size: int, name: str) -> None:
        super().__init__(name=name)
        if embedding_size % 2 != 0:
            raise NotImplementedError(
                f"odd embedding_dim {embedding_size} not supported")
        self.embedding_size = embedding_size
        self.num_positions = num_positions

    def build(self, input_shape: tf.TensorShape):
        """
        Build shared token embedding layer Shared weights logic adapted from
        https://github.com/tensorflow/models/blob/a009f4fb9d2fc4949e32192a944688925ef78659/official/transformer/v2/embedding_layer.py#L24
        """
        weight = self._init_weight(self.num_positions, self.embedding_size)
        self.weight = self.add_weight(
            name="embeddings",
            shape=[self.num_positions, self.embedding_size],
        )
        weight = tf.cast(weight, dtype=self.weight.dtype)
        self.weight.assign(weight)
        super().build(input_shape)

    @staticmethod
    def _init_weight(n_pos: int, embedding_size: int):
        """
        Identical to the XLM create_sinusoidal_embeddings except features are
        not interleaved. The cos features are in the 2nd half of the vector.
        [dim // 2:]
        """
        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / embedding_size)
                 for j in range(embedding_size)]
                for pos in range(n_pos)
            ]
        )
        table = np.zeros_like(position_enc)
        # index 0 is all zero
        table[:, 0:(embedding_size // 2)] = np.sin(position_enc[:, 0::2])
        table[:, (embedding_size // 2):] = np.cos(position_enc[:, 1::2])
        # convert to tensor
        table = tf.convert_to_tensor(table)
        tf.stop_gradient(table)
        return table

    def call(self, hidden_states: Tensor) -> Tensor:
        """Input is expected to be of size [bsz x seq_len]."""
        input_shape = _shape_list(hidden_states)
        bsz, seq_len = input_shape[:2]
        positions = tf.range(0, seq_len, delta=1, name="range")
        return tf.gather(self.weight, positions)


# Copied from
# transformers.models.wav2vec2.modeling_tf_wav2vec2.TFWav2Vec2SamePadLayer
class SamePadLayer(tf.keras.layers.Layer):
    def __init__(self, num_conv_pos_embeddings, **kwargs):
        super().__init__(**kwargs)
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    def call(self, hidden_states):
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, : -self.num_pad_remove, :]
        return hidden_states


# Copied from
# transformers.models.wav2vec2.modeling_tf_wav2vec2.TFWav2Vec2WeightNormConv1D
class WeightNormConv1D(tf.keras.layers.Conv1D):
    """Adapted from https://www.tensorflow.org/probability/api_docs/python/tfp/layers/weight_norm/WeightNorm"""

    def __init__(self, filters, kernel_size, groups, explicit_padding, **kwargs):
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            groups=groups,
            padding="valid",
            use_bias=True,
            bias_initializer="he_normal",
            **kwargs,
        )
        self.explicit_padding = explicit_padding
        self.filter_axis = 2
        self.initialized = False
        self.kernel_norm_axes = tf.constant([0, 1])

    def _init_norm(self):
        """Set the norm of the weight vector."""
        kernel_norm = tf.sqrt(tf.reduce_sum(tf.square(self.weight_v), axis=self.kernel_norm_axes))
        self.weight_g.assign(kernel_norm[:, tf.newaxis, tf.newaxis])

    def _normalize_kernel(self):
        """Generate normalized weights."""
        kernel = tf.nn.l2_normalize(self.weight_v, axis=self.kernel_norm_axes) * tf.transpose(self.weight_g)
        self.kernel = tf.transpose(kernel)

    def build(self, input_shape):
        if not self.built:
            input_shape = input_shape.as_list()
            # Conv1D output shapes are checked at build time since TF 2.7, so we need to account for padding
            input_shape[-2] += self.explicit_padding * 2
            super().build(input_shape)

            self.kernel = tf.Variable(tf.transpose(self.kernel), name="weight_v", trainable=True)
            self.weight_v = self.kernel

            self.weight_g = self.add_weight(
                name="weight_g",
                shape=(int(self.weight_v.shape[self.filter_axis]), 1, 1),
                initializer="ones",
                dtype=self.weight_v.dtype,
                trainable=True,
            )
            self.bias = self.add_weight(name="bias", shape=(self.filters,), initializer="zeros", trainable=True)

    def call(self, inputs):
        if not self.initialized:
            self._init_norm()
            self.initialized = True

        self._normalize_kernel()

        padded_inputs = tf.pad(inputs, ((0, 0), (self.explicit_padding, self.explicit_padding), (0, 0)))
        output = super().call(padded_inputs)

        return output


# Copied from transformers.models.bart.modeling_tf_bart._expand_mask
def _expand_mask(mask: Tensor) -> Tensor:
    """
    Expands attention_mask from `[bsz, seq_len]` to
    `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    src_len = _shape_list(mask)[1]
    tgt_len = src_len
    one_cst = tf.constant(1.0)
    mask = tf.cast(mask, dtype=one_cst.dtype)
    expanded_mask = tf.tile(mask[:, None, None, :], (1, 1, tgt_len, 1))
    return (one_cst - expanded_mask) * LARGE_NEGATIVE


def _get_initializer(
        initializer_range: float = 0.02) -> tf.initializers.TruncatedNormal:
    """
    Creates a :obj:`tf.initializers.TruncatedNormal` with the given range.

    Args:
        initializer_range (`float`, defaults to 0.02): Standard deviation of the initializer range.

    Returns:
        :obj:`tf.initializers.TruncatedNormal`: The truncated normal initializer.
    """
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)


def _shape_list(tensor: Tensor) -> List[int]:
    """
    Deal with dynamic shape in tensorflow cleanly.

    Args:
        tensor (:obj:`Tensor`): The tensor we want the shape of.

    Returns:
        :obj:`List[int]`: The shape of the tensor as a list.
    """
    dynamic = tf.shape(tensor)
    if tensor.shape == tf.TensorShape(None):
        return dynamic
    static = tensor.shape.as_list()
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def _get_tf_activation(activation_string):
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError(
            f"function {activation_string} not found in ACT2FN mapping "
            f"{list(ACT2FN.keys())}")
