# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# CifModel implemented by RYM

import logging
from argparse import Namespace
import contextlib
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from omegaconf import MISSING, II, open_dict
from typing import Any, Optional

from fairseq import checkpoint_utils, tasks, utils
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.tasks import FairseqTask
from fairseq.models import (
    BaseFairseqModel,
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
)
from fairseq.models.wav2vec.wav2vec2 import MASKING_DISTRIBUTION_CHOICES
from fairseq.modules import (
    LayerNorm,
    PositionalEmbedding,
    TransformerDecoderLayer,
    AdaptiveSoftmax,
    BaseLayer,
    FairseqDropout,
    LayerDropModuleList,
    SinusoidalPositionalEmbedding,
    TransformerSentenceEncoderLayer
)

from espnet.nets.pytorch_backend.nets_utils import pad_list

logger = logging.getLogger(__name__)


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


@dataclass
class Wav2Vec2AsrConfig(FairseqDataclass):
    w2v_path: str = field(
        default=MISSING, metadata={"help": "path to wav2vec 2.0 model"}
    )
    no_pretrained_weights: bool = field(
        default=False, metadata={"help": "if true, does not load pretrained weights"}
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    final_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout after transformer and before final projection"},
    )
    dropout: float = field(
        default=0.0, metadata={"help": "dropout probability inside wav2vec 2.0 model"}
    )
    attention_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability for attention weights inside wav2vec 2.0 model"
        },
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN inside wav2vec 2.0 model"
        },
    )
    conv_feature_layers: Optional[str] = field(
        default="[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]",
        metadata={
            "help": (
                "string describing convolutional feature extraction "
                "layers in form of a python list that contains "
                "[(dim, kernel_size, stride), ...]"
            ),
        },
    )
    encoder_embed_dim: Optional[int] = field(
        default=768, metadata={"help": "encoder embedding dimension"}
    )

    # masking
    apply_mask: bool = field(
        default=False, metadata={"help": "apply masking during fine-tuning"}
    )
    mask_length: int = field(
        default=10, metadata={"help": "repeat the mask indices multiple times"}
    )
    mask_prob: float = field(
        default=0.5,
        metadata={
            "help": "probability of replacing a token with mask (normalized by length)"
        },
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose masks"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), "
            "see help in compute_mask_indices"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )
    mask_min_space: Optional[int] = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )

    # channel masking
    mask_channel_length: int = field(
        default=10, metadata={"help": "length of the mask for features (channels)"}
    )
    mask_channel_prob: float = field(
        default=0.0, metadata={"help": "probability of replacing a feature with 0"}
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False, metadata={"help": "whether to allow channel masks to overlap"}
    )
    freeze_finetune_updates: int = field(
        default=0, metadata={"help": "dont finetune wav2vec for this many updates"}
    )
    feature_grad_mult: float = field(
        default=0.0, metadata={"help": "reset feature grad mult in wav2vec 2.0 to this"}
    )
    layerdrop: float = field(
        default=0.0, metadata={"help": "probability of dropping a layer in wav2vec 2.0"}
    )
    mask_channel_min_space: Optional[int] = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )
    mask_channel_before: bool = False
    normalize: bool = II("task.normalize")
    data: str = II("task.data")
    # this holds the loaded wav2vec args
    w2v_args: Any = None


@dataclass
class Wav2Vec2CtcConfig(Wav2Vec2AsrConfig):
    blank_weight: float = 0
    blank_mode: str = "add"


@register_model("wav2vec_ctc", dataclass=Wav2Vec2CtcConfig)
class Wav2VecCtc(BaseFairseqModel):
    def __init__(self, cfg: Wav2Vec2CtcConfig, w2v_encoder: BaseFairseqModel):
        super().__init__()
        self.cfg = cfg
        self.w2v_encoder = w2v_encoder
        self.blank_weight = cfg.blank_weight
        self.blank_mode = cfg.blank_mode

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: Wav2Vec2CtcConfig, task: FairseqTask):
        """Build a new model instance."""
        w2v_encoder = Wav2VecEncoder(cfg, len(task.target_dictionary))
        return cls(cfg, w2v_encoder)

    def get_logits(self, net_output, normalize=False):
        logits = net_output["encoder_out"]
        if self.blank_weight != 0:
            if self.blank_mode == "add":
                logits[..., 0] += self.blank_weight
            elif self.blank_mode == "set":
                logits[..., 0] = self.blank_weight
            else:
                raise Exception(f"invalid blank mode {self.blank_mode}")

        if net_output["padding_mask"] is not None and net_output["padding_mask"].any():
            logits[net_output["padding_mask"].T][..., 0] = float("inf")
            logits[net_output["padding_mask"].T][..., 1:] = float("-inf")

        if normalize:
            logits = utils.log_softmax(logits.type_as(net_output["encoder_out"]), dim=-1)

        return logits

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = self.get_logits(net_output)

        if log_probs:
            return utils.log_softmax(logits.type_as(net_output["encoder_out"]), dim=-1)
        else:
            return utils.softmax(logits.type_as(net_output["encoder_out"]), dim=-1)

    def forward(self, **kwargs):
        x = self.w2v_encoder(**kwargs)
        return x


@dataclass
class Wav2Vec2Seq2SeqConfig(Wav2Vec2AsrConfig):
    decoder_embed_dim: int = field(
        default=768, metadata={"help": "decoder embedding dimension"}
    )
    decoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "decoder embedding dimension for FFN"}
    )
    decoder_layers: int = field(default=6, metadata={"help": "num of decoder layers"})
    decoder_layerdrop: float = field(
        default=0.0, metadata={"help": "decoder layerdrop chance"}
    )
    decoder_attention_heads: int = field(
        default=4, metadata={"help": "num decoder attention heads"}
    )
    decoder_learned_pos: bool = field(
        default=False,
        metadata={"help": "use learned positional embeddings in the decoder"},
    )
    decoder_normalize_before: bool = field(
        default=False, metadata={"help": "apply layernorm before each decoder block"}
    )
    no_token_positional_embeddings: bool = field(
        default=False,
        metadata={
            "help": "if set, disables positional embeddings (outside self attention)"
        },
    )
    decoder_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability in the decoder"}
    )
    decoder_attention_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability for attention weights inside the decoder"
        },
    )
    decoder_activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN inside the decoder"
        },
    )
    max_target_positions: int = field(
        default=2048, metadata={"help": "max target positions"}
    )
    share_decoder_input_output_embed: bool = field(
        default=False, metadata={"help": "share decoder input and output embeddings"}
    )
    autoregressive: bool = II("task.autoregressive")


@register_model("wav2vec_seq2seq", dataclass=Wav2Vec2Seq2SeqConfig)
class Wav2Vec2Seq2SeqModel(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @classmethod
    def build_model(cls, cfg: Wav2Vec2Seq2SeqConfig, task: FairseqTask):
        """Build a new model instance."""

        assert (
            cfg.autoregressive
        ), "Please set task.autoregressive=true for seq2seq asr models"

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            return emb

        decoder_embed_tokens = build_embedding(tgt_dict, cfg.decoder_embed_dim)

        encoder = cls.build_encoder(cfg)
        decoder = cls.build_decoder(cfg, tgt_dict, decoder_embed_tokens)

        return Wav2Vec2Seq2SeqModel(encoder, decoder)

    @classmethod
    def build_encoder(cls, cfg: Wav2Vec2AsrConfig):
        return Wav2VecEncoder(cfg)

    @classmethod
    def build_decoder(cls, cfg: Wav2Vec2Seq2SeqConfig, tgt_dict, embed_tokens):
        return TransformerDecoder(cfg, tgt_dict, embed_tokens)

    def forward(self, **kwargs):
        encoder_out = self.encoder(**kwargs)
        decoder_out = self.decoder(encoder_out=encoder_out, **kwargs)
        return decoder_out

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict


class Wav2VecEncoder(FairseqEncoder):
    def __init__(self, cfg: Wav2Vec2AsrConfig, output_size=None):
        self.apply_mask = cfg.apply_mask

        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_before": cfg.mask_channel_before,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
        }

        if cfg.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(cfg.w2v_path, arg_overrides)
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            w2v_args.criterion = None
            w2v_args.lr_scheduler = None
            cfg.w2v_args = w2v_args
        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(w2v_args)

        assert cfg.normalize == w2v_args.task.normalize, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for both pre-training and here"
        )

        w2v_args.task.data = cfg.data
        task = tasks.setup_task(w2v_args.task)
        model = task.build_model(w2v_args.model)

        if state is not None and not cfg.no_pretrained_weights:
            model.load_state_dict(state["model"], strict=True)

        model.remove_pretraining_modules()

        super().__init__(task.source_dictionary)

        d = w2v_args.model.encoder_embed_dim

        self.w2v_model = model

        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0

        targ_d = None
        self.proj = None

        if output_size is not None:
            targ_d = output_size
        elif getattr(cfg, "decoder_embed_dim", d) != d:
            targ_d = cfg.decoder_embed_dim

        if targ_d is not None:
            self.proj = Linear(d, targ_d)

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward(self, source, padding_mask, **kwargs):

        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
        }

        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            res = self.w2v_model.extract_features(**w2v_args)

            x = res["x"]
            padding_mask = res["padding_mask"]

            # B x T x C -> T x B x C
            x = x.transpose(0, 1)

        x = self.final_dropout(x)

        if self.proj:
            x = self.proj(x)

        return {
            "encoder_out": x,  # T x B x C
            "padding_mask": padding_mask,  # B x T,
            "layer_results": res["layer_results"],
        }

    def forward_torchscript(self, net_input):
        if torch.jit.is_scripting():
            return self.forward(net_input["source"], net_input["padding_mask"])
        else:
            return self.forward_non_torchscript(net_input)

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = encoder_out["encoder_out"].index_select(
                1, new_order
            )
        if encoder_out["padding_mask"] is not None:
            encoder_out["padding_mask"] = encoder_out[
                "padding_mask"
            ].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return None

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict


class TransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self,
        cfg: Wav2Vec2Seq2SeqConfig,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
    ):
        super().__init__(dictionary)

        self.dropout = cfg.decoder_dropout
        self.share_input_output_embed = cfg.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = cfg.decoder_embed_dim
        self.output_embed_dim = cfg.decoder_embed_dim

        self.layerdrop = cfg.decoder_layerdrop

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = cfg.max_target_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)  # todo: try with input_embed_dim

        self.project_in_dim = (
            Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )

        self.embed_positions = (
            PositionalEmbedding(
                cfg.max_target_positions,
                embed_dim,
                self.padding_idx,
                learned=cfg.decoder_learned_pos,
            )
            if not cfg.no_token_positional_embeddings
            else None
        )

        # TODO: update this when transformer gets converted to dataclass configs
        transformer_cfg = copy.deepcopy(cfg)
        with open_dict(transformer_cfg):
            transformer_cfg.dropout = transformer_cfg.decoder_dropout
            transformer_cfg.attention_dropout = (
                transformer_cfg.decoder_attention_dropout
            )
            transformer_cfg.activation_dropout = (
                transformer_cfg.decoder_activation_dropout
            )

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                TransformerDecoderLayer(transformer_cfg, no_encoder_attn)
                for _ in range(transformer_cfg.decoder_layers)
            ]
        )

        if not self.share_input_output_embed:
            self.embed_out = nn.Parameter(
                torch.Tensor(len(dictionary), self.output_embed_dim)
            )

            nn.init.normal_(self.embed_out, mean=0, std=self.output_embed_dim ** -0.5)

        if transformer_cfg.decoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

    def forward(
        self, prev_output_tokens, encoder_out=None, incremental_state=None, **unused
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        prev_output_tokens = prev_output_tokens.long()
        x, extra = self.extract_features(
            prev_output_tokens, encoder_out, incremental_state
        )
        x = self.output_layer(x)
        return x, extra

    def extract_features(
        self, prev_output_tokens, encoder_out=None, incremental_state=None, **unused
    ):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """

        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None

        inner_states = [x]

        # decoder layers
        self_attn_padding_mask = None
        if prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)
        for layer in self.layers:
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.layerdrop):
                x, attn, _ = layer(
                    x,
                    encoder_out["encoder_out"] if encoder_out is not None else None,
                    encoder_out["padding_mask"] if encoder_out is not None else None,
                    incremental_state,
                    self_attn_mask=self.buffered_future_mask(x)
                    if incremental_state is None
                    else None,
                    self_attn_padding_mask=self_attn_padding_mask
                )
                inner_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x, {"attn": attn, "inner_states": inner_states}

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        # project back to size of vocabulary
        if self.share_input_output_embed:
            return F.linear(features, self.embed_tokens.weight)
        else:
            return F.linear(features, self.embed_out)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (
            not hasattr(self, "_future_mask")
            or self._future_mask is None
            or self._future_mask.device != tensor.device
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(tensor.new(dim, dim)), 1
            )
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict



# My wav2vec2 with CIF

@dataclass
class Wav2Vec2CifSeq2SeqConfig(Wav2Vec2Seq2SeqConfig):
    blank_weight: float = 0
    blank_mode: str = "add"

    decoder_path: str = field(
        default=MISSING, metadata={"help": "path to pretrained language model"}
    )
    decoder_args: Any = None

    cif_threshold: float = field(
        default = 1.0,
        metadata = {"help": "threshold for integrate and fire"}
    )
    cif_embedding_dim: int = field(
        default = 768,
        metadata = {"help": "cif embedding dimension"}
    )
    produce_weight_type: str = field(
        default="conv",
        metadata={"help": "type of weight producing network: dense/conv"}
    )
    apply_scaling: bool = field(
        default=True,
        metadata={"help": "whether to apply scaling technique"}
    )
    apply_tail_handling: bool = field(
        default=True,
        metadata={"help": "whether to apply tail handling technique"}
    )
    conv_cif_layer_num: int = field(
        default=1,
        metadata={"help": "number of convolutional layers of cif module"}
    )
    conv_cif_output_channels_num: int = field(
        default=768,
        metadata={"help": "dimension of output of convolutional layer of cif module"}
    )
    conv_cif_width: int = field(
        default=5,
        metadata={"help": "kernel size of convolutional layer of cif module"}
    )
    conv_cif_dropout: float = field(
        default=0.1,
    )


@register_model("wav2vec_cif", dataclass=Wav2Vec2CifSeq2SeqConfig)
class Wav2Vec2CifModel(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder, cfg:Wav2Vec2CifSeq2SeqConfig):
        super().__init__(encoder, decoder)

        self.blank_weight = cfg.blank_weight
        self.blank_mode = cfg.blank_mode

    @classmethod
    def build_model(cls, cfg: Wav2Vec2CifSeq2SeqConfig, task: FairseqTask):
        """Build a new model instance."""

        assert (
            cfg.autoregressive
        ), "Please set task.autoregressive=true for seq2seq asr models"

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            return emb

        decoder_embed_tokens = build_embedding(tgt_dict, cfg.decoder_embed_dim)

        encoder = cls.build_encoder(cfg, tgt_dict)
        decoder = cls.build_decoder(cfg, tgt_dict, decoder_embed_tokens)

        return Wav2Vec2CifModel(encoder, decoder, cfg)

    @classmethod
    def build_encoder(cls, cfg: Wav2Vec2CifSeq2SeqConfig, tgt_dict):
        return CifEncoder(cfg, tgt_dict)

    @classmethod
    def build_decoder(cls, cfg: Wav2Vec2CifSeq2SeqConfig, tgt_dict, embed_tokens):
        return PretrainedTransformerDecoder(cfg, tgt_dict, embed_tokens, no_encoder_attn=True) # encoder attention is not needed.

    def forward(self, target_lengths=None, **kwargs):
        decoder_out, _ = self.decoder(**kwargs)

        cifencoder_out = self.encoder(
            target_lengths=target_lengths if self.training else None,
            **kwargs
        )

        encoder_outputs = cifencoder_out["encoder_outputs"]
        cif_outputs = cifencoder_out["cif_outputs"]
        cif_out = cif_outputs["cif_out"]

        # logger.info(cifencoder_out)
        # logger.info(cif_outputs)
        # logger.info(decoder_out.shape)

        # Truncate
        if cif_out.shape[1] != decoder_out.shape[1]:
            # logger.info("Encoder and decoder output dimension don't match. Decoder input is {}".format(decoder_input))

            Tmin = min(cif_out.shape[1], decoder_out.shape[1])
            cif_out = cif_out[:, :Tmin, :]
            decoder_out = decoder_out[:, :Tmin, :]
            cif_outputs["padding_mask"] = cif_outputs["padding_mask"][:, :Tmin]

        # assert cif_outputs["cif_out"].shape==decoder_out.shape, "Cif output size {} doesn't match decoder output size {}".format(cif_out.shape, decoder_out.shape)

        joint_out = torch.tanh(cif_out + decoder_out)
        # logger.info(joint_out.type())

        # Project to vocabulary size
        joint_out = self.decoder.output_layer(joint_out)
        # logger.info("Joint output shape {}".format(joint_out.shape))

        return {
            "joint_outputs": {
                "joint_out": joint_out,  # B x Tmin x C
                "padding_mask": cif_outputs["padding_mask"],
                "qua_out": cif_outputs["loss_pen"],
            },

            "encoder_outputs": encoder_outputs,
        }

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    def get_logits(self, net_output, normalize=False):
        logits = net_output["joint_out"]

        if net_output["padding_mask"] is not None and net_output["padding_mask"].any():
            logits[net_output["padding_mask"]][...] = float("-inf")

        if normalize:
            logits = utils.log_softmax(logits.type_as(net_output["joint_out"]), dim=-1)

        return logits

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a cif model's output."""

        logits = self.get_logits(net_output)

        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    def get_encoder_logits(self, net_output, normalize=False):
        logits = net_output["encoder_out"]
        if self.blank_weight != 0:
            if self.blank_mode == "add":
                logits[..., 0] += self.blank_weight
            elif self.blank_mode == "set":
                logits[..., 0] = self.blank_weight
            else:
                raise Exception(f"invalid blank mode {self.blank_mode}")

        if net_output["padding_mask"] is not None and net_output["padding_mask"].any():
            logits[net_output["padding_mask"].T][..., 0] = float("inf")
            logits[net_output["padding_mask"].T][..., 1:] = float("-inf")

        if normalize:
            logits = utils.log_softmax(logits.type_as(net_output["encoder_out"]), dim=-1)

        return logits

    def get_probs_from_logits(self, logits, log_probs=False):
        """Get normalized probabilities (or log probs) from logits."""
        if log_probs:
            return utils.log_softmax(logits, dim=-1)
        else:
            return utils.softmax(logits, dim=-1)

    def get_cif_output(self, target_lengths=None, **kwargs):
        cifencoder_out = self.encoder(target_lengths=target_lengths, **kwargs)
        return cifencoder_out["cif_outputs"]

    def get_encoder_output(self, **kwargs):
        cifencoder_out = self.encoder(target_lengths=None, **kwargs)
        return cifencoder_out["encoder_outputs"]

    def step_forward_decoder(self,
                             prev_decoded_tokens,
                             cif_outputs,
                             incremental_state=None, **kwargs):
        decoder_out, _ = self.decoder(prev_output_tokens=prev_decoded_tokens, **kwargs)
        cif_out = cif_outputs["cif_out"]

        # Truncate
        if cif_out.shape[1] != decoder_out.shape[1]:
            # logger.info("Encoder and decoder output dimension don't match. Decoder input is {}".format(decoder_input))

            Tmin = min(cif_out.shape[1], decoder_out.shape[1])
            cif_out = cif_out[:, :Tmin, :]
            decoder_out = decoder_out[:, :Tmin, :]
            cif_outputs["padding_mask"] = cif_outputs["padding_mask"][:, :Tmin]

        joint_out = torch.tanh(cif_out + decoder_out)

        # Project to vocabulary size
        joint_out = self.decoder.output_layer(joint_out)
        return joint_out


class Cif(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Load configurations
        self.cif_threshold = cfg.cif_threshold
        self.cif_output_dim = cfg.cif_embedding_dim
        self.encoder_embed_dim = cfg.encoder_embed_dim
        self.produce_weight_type = cfg.produce_weight_type
        self.apply_scaling = cfg.apply_scaling
        self.apply_tail_handling = cfg.apply_tail_handling

        # Build weight generator
        if self.produce_weight_type == "dense":
            self.dense_proj = Linear(self.encoder_embed_dim, cfg.dense_cif_units_num).cuda()
            self.weight_proj = Linear(cfg.dense_cif_units_num, 1).cuda()
        elif self.produce_weight_type == "conv":
            self.cif_conv_layer_num = cfg.conv_cif_layer_num
            self.conv = torch.nn.Conv1d(
                self.encoder_embed_dim,
                cfg.conv_cif_output_channels_num,
                cfg.conv_cif_width,
                stride=1, padding=2,
                dilation=1, groups=1,
                bias=True, padding_mode='zeros'
            ).cuda()
            self.conv_dropout = torch.nn.Dropout(p=cfg.conv_cif_dropout).cuda()
            self.weight_proj = Linear(cfg.conv_cif_output_channels_num, 1).cuda()
        else:
            self.weight_proj = Linear(self.encoder_embed_dim, 1).cuda()

        # Build the final projection layer for cif outputs if necessary
        if self.cif_output_dim != self.encoder_embed_dim:
            self.cif_output_proj = Linear(self.encoder_embed_dim, self.cif_output_dim, bias=False).cuda()

    def forward(self, encoder_outputs, target_lengths=None, transpose=True):
        """
        Args:
            encoder_outputs: a dictionary that includes
                encoder_raw_out: the raw outputs of acoustic encoder, with shape B x T x C
                encoder_padding_mask: the padding mask of encoder outputs, with shape B x T
            target_lengths: the length of targets (necessary when training), with shape B
        Return:
            A dictionary:
                cif_out:
                cif_out_padding_mask:
                quantity_out:
        """
        # Gather inputs
        encoder_raw_outputs = encoder_outputs["encoder_raw_out"]
        if transpose:
            encoder_raw_outputs = encoder_raw_outputs.transpose(0, 1) # T x B x C -> B x T x C
        encoder_padding_mask = encoder_outputs["padding_mask"]  # B x T

        # Produce weights
        if self.produce_weight_type == "dense":
            proj_out = self.dense_proj(encoder_raw_outputs)
            act_proj_out = torch.relu(proj_out)
            sig_input = self.weight_proj(act_proj_out)
            weight = torch.sigmoid(sig_input)
        elif self.produce_weight_type == "conv":
            conv_input = encoder_raw_outputs.permute(0, 2, 1)
            conv_out = self.conv(conv_input)
            proj_input = conv_out.permute(0, 2, 1)
            proj_input = self.conv_dropout(proj_input)
            sig_input = self.weight_proj(proj_input)
            weight = torch.sigmoid(sig_input)
        else:
            sig_input = self.weight_proj(encoder_raw_outputs)
            weight = torch.sigmoid(sig_input)
        # weight has shape B x T x 1

        if encoder_padding_mask is not None:
            if transpose:
                not_padding_mask = ~encoder_padding_mask
            else:
                not_padding_mask = encoder_padding_mask  # need to improve here, cif padding mask is actually mark of token positions
        else:
            not_padding_mask = torch.ones(encoder_raw_outputs.shape[0], encoder_raw_outputs.shape[1]).to(encoder_raw_outputs.device)

        weight = torch.squeeze(weight, dim=-1) * not_padding_mask.int()  # weight has shape B x T
        org_weight = weight

        # Sum weights
        if self.training and self.apply_scaling and target_lengths is not None:
            # Conduct scaling when training
            weight_sum = weight.sum(-1)             # weight_sum has shape B
            normalize_scalar = torch.unsqueeze(
                target_lengths / weight_sum, -1)    # normalize_scalar has shape B x 1
            weight = weight * normalize_scalar

        # Integrate and fire
        batch_size = encoder_raw_outputs.size(0)
        max_length = encoder_raw_outputs.size(1)
        encoder_embed_dim = encoder_raw_outputs.size(2)
        padding_start_id = not_padding_mask.sum(-1)  # shape B

        # Initialize
        accumulated_weights = torch.zeros(batch_size, 0).cuda()
        accumulated_states = torch.zeros(batch_size, 0, encoder_embed_dim).cuda()
        fired_states = torch.zeros(batch_size, 0, encoder_embed_dim).cuda()
        fired_positions = torch.zeros(batch_size, 0).cuda()

        # Begin integrate and fire
        for i in range(max_length):
            # Get previous states from the recorded tensor
            prev_accumulated_weight = torch.zeros([batch_size]).cuda() if i == 0 else accumulated_weights[:, i - 1]
            prev_accumulated_state = \
                torch.zeros([batch_size, encoder_embed_dim]).cuda() if i == 0 else accumulated_states[:, i - 1, :]

            # Decide whether to fire a boundary
            cur_is_fired = ((prev_accumulated_weight + weight[:, i]) >= self.cif_threshold).unsqueeze(dim=-1)
            # cur_is_fired with shape B x 1

            # Update the accumulated weights
            cur_weight = torch.unsqueeze(weight[:, i], -1)
            # cur_weight has shape B x 1
            prev_accumulated_weight = torch.unsqueeze(prev_accumulated_weight, -1)
            # prev_accumulated_weight also has shape B x 1
            remained_weight = torch.ones_like(prev_accumulated_weight).cuda() - prev_accumulated_weight
            # remained_weight with shape B x 1

            # Obtain the accumulated weight of current step
            cur_accumulated_weight = torch.where(
                cur_is_fired,
                cur_weight - remained_weight,
                cur_weight + prev_accumulated_weight)  # B x 1

            # Obtain accumulated state of current step
            cur_accumulated_state = torch.where(
                cur_is_fired.repeat(1, encoder_embed_dim),
                (cur_weight - remained_weight) * encoder_raw_outputs[:, i, :],
                prev_accumulated_state + cur_weight * encoder_raw_outputs[:, i, :])  # B x C

            # Obtain fired state of current step:
            # firing locations has meaningful representations, while non-firing locations is all-zero embeddings
            cur_fired_state = torch.where(
                cur_is_fired.repeat(1, encoder_embed_dim),
                prev_accumulated_state + remained_weight * encoder_raw_outputs[:, i, :],
                torch.zeros([batch_size, encoder_embed_dim]).cuda())  # B x C

            # Handle the tail
            if (not self.training) and self.apply_tail_handling:
                # When encoder output position exceeds the max valid position,
                # if accumulated weights is greater than 0.6 (or some other value),
                # current state should be reserved, otherwise it is discarded.
                cur_fired_state = torch.where(
                    i == padding_start_id.unsqueeze(dim=-1).repeat([1, encoder_embed_dim]),
                    # shape B x C
                    torch.where(
                        cur_accumulated_weight.repeat([1, encoder_embed_dim]) <= 0.6,
                        # shape B x C
                        torch.zeros([batch_size, encoder_embed_dim]).cuda(),
                        # less equal than 0.6, discarded.
                        cur_accumulated_state / (cur_accumulated_weight + 1e-10)
                        # bigger than 0.6, normalized and kept.
                    ), cur_fired_state)
                # shape B x T

            # For normal condition, including both training and evaluation
            # Mask padded locations with all-zero embeddings
            cur_fired_state = torch.where(
                torch.full([batch_size, encoder_embed_dim], i).cuda() >
                padding_start_id.unsqueeze(dim=-1).repeat([1, encoder_embed_dim]),
                torch.zeros([batch_size, encoder_embed_dim]).cuda(), cur_fired_state)

            # Update accumulated arguments
            accumulated_weights = torch.cat(
                (accumulated_weights, cur_accumulated_weight), 1)  # B x T_c
            accumulated_states = torch.cat(
                (accumulated_states, torch.unsqueeze(cur_accumulated_state, 1)), 1)  # B x T_c x C
            fired_states = torch.cat(
                (fired_states, torch.unsqueeze(cur_fired_state, 1)), 1)  # B x T_c x C

        # Extract cif_outputs for each utterance
        fired_marks = (torch.abs(fired_states).sum(-1) != 0.0).int()    # B x T_c
        fired_utt_length = fired_marks.sum(-1)  # B
        fired_max_length = fired_utt_length.max().int()  # The maximum of fired times in current batch
        cif_outputs = torch.zeros([0, fired_max_length, encoder_embed_dim]).cuda()  # Initialize cif outputs

        def dynamic_partition(data: torch.Tensor, partitions: torch.Tensor, num_partitions=None):
            assert len(partitions.shape) == 1, "Only one dimensional partitions supported"
            assert (data.shape[0] == partitions.shape[0]), "Partitions requires the same size as data"
            if num_partitions is None:
                num_partitions = max(torch.unique(partitions))
            return [data[partitions == index] for index in range(num_partitions)]

        # Loop over all samples
        for j in range(batch_size):
            # Get information of j-th sample
            cur_utt_fired_mark = fired_marks[j, :]
            cur_utt_fired_state = fired_states[j, :, :]
            cur_utt_outputs = dynamic_partition(cur_utt_fired_state, cur_utt_fired_mark, 2)
            cur_utt_output = cur_utt_outputs[1]             # Get integrated representations
            cur_utt_length = cur_utt_output.size(0)         # The total number of firing
            pad_length = fired_max_length - cur_utt_length  # Get padded length
            cur_utt_output = torch.cat(
                (cur_utt_output, torch.full([pad_length, encoder_embed_dim], 0.0).cuda()), dim=0
            )  # Pad current utterance cif outputs to fired_max_length
            cur_utt_output = torch.unsqueeze(cur_utt_output, 0)
            # Reshape to 1 x T_c x C

            # Concatenate cur_utt_output and cif_outputs along batch axis
            cif_outputs = torch.cat([cif_outputs, cur_utt_output], 0)

        cif_out_padding_mask = (torch.abs(cif_outputs).sum(-1) != 0.0).long()
        # cif_out_padding_mask has shape B x T_c, where locations with value 0 is False.

        if self.training:
            quantity_out = org_weight.sum(-1)
        else:
            quantity_out = weight.sum(-1)

        cif_outputs = cif_outputs.type_as(encoder_raw_outputs)
        if self.cif_output_dim != encoder_embed_dim:
            # if C_c != C
            cif_outputs = self.cif_output_proj(cif_outputs)
            # B x T_c x C_c

        return {
            "cif_out": cif_outputs,  # B x T_c x C_c
            "padding_mask": cif_out_padding_mask,  # B x T_c
            "loss_pen": quantity_out,  # B
            "fired_marks": fired_marks  # B x T_c
        }

class CtcConstrainedCifMiddleware(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Get configurations related to continuous integrate-and-fire
        self.cif_threshold = cfg.cif_threshold
        self.cif_output_dim = cfg.cif_embedding_dim
        self.encoder_embed_dim = cfg.encoder_embed_dim
        self.produce_weight_type = cfg.produce_weight_type
        self.apply_scaling = cfg.apply_scaling
        self.apply_tail_handling = cfg.apply_tail_handling
        # self.tail_handling_firing_threshold = 0.99
        # print(self.tail_handling_firing_threshold)
        self.add_cif_ctxt_layers = cfg.add_cif_ctxt_layers

        # Build weight projection layer to compute weight from encoder outputs
        if self.produce_weight_type == "dense":
            self.dense_proj = Linear(self.encoder_embed_dim, cfg.dense_cif_units_num).cuda()
            self.weight_proj = Linear(cfg.dense_cif_units_num, 1).cuda()
        elif self.produce_weight_type == "conv":
            self.cif_conv_layer_num = cfg.conv_cif_layer_num
            self.conv = torch.nn.Conv1d(
                self.encoder_embed_dim,
                cfg.conv_cif_output_channels_num,
                cfg.conv_cif_width,
                stride=1, padding=int(cfg.conv_cif_width / 2),
                dilation=1, groups=1,
                bias=True, padding_mode='zeros'
            ).cuda()
            self.conv_dropout = torch.nn.Dropout(p=cfg.conv_cif_dropout).cuda()
            self.weight_proj = Linear(cfg.conv_cif_output_channels_num, 1).cuda()
        else:
            self.weight_proj = Linear(self.encoder_embed_dim, 1).cuda()

        # Build the final projection layer for cif outputs
        if self.cif_output_dim != self.encoder_embed_dim:
            self.cif_output_proj = Linear(self.encoder_embed_dim, self.cif_output_dim, bias=False).cuda()

        # Build cif contextual layers
        if self.add_cif_ctxt_layers:
            self.cif_ctxt_embed_dim = cfg.cif_ctxt_embed_dim
            self.cif_ctxt_stacks = nn.ModuleList([
                TransformerSentenceEncoderLayer(
                    embedding_dim=cfg.cif_ctxt_embed_dim,
                    ffn_embedding_dim=cfg.cif_ctxt_ffn_embed_dim,
                    num_attention_heads=cfg.cif_ctxt_attention_heads,
                    dropout=cfg.cif_ctxt_dropout,
                    activation_dropout=cfg.cif_ctxt_activation_dropout,
                    attention_dropout=cfg.cif_ctxt_attention_dropout,
                    layer_norm_first=cfg.cif_ctxt_normalize_before
                ) for _ in range(cfg.cif_ctxt_layers)])

        # CTC Constrained training settings
        self.use_ctc_constraint = cfg.use_ctc_constraint
        if self.use_ctc_constraint:
            self.ctc_prob_threshold = cfg.ctc_prob_threshold

    def forward(self, encoder_outputs, target_lengths=None, input_lengths=None, ctc_logits=None):
        """
        Args:
            encoder_out: B x T x C
            encoder_padding_mask: B x T
            targets_length: B
            ctc_logits: B x T x V (including blank_token_id)
        """

        # Prepare inputs
        encoder_raw_out = encoder_outputs["encoder_raw_out"].transpose(0, 1)  # B x T x C
        encoder_padding_mask = encoder_outputs["padding_mask"]

        # Forward weight generation
        if self.produce_weight_type == "dense":
            proj_out = self.dense_proj(encoder_raw_out)
            act_proj_out = torch.relu(proj_out)
            sig_input = self.weight_proj(act_proj_out)
            weight = torch.sigmoid(sig_input)
            # weight has shape [batch_size, length, 1]
        elif self.produce_weight_type == "conv":
            conv_input = encoder_raw_out.permute(0, 2, 1)
            # Adjust the shape of convolution layer input [B, C_in, T]
            conv_out = self.conv(conv_input)
            # conv_out has shape [B, C_out, T]
            proj_input = conv_out.permute(0, 2, 1)
            proj_input = self.conv_dropout(proj_input)
            # Adjust conv output to shape [B, T, C_cif]
            sig_input = self.weight_proj(proj_input)
            weight = torch.sigmoid(sig_input)
        else:
            sig_input = self.weight_proj(encoder_raw_out)
            weight = torch.sigmoid(sig_input)

        if encoder_padding_mask is not None:
            not_padding_mask = ~encoder_padding_mask
        else:
            not_padding_mask = torch.ones(encoder_raw_out.shape[0], encoder_raw_out.shape[1]).to(encoder_raw_out.device)
        weight = torch.squeeze(weight, dim=-1) * not_padding_mask.int()  # weight has shape B x T
        org_weight = weight

        # Sum weights
        if self.training and self.apply_scaling and target_lengths is not None:
            # if self.apply_scaling and target_lengths is not None:   # For validation debugging
            # Conduct scaling when training
            # (target_lengths + 1 because this target_lengths does not take <eos> into consideration)
            weight_sum = weight.sum(-1)  # weight_sum has shape [batch_size]
            normalize_scalar = torch.unsqueeze(
                target_lengths / weight_sum, -1)  # normalize_scalar has shape [batch_size, 1]
            # weight = weight * normalize_scalar
            weight = weight * normalize_scalar

            # TODO: Check weight
            # print("weight_sum: ", weight_sum)
            # print("normalize_scalar: ", normalize_scalar)
            # print("weight: ", weight)

        ctc_border_marks = None
        if self.use_ctc_constraint and ctc_logits is not None:
            ctc_probs = utils.softmax(ctc_logits.transpose(0, 1).float(), dim=-1)  # B x T x V

            # FIXME: remember the default blank id should be <bos> id (0)
            blank_probs = ctc_probs[:, :, 0]  # B x T

            non_blank_probs = 1.0 - blank_probs  # B x T
            ctc_border_marks = (non_blank_probs > self.ctc_prob_threshold).int()  # B x T
            # Seems like [[0,0,0,0,1,0,1], ...]

        # Integrate and fire
        batch_size = encoder_raw_out.size(0)
        max_length = encoder_raw_out.size(1)
        encoder_embed_dim = encoder_raw_out.size(2)
        padding_start_id = not_padding_mask.sum(-1)  # B

        # Initialize
        accumulated_weights = torch.zeros(batch_size, 0, dtype=encoder_raw_out.dtype).cuda()
        accumulated_states = torch.zeros(batch_size, 0, encoder_embed_dim, dtype=encoder_raw_out.dtype).cuda()
        fired_states = torch.zeros(batch_size, 0, encoder_embed_dim, dtype=encoder_raw_out.dtype).cuda()
        ctc_accum_weights = torch.zeros(batch_size, 0, dtype=encoder_raw_out.dtype).cuda() \
            if self.use_ctc_constraint else None  # B x T

        # Begin integrate and fire
        for i in range(max_length):
            # Get previous states from the recorded tensor
            prev_accumulated_weight = torch.zeros([batch_size], dtype=encoder_raw_out.dtype).cuda() \
                if i == 0 else accumulated_weights[:, i - 1]
            prev_accumulated_state = torch.zeros([batch_size, encoder_embed_dim], dtype=encoder_raw_out.dtype).cuda() \
                if i == 0 else accumulated_states[:, i - 1, :]

            # Decide whether positioning a boundary
            cur_is_fired = ((prev_accumulated_weight + weight[:, i]) >= self.cif_threshold).unsqueeze(dim=-1)
            # cur_is_fired with shape [batch_size, 1]

            # Update the accumulated weights by considering whether positioning a boundary
            cur_weight = torch.unsqueeze(weight[:, i], -1)
            # cur_weight has shape [batch_size, 1]
            prev_accumulated_weight = torch.unsqueeze(prev_accumulated_weight, -1)
            # prev_accumulated_weight also has shape [batch_size ,1]
            remained_weight = \
                torch.ones_like(prev_accumulated_weight, dtype=encoder_raw_out.dtype).cuda() - prev_accumulated_weight
            # remained_weight with shape [batch_size ,1]

            # Obtain the accumulated weight of current step
            cur_accumulated_weight = torch.where(
                cur_is_fired,
                cur_weight - remained_weight,
                cur_weight + prev_accumulated_weight)  # [B, 1]

            cur_ctc_accum_weight = None
            if self.use_ctc_constraint and ctc_border_marks is not None:
                if i == 0:
                    prev_ctc_accum_weight = torch.zeros([batch_size], dtype=encoder_raw_out.dtype).cuda()  # B
                else:
                    prev_ctc_border_marks = ctc_border_marks[:, i - 1]  # B
                    prev_ctc_accum_weight = torch.where(
                        prev_ctc_border_marks.float() == 1.0,  # B
                        torch.zeros([batch_size], dtype=encoder_raw_out.dtype).cuda(),  # B
                        ctc_accum_weights[:, i - 1]  # B
                    )  # B x 1
                cur_ctc_accum_weight = prev_ctc_accum_weight.unsqueeze(-1) + cur_weight

            # Obtain accumulated state of current step
            cur_accumulated_state = torch.where(
                cur_is_fired.repeat(1, encoder_embed_dim),
                (cur_weight - remained_weight) * encoder_raw_out[:, i, :],
                prev_accumulated_state + cur_weight * encoder_raw_out[:, i, :])  # B x C

            # Obtain fired state of current step
            # firing locations has meaningful representations, while non-firing locations is all-zero embeddings
            cur_fired_state = torch.where(
                cur_is_fired.repeat(1, encoder_embed_dim),
                prev_accumulated_state + remained_weight * encoder_raw_out[:, i, :],
                torch.zeros([batch_size, encoder_embed_dim], dtype=encoder_raw_out.dtype).cuda(),
            )  # B x C

            # TODO: Check state
            # if i == (max_length - 1):
            #     print("prev_accumulated_weight: ", prev_accumulated_weight)
            #     print("cur_weight: ", cur_weight)
            #     print("remained_weight: ", remained_weight)
            #     print("cur_fired_state: ", cur_fired_state[0, :10])

            # Handling the speech tail by rounding up and down
            if (not self.training) and self.apply_tail_handling:
                # When encoder output position exceeds the max valid position,
                # if accumulated weights is greater than tail_handling_firing_threshold,
                # current state should be reserved, otherwise it is discarded.

                # TODO: Check evaluate
                # print("______________________")
                # print(i)
                # print("cur_accumulated_state:", cur_accumulated_state[:, :10])
                # print("cur_accumulated_weight: ", cur_accumulated_weight)
                # print(i == padding_start_id)
                cur_fired_state = torch.where(
                    i == padding_start_id.unsqueeze(dim=-1).repeat([1, encoder_embed_dim]),  # B x C
                    torch.where(
                        cur_accumulated_weight.repeat([1, encoder_embed_dim]) <= 0.6,  # B x C
                        torch.zeros([batch_size, encoder_embed_dim], dtype=encoder_raw_out.dtype).cuda(),
                        # less equal than tail_handling_firing_threshold, discarded.
                        cur_accumulated_state / (cur_accumulated_weight + 1e-10)
                        # bigger than tail_handling_firing_threshold, normalized and kept.
                        # eps = 1e-10 for preveting overflow.
                    ),
                    cur_fired_state,
                )  # B x C

            # For normal condition, including both training and evaluation
            # Mask padded locations with all-zero embeddings
            cur_fired_state = torch.where(
                torch.full([batch_size, encoder_embed_dim], i, dtype=encoder_raw_out.dtype).cuda() >
                padding_start_id.unsqueeze(dim=-1).repeat([1, encoder_embed_dim]),  # B x C
                torch.zeros([batch_size, encoder_embed_dim], dtype=encoder_raw_out.dtype).cuda(),
                cur_fired_state)

            # Update accumulated arguments
            accumulated_weights = torch.cat(
                (accumulated_weights, cur_accumulated_weight), 1)  # B x T
            accumulated_states = torch.cat(
                (accumulated_states, torch.unsqueeze(cur_accumulated_state, 1)), 1)  # shape = [B, L, D]
            fired_states = torch.cat(
                (fired_states, torch.unsqueeze(cur_fired_state, 1)), 1)  # shape = [B, L, D]
            if self.use_ctc_constraint and cur_ctc_accum_weight is not None:
                ctc_accum_weights = torch.cat(
                    [ctc_accum_weights, cur_ctc_accum_weight], -1)  # B x T

        # Extracts cif_outputs for each utterance
        fired_marks = (torch.abs(fired_states).sum(-1) != 0.0).int()  # B x T
        fired_utt_length = fired_marks.sum(-1)  # B
        fired_max_length = fired_utt_length.max().int()  # The maximum of fired times in current batch
        cif_outputs = torch.zeros(
            [0, fired_max_length, encoder_embed_dim], dtype=encoder_raw_out.dtype).cuda()  # Initialize cif outputs
        cif_durations = torch.zeros(
            [0, fired_max_length], dtype=torch.int32).cuda()  # Initialize cif durations

        def dynamic_partition(data: torch.Tensor, partitions: torch.Tensor, num_partitions=None):
            assert len(partitions.shape) == 1, "Only one dimensional partitions supported"
            assert (data.shape[0] == partitions.shape[0]), "Partitions requires the same size as data"
            if num_partitions is None:
                num_partitions = max(torch.unique(partitions))
            return [data[partitions == i] for i in range(num_partitions)]

        for j in range(batch_size):
            # Get information of j-th sample
            cur_utt_fired_mark = fired_marks[j, :]
            cur_utt_fired_state = fired_states[j, :, :]
            cur_utt_outputs = dynamic_partition(cur_utt_fired_state, cur_utt_fired_mark, 2)
            cur_utt_output = cur_utt_outputs[1]  # Get integrated representations
            cur_utt_length = cur_utt_output.size(0)  # The total number of firing
            pad_length = fired_max_length - cur_utt_length  # Calculate padding length
            cur_utt_output = torch.cat(
                (cur_utt_output, torch.full([pad_length, encoder_embed_dim], 0.0, dtype=encoder_raw_out.dtype).cuda()),
                dim=0
            )  # Pad current utterance cif outputs to fired_max_length
            cur_utt_output = torch.unsqueeze(cur_utt_output, 0)
            # Reshape to [1, fired_max_length, encoder_embed_dim]

            # Concatenate cur_utt_output and cif_outputs along batch axis
            cif_outputs = torch.cat([cif_outputs, cur_utt_output], 0)

            # Collect cif durations
            cur_fired_indices = torch.nonzero(cur_utt_fired_mark)[:, -1]
            shifted_cur_fired_indices = torch.cat(
                [-1 * torch.ones([1], dtype=torch.int32).cuda(), cur_fired_indices], dim=-1
            )[:cur_fired_indices.size(0)]
            cur_cif_durations = cur_fired_indices - shifted_cur_fired_indices
            cur_cif_durations = torch.cat(
                (cur_cif_durations, torch.full([pad_length], 0, dtype=torch.int32).cuda()), dim=0
            ).unsqueeze(dim=0)
            cif_durations = torch.cat([cif_durations, cur_cif_durations], dim=0)  # cancat at batch axis

        cif_out_padding_mask = (torch.abs(cif_outputs).sum(-1) != 0.0).int()
        # cif_out_padding_mask shape = [batch_size, fired_max_length], where locations with value 0 is False.

        # TODO： Check
        # print("_____cif check: ")
        # print("fired_marks: ", fired_marks.size(), fired_marks)
        # print("fired_utt_length :", fired_utt_length)
        # print("cur_cif_durations: ", cur_cif_durations.size(), cur_cif_durations)
        # print("weight: ", weight.sum(-1), weight)

        if self.training:
            # In training phase, use the sum of original weights
            # as quantity out for quantity loss.
            quantity_out = org_weight.sum(-1)
        else:
            quantity_out = weight.sum(-1)

        if self.cif_output_dim != encoder_embed_dim:
            cif_outputs = self.cif_output_proj(cif_outputs)

        ctxt_cif_outputs = None
        if self.add_cif_ctxt_layers and self.cif_output_dim == self.cif_ctxt_embed_dim:
            x = cif_outputs.transpose(0, 1)
            padding_mask = ~cif_out_padding_mask.bool()
            for layer in self.cif_ctxt_stacks:
                x, _ = layer(x, self_attn_padding_mask=padding_mask, need_weights=False)
            ctxt_cif_outputs = x.transpose(0, 1)

        ctc_align_outputs = None
        if self.use_ctc_constraint and ctc_accum_weights is not None:
            org_ctc_align_outputs = ctc_accum_weights * ctc_border_marks  # B x T_a
            ctc_align_max_len = ctc_border_marks.size(1)
            ctc_align_outputs = torch.zeros(
                [0, ctc_align_max_len]).type_as(org_ctc_align_outputs).cuda()
            for k in range(batch_size):
                cur_border_marks = ctc_border_marks[k, :]  # T
                cur_borders_num = cur_border_marks.sum()  # 1
                cur_ctc_accum_weight = ctc_accum_weights[k, :]  # T_a
                compressed_ctc_weight = cur_ctc_accum_weight[cur_border_marks.float() != 0.0]
                pad_length = ctc_align_max_len - cur_borders_num  # get padding length
                padded_compressed_ctc_weight = torch.cat([
                    compressed_ctc_weight,
                    torch.full([pad_length], 0.0).type_as(compressed_ctc_weight).cuda()], dim=0
                ).unsqueeze(0)  # 1 x T
                ctc_align_outputs = torch.cat(
                    [ctc_align_outputs, padded_compressed_ctc_weight], dim=0)  # B x T

                # FIXME: fix it if there are any bugs, Check it thoroughly & carefully
                # print(ctc_align_max_len)
                # print(cur_borders_num)
                # print(pad_length)
                # print(ctc_align_outputs.size())
                # print(padded_compressed_ctc_weight.size())

        return {
            "cif_out": cif_outputs,  # shape = [batch_size, fired_max_length, cif_output_dim]
            "padding_mask": cif_out_padding_mask,  # shape = [batch_size, fired_max_length]
            "loss_pen": quantity_out,  # shape = [batch_size]
            "ctc_align_outputs": ctc_align_outputs,  # B x T
        }


class LexiconConstraintCif(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Load configurations
        self.cif_threshold = cfg.cif_threshold
        self.cif_output_dim = cfg.cif_embedding_dim
        self.encoder_embed_dim = cfg.encoder_embed_dim
        self.produce_weight_type = cfg.produce_weight_type
        self.apply_scaling = cfg.apply_scaling
        self.apply_tail_handling = cfg.apply_tail_handling

        # Build weight generator
        if self.produce_weight_type == "dense":
            self.dense_proj = Linear(self.encoder_embed_dim, cfg.dense_cif_units_num).cuda()
            self.weight_proj = Linear(cfg.dense_cif_units_num, 1).cuda()
        elif self.produce_weight_type == "conv":
            self.cif_conv_layer_num = cfg.conv_cif_layer_num
            self.conv = torch.nn.Conv1d(
                self.encoder_embed_dim,
                cfg.conv_cif_output_channels_num,
                cfg.conv_cif_width,
                stride=1, padding=2,
                dilation=1, groups=1,
                bias=True, padding_mode='zeros'
            ).cuda()
            self.conv_dropout = torch.nn.Dropout(p=cfg.conv_cif_dropout).cuda()
            self.weight_proj = Linear(cfg.conv_cif_output_channels_num, 1).cuda()
        else:
            self.weight_proj = Linear(self.encoder_embed_dim, 1).cuda()

        # Build the final projection layer for cif outputs if necessary
        if self.cif_output_dim != self.encoder_embed_dim:
            self.cif_output_proj = Linear(self.encoder_embed_dim, self.cif_output_dim, bias=False).cuda()

    def forward(self, encoder_outputs, target_lengths=None, transpose=True, phone_segs=None, integrate_weights=None):
        """
        Args:
            encoder_outputs: a dictionary that includes
                encoder_raw_out: the raw outputs of acoustic encoder, with shape B x T x C
                encoder_padding_mask: the padding mask of encoder outputs, with shape B x T
            target_lengths: the length of targets (necessary when training), with shape B
        Return:
            A dictionary:
                cif_out:
                cif_out_padding_mask:
                quantity_out:
        """
        # Gather inputs
        encoder_raw_outputs = encoder_outputs["encoder_raw_out"]
        if transpose:
            encoder_raw_outputs = encoder_raw_outputs.transpose(0, 1) # T x B x C -> B x T x C
        encoder_padding_mask = encoder_outputs["padding_mask"]  # B x T

        # Produce weights
        if self.produce_weight_type == "dense":
            proj_out = self.dense_proj(encoder_raw_outputs)
            act_proj_out = torch.relu(proj_out)
            sig_input = self.weight_proj(act_proj_out)
            weight = torch.sigmoid(sig_input)
        elif self.produce_weight_type == "conv":
            conv_input = encoder_raw_outputs.permute(0, 2, 1)
            conv_out = self.conv(conv_input)
            proj_input = conv_out.permute(0, 2, 1)
            proj_input = self.conv_dropout(proj_input)
            sig_input = self.weight_proj(proj_input)
            weight = torch.sigmoid(sig_input)
        else:
            sig_input = self.weight_proj(encoder_raw_outputs)
            weight = torch.sigmoid(sig_input)
        # weight has shape B x T x 1

        if encoder_padding_mask is not None:
            if transpose:
                not_padding_mask = ~encoder_padding_mask
            else:
                not_padding_mask = encoder_padding_mask  # need to improve here, cif padding mask is actually mark of token positions
        else:
            not_padding_mask = torch.ones(encoder_raw_outputs.shape[0], encoder_raw_outputs.shape[1]).to(encoder_raw_outputs.device)

        weight = torch.squeeze(weight, dim=-1) * not_padding_mask.int()  # weight has shape B x T
        org_weight = weight

        # Sum weights
        if self.training and self.apply_scaling and target_lengths is not None:
            # Conduct scaling when training
            weight_sum = weight.sum(-1)             # weight_sum has shape B
            normalize_scalar = torch.unsqueeze(
                target_lengths / weight_sum, -1)    # normalize_scalar has shape B x 1
            weight = weight * normalize_scalar

        # Integrate and fire
        batch_size = encoder_raw_outputs.size(0)
        max_length = encoder_raw_outputs.size(1)
        encoder_embed_dim = encoder_raw_outputs.size(2)
        padding_start_id = not_padding_mask.sum(-1)  # shape B

        # Initialize
        accumulated_weights = torch.zeros(batch_size, 0).cuda()
        accumulated_states = torch.zeros(batch_size, 0, encoder_embed_dim).cuda()
        fired_states = torch.zeros(batch_size, 0, encoder_embed_dim).cuda()
        fired_positions = torch.zeros(batch_size, 0).cuda()

        # Begin integrate and fire
        for i in range(max_length):
            # Get previous states from the recorded tensor
            prev_accumulated_weight = torch.zeros([batch_size]).cuda() if i == 0 else accumulated_weights[:, i - 1]
            prev_accumulated_state = \
                torch.zeros([batch_size, encoder_embed_dim]).cuda() if i == 0 else accumulated_states[:, i - 1, :]

            # Decide whether to fire a boundary
            cur_is_fired = ((prev_accumulated_weight + weight[:, i]) >= self.cif_threshold).unsqueeze(dim=-1)
            # cur_is_fired with shape B x 1

            # Update the accumulated weights
            cur_weight = torch.unsqueeze(weight[:, i], -1)
            # cur_weight has shape B x 1
            prev_accumulated_weight = torch.unsqueeze(prev_accumulated_weight, -1)
            # prev_accumulated_weight also has shape B x 1
            remained_weight = torch.ones_like(prev_accumulated_weight).cuda() - prev_accumulated_weight
            # remained_weight with shape B x 1

            # Obtain the accumulated weight of current step
            cur_accumulated_weight = torch.where(
                cur_is_fired,
                cur_weight - remained_weight,
                cur_weight + prev_accumulated_weight)  # B x 1

            # Obtain accumulated state of current step
            cur_accumulated_state = torch.where(
                cur_is_fired.repeat(1, encoder_embed_dim),
                (cur_weight - remained_weight) * encoder_raw_outputs[:, i, :],
                prev_accumulated_state + cur_weight * encoder_raw_outputs[:, i, :])  # B x C

            # Obtain fired state of current step:
            # firing locations has meaningful representations, while non-firing locations is all-zero embeddings
            cur_fired_state = torch.where(
                cur_is_fired.repeat(1, encoder_embed_dim),
                prev_accumulated_state + remained_weight * encoder_raw_outputs[:, i, :],
                torch.zeros([batch_size, encoder_embed_dim]).cuda())  # B x C

            # Handle the tail
            if (not self.training) and self.apply_tail_handling:
                # When encoder output position exceeds the max valid position,
                # if accumulated weights is greater than 0.6 (or some other value),
                # current state should be reserved, otherwise it is discarded.
                cur_fired_state = torch.where(
                    i == padding_start_id.unsqueeze(dim=-1).repeat([1, encoder_embed_dim]),
                    # shape B x C
                    torch.where(
                        cur_accumulated_weight.repeat([1, encoder_embed_dim]) <= 0.6,
                        # shape B x C
                        torch.zeros([batch_size, encoder_embed_dim]).cuda(),
                        # less equal than 0.6, discarded.
                        cur_accumulated_state / (cur_accumulated_weight + 1e-10)
                        # bigger than 0.6, normalized and kept.
                    ), cur_fired_state)
                # shape B x T

            # For normal condition, including both training and evaluation
            # Mask padded locations with all-zero embeddings
            cur_fired_state = torch.where(
                torch.full([batch_size, encoder_embed_dim], i).cuda() >
                padding_start_id.unsqueeze(dim=-1).repeat([1, encoder_embed_dim]),
                torch.zeros([batch_size, encoder_embed_dim]).cuda(), cur_fired_state)

            # Update accumulated arguments
            accumulated_weights = torch.cat(
                (accumulated_weights, cur_accumulated_weight), 1)  # B x T_c
            accumulated_states = torch.cat(
                (accumulated_states, torch.unsqueeze(cur_accumulated_state, 1)), 1)  # B x T_c x C
            fired_states = torch.cat(
                (fired_states, torch.unsqueeze(cur_fired_state, 1)), 1)  # B x T_c x C

        # Extract cif_outputs for each utterance
        fired_marks = (torch.abs(fired_states).sum(-1) != 0.0).int()    # B x T_c
        fired_utt_length = fired_marks.sum(-1)  # B
        fired_max_length = fired_utt_length.max().int()  # The maximum of fired times in current batch
        cif_outputs = torch.zeros([0, fired_max_length, encoder_embed_dim]).cuda()  # Initialize cif outputs

        def dynamic_partition(data: torch.Tensor, partitions: torch.Tensor, num_partitions=None):
            assert len(partitions.shape) == 1, "Only one dimensional partitions supported"
            assert (data.shape[0] == partitions.shape[0]), "Partitions requires the same size as data"
            if num_partitions is None:
                num_partitions = max(torch.unique(partitions))
            return [data[partitions == index] for index in range(num_partitions)]

        # Loop over all samples
        for j in range(batch_size):
            # Get information of j-th sample
            cur_utt_fired_mark = fired_marks[j, :]
            cur_utt_fired_state = fired_states[j, :, :]
            cur_utt_outputs = dynamic_partition(cur_utt_fired_state, cur_utt_fired_mark, 2)
            cur_utt_output = cur_utt_outputs[1]             # Get integrated representations
            cur_utt_length = cur_utt_output.size(0)         # The total number of firing
            pad_length = fired_max_length - cur_utt_length  # Get padded length
            cur_utt_output = torch.cat(
                (cur_utt_output, torch.full([pad_length, encoder_embed_dim], 0.0).cuda()), dim=0
            )  # Pad current utterance cif outputs to fired_max_length
            cur_utt_output = torch.unsqueeze(cur_utt_output, 0)
            # Reshape to 1 x T_c x C

            # Concatenate cur_utt_output and cif_outputs along batch axis
            cif_outputs = torch.cat([cif_outputs, cur_utt_output], 0)

        cif_out_padding_mask = (torch.abs(cif_outputs).sum(-1) != 0.0).long()
        # cif_out_padding_mask has shape B x T_c, where locations with value 0 is False.

        if self.training:
            for k in range(batch_size):
                cur_weight = org_weight[k, :]
                cur_integrate_weight = integrate_weights[k]
                cur_phone_segs = phone_segs[k].tolist()
                phone_pos = 0
                cur_lexicon_out = torch.zeros([1], dtype=encoder_raw_outputs.dtype).cuda()
                for i in range(len(cur_phone_segs)):
                    if i != len(cur_phone_segs) - 1:
                        lexicon_weight = cur_weight[phone_pos : phone_pos + cur_phone_segs[i]].sum()
                    else:
                        lexicon_weight = cur_weight[phone_pos :].sum()
                    cur_lexicon_out += torch.abs(lexicon_weight - cur_integrate_weight[i])
                    phone_pos += cur_phone_segs[i]

                if k == 0:
                    lexicon_out = cur_lexicon_out
                else:
                    lexicon_out = torch.cat([lexicon_out, cur_lexicon_out], -1)

            quantity_out = lexicon_out.sum(-1)
        else:
            for k in range(batch_size):
                cur_weight = weight[k, :]
                cur_integrate_weight = integrate_weights[k]
                cur_phone_segs = phone_segs[k].tolist()
                phone_pos = 0
                cur_lexicon_out = torch.zeros([1], dtype=encoder_raw_outputs.dtype).cuda()
                for i in range(len(cur_phone_segs)):
                    if i != len(cur_phone_segs) - 1:
                        lexicon_weight = cur_weight[phone_pos: phone_pos + cur_phone_segs[i]].sum()
                    else:
                        lexicon_weight = cur_weight[phone_pos:].sum()
                    cur_lexicon_out += torch.abs(lexicon_weight - cur_integrate_weight[i])
                    phone_pos += cur_phone_segs[i]

                if k == 0:
                    lexicon_out = cur_lexicon_out
                else:
                    lexicon_out = torch.cat([lexicon_out, cur_lexicon_out], -1)
            quantity_out = lexicon_out.sum(-1)

        cif_outputs = cif_outputs.type_as(encoder_raw_outputs)
        if self.cif_output_dim != encoder_embed_dim:
            # if C_c != C
            cif_outputs = self.cif_output_proj(cif_outputs)
            # B x T_c x C_c

        return {
            "cif_out": cif_outputs,  # B x T_c x C_c
            "padding_mask": cif_out_padding_mask,  # B x T_c
            "loss_pen": quantity_out,  # 1
            "fired_marks": fired_marks  # B x T_c
        }


class CifEncoder(Wav2VecEncoder):
    def __init__(self, cfg: Wav2Vec2CifSeq2SeqConfig, dictionary, output_size=None):
        super().__init__(cfg, output_size)

        self.cif = Cif(cfg)

        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_before": cfg.mask_channel_before,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
        }

        if cfg.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(cfg.w2v_path, arg_overrides)
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            w2v_args.criterion = None
            w2v_args.lr_scheduler = None
            cfg.w2v_args = w2v_args
        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(w2v_args)

        encoder_embed_d = w2v_args.model.encoder_embed_dim
        cif_embed_d = cfg.cif_embedding_dim

        target_dim = len(dictionary)
        decoder_dim = cfg.decoder_embed_dim

        self.proj_encoder_to_target = None
        if target_dim != encoder_embed_d:
            self.proj_encoder_to_target = Linear(encoder_embed_d, target_dim)

        self.proj_cif_to_decoder = None
        if decoder_dim != cif_embed_d:
            self.proj_cif_to_decoder = Linear(cif_embed_d, decoder_dim)

    def forward(self, source, padding_mask, target_lengths=None, **kwargs):
        """
        Retures:
            tuples:
                - Cif ouput to compute joint output
                - Encoder output to compute ctc loss when training
        """

        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
        }

        # logger.info("Input of Encoder network's type is {}".format(w2v_args["source"].type()))

        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            res = self.w2v_model.extract_features(**w2v_args)

            x = res["x"]
            padding_mask = res["padding_mask"]

            # B x T x C -> T x B x C
            x = x.transpose(0, 1)

        x = self.final_dropout(x)

        encoder_raw_out = x
        encoder_raw_outputs = {
            "encoder_raw_out": encoder_raw_out,
            "padding_mask": padding_mask,
        }

        # logger.info("Input of cif network's type is {}".format(x.type()))
        # logger.info("Encoder padding {}".format(padding_mask))

        if self.proj_encoder_to_target:
            x = self.proj_encoder_to_target(x)

        # logger.info("encoder out size {}".format(x.shape))
        encoder_outputs = {
            "encoder_out": x,  # T x B x C
            "padding_mask": padding_mask,
        }

        cif_outputs = self.cif(encoder_raw_outputs, target_lengths)
        cif_out = cif_outputs["cif_out"].type_as(x)
        cif_padding = cif_outputs["padding_mask"]
        loss_pen = cif_outputs["loss_pen"]

        if self.proj_cif_to_decoder:
            cif_out = self.proj_cif_to_decoder(cif_out)

        # logger.info("Cif padding {}".format(cif_padding))

        return {
            "cif_outputs":{
                "cif_out": cif_out,
                "padding_mask": cif_padding,  # B x T
                "loss_pen": loss_pen,
            },
            "encoder_outputs": encoder_outputs
        }


class CifDecoder(TransformerDecoder):
    def __init__(self,
                 cfg: Wav2Vec2CifSeq2SeqConfig,
                 dictionary,
                 embed_tokens,
                 no_encoder_attn=True):

        super().__init__(
                cfg,
                dictionary,
                embed_tokens,
                no_encoder_attn,
        )

        self.embed_out = nn.Parameter(
            torch.Tensor(len(dictionary), self.output_embed_dim)
        )
        nn.init.normal_(self.embed_out, mean=0, std=self.output_embed_dim ** -0.5)

    def forward(
        self, prev_output_tokens, encoder_out=None, incremental_state=None, **unused
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        prev_output_tokens = prev_output_tokens.long()
        x, extra = self.extract_features(
            prev_output_tokens, encoder_out, incremental_state
        )

        return x, extra

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        # project back to size of vocabulary
        return F.linear(features, self.embed_out)

    def recognize_beam(self, beam_size=None):
        """ Beam search for infer step"""



@register_model("wav2vec_cifencoder", dataclass=Wav2Vec2CifSeq2SeqConfig)
class Wav2Vec2CifEncoderModel(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder, tgt_dict_len, cfg: Wav2Vec2CifSeq2SeqConfig):
        super().__init__(encoder, decoder)

        self.blank_weight = cfg.blank_weight
        self.blank_mode = cfg.blank_mode
        self.output_embed_dim = cfg.encoder_embed_dim

        self.embed_out = nn.Parameter(
            torch.Tensor(tgt_dict_len, self.output_embed_dim)
        )
        nn.init.normal_(self.embed_out, mean=0, std=self.output_embed_dim ** -0.5)

    @classmethod
    def build_model(cls, cfg: Wav2Vec2CifSeq2SeqConfig, task: FairseqTask):
        """Build a new model instance."""

        assert (
            cfg.autoregressive
        ), "Please set task.autoregressive=true for seq2seq asr models"

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        encoder = cls.build_encoder(cfg, tgt_dict)
        decoder = cls.build_decoder(cfg, tgt_dict)

        return Wav2Vec2CifEncoderModel(encoder, decoder, len(tgt_dict), cfg)

    @classmethod
    def build_encoder(cls, cfg: Wav2Vec2CifSeq2SeqConfig, tgt_dict):
        return CifEncoder(cfg, tgt_dict)

    @classmethod
    def build_decoder(cls, cfg: Wav2Vec2Seq2SeqConfig, tgt_dict):
        return FairseqIncrementalDecoder(tgt_dict)

    def output_layer(self, features, **kwargs):
        return F.linear(features, self.embed_out)

    def forward(self, target_lengths=None, **kwargs):
        cifencoder_out = self.encoder(
            target_lengths=target_lengths if self.training else None,
            **kwargs
        )

        '''
        cifencoder_out = {
            "cif_outputs":{
                "cif_out": cif_out,
                "padding_mask": cif_padding,  # B x T
                "loss_pen": loss_pen,
            },
            "encoder_outputs": encoder_outputs
        }
        '''

        cif_outputs = cifencoder_out["cif_outputs"]
        cif_outputs["cif_out"] = self.output_layer(cif_outputs["cif_out"])

        return cifencoder_out

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    def get_logits(self, net_output, normalize=False):
        logits = net_output["cif_out"]

        if net_output["padding_mask"] is not None and net_output["padding_mask"].any():
            logits[net_output["padding_mask"]][...] = float("-inf")

        if normalize:
            logits = utils.log_softmax(logits.type_as(net_output["cif_out"]), dim=-1)

        return logits

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a cif model's output."""

        logits = self.get_logits(net_output)

        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    def get_encoder_logits(self, net_output, normalize=False):
        logits = net_output["encoder_out"]
        if self.blank_weight != 0:
            if self.blank_mode == "add":
                logits[..., 0] += self.blank_weight
            elif self.blank_mode == "set":
                logits[..., 0] = self.blank_weight
            else:
                raise Exception(f"invalid blank mode {self.blank_mode}")

        if net_output["padding_mask"] is not None and net_output["padding_mask"].any():
            logits[net_output["padding_mask"].T][..., 0] = float("inf")
            logits[net_output["padding_mask"].T][..., 1:] = float("-inf")

        if normalize:
            logits = utils.log_softmax(logits.type_as(net_output["encoder_out"]), dim=-1)

        return logits

    def get_encoder_normalized_probs(self, logits, log_probs):
        """Get normalized probabilities (or log probs) from a encoder's output."""

        if log_probs:
            return utils.log_softmax(logits, dim=-1)
        else:
            return utils.softmax(logits, dim=-1)

    def get_probs_from_logits(self, logits, log_probs=False):
        """Get normalized probabilities (or log probs) from logits."""
        if log_probs:
            return utils.log_softmax(logits, dim=-1)
        else:
            return utils.softmax(logits, dim=-1)

    def get_cif_output(self, target_lengths=None, **kwargs):
        cifencoder_out = self.encoder(target_lengths=target_lengths, **kwargs)
        return cifencoder_out["cif_outputs"]

    def step_forward_decoder(self,
                             prev_decoded_tokens,
                             cif_outputs,
                             incremental_state=None, **kwargs):
        cif_out = cif_outputs["cif_out"]

        # Project to vocabulary size
        cif_out = self.output_layer(cif_out)
        return cif_out


@dataclass
class Wav2Vec2MultiScaleCifConfig(Wav2Vec2Seq2SeqConfig):
    blank_weight: float = 0
    blank_mode: str = "add"

    use_pretrained_decoder: bool = field(
        default=False,
        metadata={"help": "whether to use pretrained language model"}
    )
    decoder_path: str = field(
        default=MISSING, metadata={"help": "path to pretrained language model"}
    )
    decoder_args: Any = None

    cif_threshold: float = field(
        default=1.0,
        metadata={"help": "threshold for cif integrate and fire"}
    )

    cif_embedding_dim: int = field(
        default=768,
        metadata={"help": "cif embedding dimension"}
    )
    produce_weight_type: str = field(
        default="conv",
        metadata={"help": "type of weight producing network: dense/conv"}
    )
    apply_scaling: bool = field(
        default=True,
        metadata={"help": "whether to apply scaling technique"}
    )
    apply_tail_handling: bool = field(
        default=True,
        metadata={"help": "whether to apply tail handling technique"}
    )
    conv_cif_layer_num: int = field(
        default=1,
        metadata={"help": "number of convolutional layers of cif module"}
    )
    conv_cif_output_channels_num: int = field(
        default=768,
        metadata={"help": "dimension of output of convolutional layer of cif module"}
    )
    conv_cif_width: int = field(
        default=5,
        metadata={"help": "kernel size of convolutional layer of cif module"}
    )
    conv_cif_dropout: float = field(
        default=0.1,
    )
    # CtcConstraintCif settings
    add_cif_ctxt_layers: bool = field(
        default=False,
        metadata={"help": "whether to apply cif transformer context layers"}
    )
    cif_ctxt_embed_dim: int = field(
        default=768,
    )
    cif_ctxt_ffn_embed_dim: int = field(
        default=2048,
    )
    cif_ctxt_attention_heads: int = field(
        default=4,
    )
    cif_ctxt_dropout: float = field(
        default=0.1,
    )
    cif_ctxt_activation_dropout: float = field(
        default=0.1,
    )
    cif_ctxt_attention_dropout: float = field(
        default=0.1,
    )
    cif_ctxt_normalize_before: bool = field(
        default=True,
    )
    calculate_ctc_logits: bool = field(
        default=True,
    )
    use_ctc_constraint: bool = field(
        default=True,
    )
    ctc_prob_threshold: float = field(
        default=0.5,
    )


    phone_cif_embedding_dim: int = field(
        default=768,
        metadata={"help": "phone cif embedding dimension"}
    )
    phone_produce_weight_type: str = field(
        default="conv",
        metadata={"help": "type of weight producing network for phone cif: dense/conv"}
    )
    phone_apply_scaling: bool = field(
        default=True,
        metadata={"help": "whether to apply scaling technique in phone cif"}
    )
    phone_apply_tail_handling: bool = field(
        default=True,
        metadata={"help": "whether to apply tail handling technique in phone cif"}
    )
    phone_conv_cif_layer_num: int = field(
        default=1,
        metadata={"help": "number of convolutional layers of phone cif module"}
    )
    phone_conv_cif_output_channels_num: int = field(
        default=768,
        metadata={"help": "dimension of output of convolutional layer of phone cif module"}
    )
    phone_conv_cif_width: int = field(
        default=5,
        metadata={"help": "kernel size of convolutional layer of phone cif module"}
    )
    phone_conv_cif_dropout: float = field(
        default=0.1,
    )

    char_cif_embedding_dim: int = field(
        default=768,
        metadata={"help": "character cif embedding dimension"}
    )
    char_produce_weight_type: str = field(
        default="conv",
        metadata={"help": "type of weight producing network for char cif: dense/conv"}
    )
    char_apply_scaling: bool = field(
        default=True,
        metadata={"help": "whether to apply scaling technique in char cif"}
    )
    char_apply_tail_handling: bool = field(
        default=True,
        metadata={"help": "whether to apply tail handling technique in char cif"}
    )
    char_conv_cif_layer_num: int = field(
        default=1,
        metadata={"help": "number of convolutional layers of char cif module"}
    )
    char_conv_cif_output_channels_num: int = field(
        default=768,
        metadata={"help": "dimension of output of convolutional layer of char cif module"}
    )
    char_conv_cif_width: int = field(
        default=5,
        metadata={"help": "kernel size of convolutional layer of char cif module"}
    )
    char_conv_cif_dropout: float = field(
        default=0.1,
    )

    apply_joint_cnn: bool = field(
        default=False,
        metadata={"help": "whether to send cif output and decoder output through cnn"}
    )
    joint_conv_width: int = field(
        default=3,
        metadata={"help": "kernel size of convolutional layer of joint cnn"}
    )
    joint_conv_dropout: float = field(
        default=0.1,
    )

class MultiscaleCifEncoder(CifEncoder):
    def __init__(self,
                 cfg: Wav2Vec2MultiScaleCifConfig,
                 dictionary,
                 output_size=None):
        super().__init__(cfg, dictionary, output_size)

        phone_targ_d = len(dictionary)

        self.proj_to_phone_d = None
        if cfg.phone_cif_embedding_dim != phone_targ_d:
            self.proj_to_phone_d = Linear(cfg.phone_cif_embedding_dim, phone_targ_d)

        self.proj_cif_c2p = None
        if cfg.phone_cif_embedding_dim != output_size:
            self.proj_cif_c2p = Linear(cfg.phone_cif_embedding_dim, output_size)

        self.use_ctc_constraint = cfg.use_ctc_constraint
        if self.use_ctc_constraint:
            self.cif = CtcConstrainedCifMiddleware(cfg)
            self.ctc_prob_threshold = cfg.ctc_prob_threshold
        else:
            self.cif = Cif(cfg)


    def forward(self, source, padding_mask, target_lengths=None, **kwargs):
        """
        Retures:
            tuples:
                - Cif ouput to compute joint output
                - Encoder output to compute ctc loss when training
        """

        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
        }

        # logger.info("Input of Encoder network's type is {}".format(w2v_args["source"].type()))

        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            res = self.w2v_model.extract_features(**w2v_args)

            x = res["x"]
            padding_mask = res["padding_mask"]

            # B x T x C -> T x B x C
            x = x.transpose(0, 1)

        x = self.final_dropout(x)

        encoder_raw_out = x
        encoder_raw_outputs = {
            "encoder_raw_out": encoder_raw_out,
            "padding_mask": padding_mask,
        }

        # logger.info("Input of cif network's type is {}".format(x.type()))
        # logger.info("Encoder padding {}".format(padding_mask))

        if self.proj_encoder_to_target:
            x = self.proj_encoder_to_target(x)

        # logger.info("encoder out size {}".format(x.shape))
        encoder_outputs = {
            "encoder_out": x,  # T x B x C
            "padding_mask": padding_mask,
        }

        if self.use_ctc_constraint:
            cif_outputs = self.cif(
                encoder_outputs=encoder_raw_outputs,
                target_lengths=target_lengths,
                ctc_logits=encoder_raw_out,
            )
        else:
            cif_outputs = self.cif(encoder_raw_outputs, target_lengths)
        cif_raw_out = cif_outputs["cif_out"].type_as(x)
        cif_padding = cif_outputs["padding_mask"].long()
        loss_pen = cif_outputs["loss_pen"]

        # project to phone dictionary size, for ce loss
        cif_out = cif_raw_out
        if self.proj_to_phone_d:
            cif_out = self.proj_to_phone_d(cif_raw_out)

        # project to char cif embedding dimension
        if self.proj_cif_c2p:
            cif_raw_out = self.proj_cif_c2p(cif_raw_out)

        # logger.info("Cif padding {}".format(cif_padding))

        return {
            "cif_raw_outputs":{
                "cif_raw_out": cif_raw_out,
                "padding_mask": cif_padding,  # B x T
            },
            "cif_outputs":{
                "cif_out": cif_out,
                "padding_mask": cif_padding,
                "loss_pen": loss_pen,
                "ctc_align_out": cif_outputs["ctc_align_outputs"] if self.use_ctc_constraint else None
            },
            "encoder_outputs": encoder_outputs
        }

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        # project back to size of vocabulary
        return F.linear(features, self.embed_out)

class PretrainedTransformerDecoder(FairseqIncrementalDecoder):
    def __init__(self,
                 cfg: Wav2Vec2MultiScaleCifConfig,
                 dictionary,
                 embed_tokens,
                 no_encoder_attn=False
    ):

        arg_overrides = {
            "dropout": cfg.decoder_dropout,
            "activation_dropout": cfg.decoder_activation_dropout,
            "attention_dropout": cfg.decoder_attention_dropout,
            "share_decoder_input_output_embed": cfg.share_decoder_input_output_embed,
        }

        if cfg.decoder_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(cfg.decoder_path, arg_overrides)
            decoder_args = state.get("cfg", None)
            if decoder_args is None:
                decoder_args = convert_namespace_to_omegaconf(state["args"])
            decoder_args.criterion = None
            decoder_args.lr_scheduler = None
            cfg.decoder_args = decoder_args
        else:
            state = None
            decoder_args = cfg.decoder_args
            if isinstance(decoder_args, Namespace):
                cfg.decoder_args = decoder_args = convert_namespace_to_omegaconf(decoder_args)

        cfg.decoder_embed_dim = decoder_args.model.decoder_output_dim

        task = tasks.setup_task(decoder_args.task)
        model = task.build_model(decoder_args.model)

        if state is not None and not cfg.no_pretrained_weights:
            model.load_state_dict(state["model"], strict=True)

        super().__init__(dictionary)

        self.decoder_model = model
        self.output_embed_dim = cfg.decoder_embed_dim
        self.share_input_output_embed = decoder_args.model.share_decoder_input_output_embed

        if not self.share_input_output_embed:
            self.embed_out = nn.Parameter(
                torch.Tensor(len(dictionary), self.output_embed_dim)
            )
            nn.init.normal_(self.embed_out, mean=0, std=self.output_embed_dim ** -0.5)
        self.embed_tokens = embed_tokens


    def forward(
        self, prev_output_tokens, encoder_out=None, incremental_state=None, **kwargs
    ):
        decoder_input = {
            "prev_output_tokens": prev_output_tokens,
            "encoder_out": encoder_out,
            "incremental_state": incremental_state,
        }

        x, extra = self.decoder_model.decoder.extract_features(**decoder_input)

        return x, extra

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        # project back to size of vocabulary
        if self.share_input_output_embed:
            return F.linear(features, self.embed_tokens.weight)
        else:
            return F.linear(features, self.embed_out)


@register_model("wav2vec_mscif", dataclass=Wav2Vec2MultiScaleCifConfig)
class Wav2VecMultiScaleCifModel(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder, char_cif, cfg:Wav2Vec2MultiScaleCifConfig):
        super().__init__(encoder, decoder)

        self.char_cif = char_cif
        self.blank_weight = cfg.blank_weight
        self.blank_mode = cfg.blank_mode

        self.proj_to_decoder_dim = None

        targ_d = cfg.decoder_embed_dim
        char_cif_d = cfg.char_cif_embedding_dim

        if targ_d != char_cif_d:
            self.proj_to_decoder_dim = Linear(char_cif_d, targ_d)

        self.apply_joint_cnn = cfg.apply_joint_cnn
        if cfg.apply_joint_cnn:
            self.char_cif_out_conv = torch.nn.Conv1d(
                cfg.char_cif_embedding_dim,
                cfg.decoder_embed_dim,
                cfg.joint_conv_width,
                stride=1, padding=1,
                dilation=1, groups=1,
                bias=True, padding_mode='zeros'
            ).cuda()
            self.char_conv_dropout = torch.nn.Dropout(cfg.joint_conv_dropout).cuda()
            self.decoder_out_conv = torch.nn.Conv1d(
                cfg.decoder_embed_dim,
                cfg.decoder_embed_dim,
                cfg.joint_conv_width,
                stride=1, padding=1,
                dilation=1, groups=1,
                bias=True, padding_mode='zeros'
            ).cuda()
            self.decoder_conv_dropout = torch.nn.Dropout(cfg.joint_conv_dropout).cuda()

        self.use_ctc_constraint = cfg.use_ctc_constraint

    @classmethod
    def build_model(cls, cfg: Wav2Vec2MultiScaleCifConfig, task: FairseqTask):
        """Build a new model instance."""

        assert (
            cfg.autoregressive
        ), "Please set task.autoregressive=true for seq2seq asr models"

        src_dict = task.source_dictionary
        p_tgt_dict = task.target_phone_dictionary
        c_tgt_dict = task.target_char_dictionary

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            return emb

        decoder_embed_tokens = build_embedding(c_tgt_dict, cfg.decoder_embed_dim)

        phone_cfg_overrides = {
            "cif_embedding_dim": cfg.phone_cif_embedding_dim,
            "produce_weight_type": cfg.phone_produce_weight_type,
            "apply_scaling": cfg.phone_apply_scaling,
            "apply_tail_handling": cfg.phone_apply_tail_handling,
            "conv_cif_layer_num": cfg.phone_conv_cif_layer_num,
            "conv_cif_output_channels_num": cfg.phone_conv_cif_output_channels_num,
            "conv_cif_width": cfg.phone_conv_cif_width,
            "conv_cif_dropout": cfg.phone_conv_cif_dropout,
        }

        char_cfg_overrides = {
            "cif_embedding_dim": cfg.char_cif_embedding_dim,
            "produce_weight_type": cfg.char_produce_weight_type,
            "apply_scaling": cfg.char_apply_scaling,
            "apply_tail_handling": cfg.char_apply_tail_handling,
            "conv_cif_layer_num": cfg.char_conv_cif_layer_num,
            "conv_cif_output_channels_num": cfg.char_conv_cif_output_channels_num,
            "conv_cif_width": cfg.char_conv_cif_width,
            "conv_cif_dropout": cfg.char_conv_cif_dropout,
        }

        phone_cif_cfg = copy.deepcopy(cfg)
        for key, value in phone_cfg_overrides.items():
            setattr(phone_cif_cfg, key, value)

        char_cif_cfg = copy.deepcopy(cfg)
        for key, value in char_cfg_overrides.items():
            setattr(char_cif_cfg, key, value)

        decoder = cls.build_decoder(char_cif_cfg, c_tgt_dict, decoder_embed_tokens)
        encoder = cls.build_encoder(phone_cif_cfg, p_tgt_dict, cfg.char_cif_embedding_dim)
        char_cif = cls.build_char_cif(char_cif_cfg)

        return Wav2VecMultiScaleCifModel(encoder, decoder, char_cif, cfg)

    @classmethod
    def build_char_cif(cls, cfg: Wav2Vec2MultiScaleCifConfig):
        return Cif(cfg)

    @classmethod
    def build_encoder(cls, cfg: Wav2Vec2MultiScaleCifConfig, tgt_dict, output_size=None):
        return MultiscaleCifEncoder(cfg, tgt_dict, output_size)

    @classmethod
    def build_decoder(cls, cfg: Wav2Vec2MultiScaleCifConfig, tgt_dict, embed_tokens):
        if cfg.use_pretrained_decoder:
            return PretrainedTransformerDecoder(cfg, tgt_dict, embed_tokens, no_encoder_attn=True) # encoder attention is not needed.
        else:
            return CifDecoder(cfg, tgt_dict, embed_tokens, no_encoder_attn=True)

    def forward(self, phone_target_lengths=None, char_target_lengths=None, **kwargs):
        assert "phone_prev_output_tokens" in kwargs.keys(), "Phone target is not added to input."
        assert "char_prev_output_tokens" in kwargs.keys(), "Char target is  not added to input."

        ''' Phone scale '''
        p_cif_encoder_out = self.encoder(target_lengths=phone_target_lengths, **kwargs)

        '''
        p_cif_encoder_out {
            "cif_outputs":{
                "cif_out": cif_out,
                "padding_mask": cif_padding,  # B x T
                "loss_pen": loss_pen,
            },
            "encoder_outputs": {
                "encoder_out": x,  # T x B x C
                "padding_mask": padding_mask,
            }
        }
        '''

        p_encoder_outputs = p_cif_encoder_out["encoder_outputs"]  # for ctc loss
        p_cif_outputs = p_cif_encoder_out["cif_outputs"]
        p_cif_raw_outputs = p_cif_encoder_out["cif_raw_outputs"]

        ''' Char scale '''

        char_prev_output_tokens = kwargs["char_prev_output_tokens"]

        c_inputs = { # for char cif input
            "encoder_raw_out": p_cif_raw_outputs["cif_raw_out"],
            "padding_mask": p_cif_raw_outputs["padding_mask"]
        }

        c_cif_outputs = self.char_cif(c_inputs, target_lengths=char_target_lengths, transpose=False)
        c_cif_out = c_cif_outputs["cif_out"]

        decoder_out, _ = self.decoder(char_prev_output_tokens, **kwargs)

        # Truncate
        if c_cif_out.shape[1] != decoder_out.shape[1]:
            # logger.info("Encoder and decoder output dimension don't match. Decoder input is {}".format(decoder_input))

            Tmin = min(c_cif_out.shape[1], decoder_out.shape[1])
            c_cif_out = c_cif_out[:, :Tmin, :]
            decoder_out = decoder_out[:, :Tmin, :]
            c_cif_outputs["padding_mask"] = c_cif_outputs["padding_mask"][:, :Tmin]

        if self.apply_joint_cnn:
            c_cif_out = self.char_cif_out_conv(c_cif_out.permute(0, 2, 1)).permute(0, 2, 1)
            c_cif_out = self.char_conv_dropout(c_cif_out)
            decoder_out = self.decoder_out_conv(decoder_out.permute(0, 2, 1)).permute(0, 2, 1)
            decoder_out = self.decoder_conv_dropout(decoder_out)
        elif self.proj_to_decoder_dim:
            c_cif_out = self.proj_to_decoder_dim(c_cif_out)

        joint_out = torch.tanh(c_cif_out + decoder_out)
        # logger.info(joint_out.type())

        # Project to vocabulary size
        joint_out = self.decoder.output_layer(joint_out)
        # logger.info("Joint output shape {}".format(joint_out.shape))

        return {
            "joint_outputs": {
                "joint_out": joint_out,  # B x Tmin x C
                "padding_mask": c_cif_outputs["padding_mask"],
                "fired_marks": c_cif_outputs["fired_marks"],
                "qua_out": c_cif_outputs["loss_pen"],
            },

            "phone_cif_outputs": {
                "phone_cif_out": p_cif_outputs["cif_out"],
                "padding_mask": p_cif_outputs["padding_mask"],
                "qua_out": p_cif_outputs["loss_pen"],
                "ctc_align_out": p_cif_outputs["ctc_align_out"]
            },

            # for ctc loss
            "phone_encoder_outputs": {
                "encoder_out": p_encoder_outputs["encoder_out"],
                "padding_mask": p_encoder_outputs["padding_mask"],
            },
        }

    def get_logits(self, net_output, normalize=False):
        logits = net_output["joint_out"] if "joint_out" in net_output.keys() else net_output["phone_cif_out"]

        if net_output["padding_mask"] is not None and net_output["padding_mask"].any():
            logits[net_output["padding_mask"]][...] = float("-inf")

        if normalize:
            logits = utils.log_softmax(logits.type_as(net_output["joint_out"]), dim=-1)

        return logits

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a cif model's output."""

        logits = self.get_logits(net_output)

        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    def get_encoder_logits(self, net_output, normalize=False):
        logits = net_output["encoder_out"]
        if self.blank_weight != 0:
            if self.blank_mode == "add":
                logits[..., 0] += self.blank_weight
            elif self.blank_mode == "set":
                logits[..., 0] = self.blank_weight
            else:
                raise Exception(f"invalid blank mode {self.blank_mode}")

        if net_output["padding_mask"] is not None and net_output["padding_mask"].any():
            logits[net_output["padding_mask"].T][..., 0] = float("inf")
            logits[net_output["padding_mask"].T][..., 1:] = float("-inf")

        if normalize:
            logits = utils.log_softmax(logits.type_as(net_output["encoder_out"]), dim=-1)

        return logits

    def get_encoder_normalized_probs(self, logits, log_probs):
        """Get normalized probabilities (or log probs) from a encoder's output."""

        if log_probs:
            return utils.log_softmax(logits, dim=-1)
        else:
            return utils.softmax(logits, dim=-1)

    def get_probs_from_logits(self, logits, log_probs=False):
        """Get normalized probabilities (or log probs) from logits."""
        if log_probs:
            return utils.log_softmax(logits, dim=-1)
        else:
            return utils.softmax(logits, dim=-1)

    def get_cif_output(self, target_lengths=None, **kwargs):
        p_cif_encoder_out = self.encoder(target_lengths=None, **kwargs)

        p_cif_raw_outputs = p_cif_encoder_out["cif_raw_outputs"]
        p_cif_out = p_cif_raw_outputs["cif_raw_out"]

        p_raw_outputs = {  # for char cif input
            "encoder_raw_out": p_cif_out,
            "padding_mask": p_cif_raw_outputs["padding_mask"]
        }

        c_cif_outputs = self.char_cif(p_raw_outputs, target_lengths=target_lengths, transpose=False)
        return c_cif_outputs

    def get_phone_output(self, target_lengths=None, **kwargs):
        p_cif_encoder_out = self.encoder(target_lengths=target_lengths, **kwargs)
        return p_cif_encoder_out["cif_outputs"]

    def step_forward_decoder(self,
                             prev_decoded_tokens,
                             cif_outputs,
                             incremental_state=None, **kwargs):
        decoder_out, _ = self.decoder(prev_output_tokens=prev_decoded_tokens, **kwargs)
        cif_out = cif_outputs["cif_out"]

        # Truncate
        if cif_out.shape[1] != decoder_out.shape[1]:
            # logger.info("Encoder and decoder output dimension don't match. Decoder input is {}".format(decoder_input))

            Tmin = min(cif_out.shape[1], decoder_out.shape[1])
            cif_out = cif_out[:, :Tmin, :]
            decoder_out = decoder_out[:, :Tmin, :]
            cif_outputs["padding_mask"] = cif_outputs["padding_mask"][:, :Tmin]

        if self.apply_joint_cnn:
            cif_out = self.char_cif_out_conv(cif_out.permute(0, 2, 1)).permute(0, 2, 1)
            cif_out = self.char_conv_dropout(cif_out)
            decoder_out = self.decoder_out_conv(decoder_out.permute(0, 2, 1)).permute(0, 2, 1)
            decoder_out = self.decoder_conv_dropout(decoder_out)
        elif self.proj_to_decoder_dim:
            cif_out = self.proj_to_decoder_dim(cif_out)

        joint_out = torch.tanh(cif_out + decoder_out)

        # Project to vocabulary size
        joint_out = self.decoder.output_layer(joint_out)
        return joint_out

    def step_forward_phone_decoder(self,
                                   cif_outputs,
                                   incremental_state=None, **kwargs):
        cif_out = cif_outputs["cif_out"]

        return cif_out


@register_model("lexicon_constraint_ml_cif", dataclass=Wav2Vec2MultiScaleCifConfig)
class LexiconConstraintMultiLevelCifModel(Wav2VecMultiScaleCifModel):
    def __init__(self, encoder, decoder, char_cif, cfg:Wav2Vec2MultiScaleCifConfig):
        super().__init__(encoder, decoder, char_cif, cfg)

    @classmethod
    def build_model(cls, cfg: Wav2Vec2MultiScaleCifConfig, task: FairseqTask):
        """Build a new model instance."""

        assert (
            cfg.autoregressive
        ), "Please set task.autoregressive=true for seq2seq asr models"

        src_dict = task.source_dictionary
        p_tgt_dict = task.target_phone_dictionary
        c_tgt_dict = task.target_char_dictionary

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            return emb

        decoder_embed_tokens = build_embedding(c_tgt_dict, cfg.decoder_embed_dim)

        phone_cfg_overrides = {
            "cif_embedding_dim": cfg.phone_cif_embedding_dim,
            "produce_weight_type": cfg.phone_produce_weight_type,
            "apply_scaling": cfg.phone_apply_scaling,
            "apply_tail_handling": cfg.phone_apply_tail_handling,
            "conv_cif_layer_num": cfg.phone_conv_cif_layer_num,
            "conv_cif_output_channels_num": cfg.phone_conv_cif_output_channels_num,
            "conv_cif_width": cfg.phone_conv_cif_width,
            "conv_cif_dropout": cfg.phone_conv_cif_dropout,
        }

        char_cfg_overrides = {
            "cif_embedding_dim": cfg.char_cif_embedding_dim,
            "produce_weight_type": cfg.char_produce_weight_type,
            "apply_scaling": cfg.char_apply_scaling,
            "apply_tail_handling": cfg.char_apply_tail_handling,
            "conv_cif_layer_num": cfg.char_conv_cif_layer_num,
            "conv_cif_output_channels_num": cfg.char_conv_cif_output_channels_num,
            "conv_cif_width": cfg.char_conv_cif_width,
            "conv_cif_dropout": cfg.char_conv_cif_dropout,
        }

        phone_cif_cfg = copy.deepcopy(cfg)
        for key, value in phone_cfg_overrides.items():
            setattr(phone_cif_cfg, key, value)

        char_cif_cfg = copy.deepcopy(cfg)
        for key, value in char_cfg_overrides.items():
            setattr(char_cif_cfg, key, value)

        decoder = cls.build_decoder(char_cif_cfg, c_tgt_dict, decoder_embed_tokens)
        encoder = cls.build_encoder(phone_cif_cfg, p_tgt_dict, cfg.char_cif_embedding_dim)
        char_cif = cls.build_char_cif(char_cif_cfg)

        return LexiconConstraintMultiLevelCifModel(encoder, decoder, char_cif, cfg)

    @classmethod
    def build_char_cif(cls, cfg: Wav2Vec2MultiScaleCifConfig):
        return LexiconConstraintCif(cfg)

    @classmethod
    def build_encoder(cls, cfg: Wav2Vec2MultiScaleCifConfig, tgt_dict, output_size=None):
        return MultiscaleCifEncoder(cfg, tgt_dict, output_size)

    @classmethod
    def build_decoder(cls, cfg: Wav2Vec2MultiScaleCifConfig, tgt_dict, embed_tokens):
        if cfg.use_pretrained_decoder:
            return PretrainedTransformerDecoder(cfg, tgt_dict, embed_tokens,
                                                no_encoder_attn=True)  # encoder attention is not needed.
        else:
            return CifDecoder(cfg, tgt_dict, embed_tokens, no_encoder_attn=True)

    def forward(self, phone_target_lengths=None, char_target_lengths=None, phone_segs=None, integrate_weights=None, **kwargs):
        assert "phone_prev_output_tokens" in kwargs.keys(), "Phone target is not added to input."
        assert "char_prev_output_tokens" in kwargs.keys(), "Char target is  not added to input."

        ''' Phone scale '''
        p_cif_encoder_out = self.encoder(target_lengths=phone_target_lengths, **kwargs)

        '''
        p_cif_encoder_out {
            "cif_outputs":{
                "cif_out": cif_out,
                "padding_mask": cif_padding,  # B x T
                "loss_pen": loss_pen,
            },
            "encoder_outputs": {
                "encoder_out": x,  # T x B x C
                "padding_mask": padding_mask,
            }
        }
        '''

        p_encoder_outputs = p_cif_encoder_out["encoder_outputs"]  # for ctc loss
        p_cif_outputs = p_cif_encoder_out["cif_outputs"]
        p_cif_raw_outputs = p_cif_encoder_out["cif_raw_outputs"]

        ''' Char scale '''

        char_prev_output_tokens = kwargs["char_prev_output_tokens"]

        c_inputs = { # for char cif input
            "encoder_raw_out": p_cif_raw_outputs["cif_raw_out"],
            "padding_mask": p_cif_raw_outputs["padding_mask"]
        }

        c_cif_outputs = self.char_cif(c_inputs, target_lengths=char_target_lengths, transpose=False, phone_segs=phone_segs, integrate_weights=integrate_weights)
        c_cif_out = c_cif_outputs["cif_out"]

        decoder_out, _ = self.decoder(char_prev_output_tokens, **kwargs)

        # Truncate
        if c_cif_out.shape[1] != decoder_out.shape[1]:
            # logger.info("Encoder and decoder output dimension don't match. Decoder input is {}".format(decoder_input))

            Tmin = min(c_cif_out.shape[1], decoder_out.shape[1])
            c_cif_out = c_cif_out[:, :Tmin, :]
            decoder_out = decoder_out[:, :Tmin, :]
            c_cif_outputs["padding_mask"] = c_cif_outputs["padding_mask"][:, :Tmin]

        if self.apply_joint_cnn:
            c_cif_out = self.char_cif_out_conv(c_cif_out.permute(0, 2, 1)).permute(0, 2, 1)
            c_cif_out = self.char_conv_dropout(c_cif_out)
            decoder_out = self.decoder_out_conv(decoder_out.permute(0, 2, 1)).permute(0, 2, 1)
            decoder_out = self.decoder_conv_dropout(decoder_out)
        elif self.proj_to_decoder_dim:
            c_cif_out = self.proj_to_decoder_dim(c_cif_out)

        joint_out = torch.tanh(c_cif_out + decoder_out)
        # logger.info(joint_out.type())

        # Project to vocabulary size
        joint_out = self.decoder.output_layer(joint_out)
        # logger.info("Joint output shape {}".format(joint_out.shape))

        return {
            "joint_outputs": {
                "joint_out": joint_out,  # B x Tmin x C
                "padding_mask": c_cif_outputs["padding_mask"],
                "fired_marks": c_cif_outputs["fired_marks"],
                "qua_out": c_cif_outputs["loss_pen"],
            },

            "phone_cif_outputs": {
                "phone_cif_out": p_cif_outputs["cif_out"],
                "padding_mask": p_cif_outputs["padding_mask"],
                "qua_out": p_cif_outputs["loss_pen"],
                "ctc_align_out": p_cif_outputs["ctc_align_out"]
            },

            # for ctc loss
            "phone_encoder_outputs": {
                "encoder_out": p_encoder_outputs["encoder_out"],
                "padding_mask": p_encoder_outputs["padding_mask"],
            },
        }

    def get_logits(self, net_output, normalize=False):
        logits = net_output["joint_out"] if "joint_out" in net_output.keys() else net_output["phone_cif_out"]

        if net_output["padding_mask"] is not None and net_output["padding_mask"].any():
            logits[net_output["padding_mask"]][...] = float("-inf")

        if normalize:
            logits = utils.log_softmax(logits.type_as(net_output["joint_out"]), dim=-1)

        return logits

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a cif model's output."""

        logits = self.get_logits(net_output)

        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    def get_encoder_logits(self, net_output, normalize=False):
        logits = net_output["encoder_out"]
        if self.blank_weight != 0:
            if self.blank_mode == "add":
                logits[..., 0] += self.blank_weight
            elif self.blank_mode == "set":
                logits[..., 0] = self.blank_weight
            else:
                raise Exception(f"invalid blank mode {self.blank_mode}")

        if net_output["padding_mask"] is not None and net_output["padding_mask"].any():
            logits[net_output["padding_mask"].T][..., 0] = float("inf")
            logits[net_output["padding_mask"].T][..., 1:] = float("-inf")

        if normalize:
            logits = utils.log_softmax(logits.type_as(net_output["encoder_out"]), dim=-1)

        return logits

    def get_encoder_normalized_probs(self, logits, log_probs):
        """Get normalized probabilities (or log probs) from a encoder's output."""

        if log_probs:
            return utils.log_softmax(logits, dim=-1)
        else:
            return utils.softmax(logits, dim=-1)

    def get_probs_from_logits(self, logits, log_probs=False):
        """Get normalized probabilities (or log probs) from logits."""
        if log_probs:
            return utils.log_softmax(logits, dim=-1)
        else:
            return utils.softmax(logits, dim=-1)

    def get_cif_output(self, target_lengths=None, **kwargs):
        p_cif_encoder_out = self.encoder(target_lengths=None, **kwargs)

        p_cif_raw_outputs = p_cif_encoder_out["cif_raw_outputs"]
        p_cif_out = p_cif_raw_outputs["cif_raw_out"]

        p_raw_outputs = {  # for char cif input
            "encoder_raw_out": p_cif_out,
            "padding_mask": p_cif_raw_outputs["padding_mask"]
        }

        c_cif_outputs = self.char_cif(p_raw_outputs, target_lengths=target_lengths, transpose=False)
        return c_cif_outputs

    def get_phone_output(self, target_lengths=None, **kwargs):
        p_cif_encoder_out = self.encoder(target_lengths=target_lengths, **kwargs)
        return p_cif_encoder_out["cif_outputs"]

    def step_forward_decoder(self,
                             prev_decoded_tokens,
                             cif_outputs,
                             incremental_state=None, **kwargs):
        decoder_out, _ = self.decoder(prev_output_tokens=prev_decoded_tokens, **kwargs)
        cif_out = cif_outputs["cif_out"]

        # Truncate
        if cif_out.shape[1] != decoder_out.shape[1]:
            # logger.info("Encoder and decoder output dimension don't match. Decoder input is {}".format(decoder_input))

            Tmin = min(cif_out.shape[1], decoder_out.shape[1])
            cif_out = cif_out[:, :Tmin, :]
            decoder_out = decoder_out[:, :Tmin, :]
            cif_outputs["padding_mask"] = cif_outputs["padding_mask"][:, :Tmin]

        if self.apply_joint_cnn:
            cif_out = self.char_cif_out_conv(cif_out.permute(0, 2, 1)).permute(0, 2, 1)
            cif_out = self.char_conv_dropout(cif_out)
            decoder_out = self.decoder_out_conv(decoder_out.permute(0, 2, 1)).permute(0, 2, 1)
            decoder_out = self.decoder_conv_dropout(decoder_out)
        elif self.proj_to_decoder_dim:
            cif_out = self.proj_to_decoder_dim(cif_out)

        joint_out = torch.tanh(cif_out + decoder_out)

        # Project to vocabulary size
        joint_out = self.decoder.output_layer(joint_out)
        return joint_out

    def step_forward_phone_decoder(self,
                                   cif_outputs,
                                   incremental_state=None, **kwargs):
        cif_out = cif_outputs["cif_out"]

        return cif_out