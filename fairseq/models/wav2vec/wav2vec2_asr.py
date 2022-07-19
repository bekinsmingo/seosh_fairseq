# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import copy
import logging
import math
import re
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import II, MISSING, open_dict

from fairseq import checkpoint_utils, tasks, utils
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import (
    BaseFairseqModel,
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
)
from fairseq.models.wav2vec.wav2vec2 import MASKING_DISTRIBUTION_CHOICES
from fairseq.modules import LayerNorm, PositionalEmbedding, TransformerDecoderLayer
from fairseq.tasks import FairseqTask

from typing import Dict, List, Optional, Tuple
from torch import Tensor

from pdb import set_trace as Tra

logger = logging.getLogger(__name__)


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
    require_same_masks: bool = field(
        default=True,
        metadata={
            "help": "whether to number of masked timesteps must be the same across all "
            "examples in a batch"
        },
    )
    mask_dropout: float = field(
        default=0.0,
        metadata={"help": "percent of masks to unmask for each sample"},
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
    offload_activations: bool = field(
        default=False, metadata={"help": "offload_activations"}
    )
    min_params_to_wrap: int = field(
        default=int(1e8),
        metadata={
            "help": "minimum number of params for a layer to be wrapped with FSDP() when "
            "training with --ddp-backend=fully_sharded. Smaller values will "
            "improve memory efficiency, but may make torch.distributed "
            "communication less efficient due to smaller input sizes. This option "
            "is set to 0 (i.e., always wrap) when --checkpoint-activations or "
            "--offload-activations are passed."
        },
    )

    checkpoint_activations: bool = field(
        default=False,
        metadata={"help": "recompute activations and save memory for extra compute"},
    )
    ddp_backend: str = II("distributed_training.ddp_backend")





#############################################################
########################     CTC     ########################
#############################################################

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

        ## 왜 xlsr-ctc 하는데 여기서 터지니
        '''
        (Pdb) len(task.target_dictionary)
        32

        (Pdb) task
        <fairseq.tasks.audio_finetuning.AudioFinetuningTask object at 0x7fcb576fa9d0>

        (Pdb) cfg
        {'_name': 'wav2vec_ctc', 'w2v_path': '/workspace/s2st/xlsr/xlsr2_960m_1000k.pt', 'no_pretrained_weights': False, 
        'dropout_input': 0.0, 'final_dropout': 0.0, 'dropout': 0.0, 'attention_dropout': 0.0, 'activation_dropout': 0.1, 
        'conv_feature_layers': '[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]', 'encoder_embed_dim': 768, 
        'apply_mask': True, 'mask_length': 10, 'mask_prob': 0.75, 'mask_selection': static, 'mask_other': 0.0, 'no_mask_overlap': False, 
        'mask_min_space': 1, 'require_same_masks': True, 'mask_dropout': 0.0, 'mask_channel_length': 64, 'mask_channel_prob': 0.25, 
        'mask_channel_selection': static, 'mask_channel_other': 0.0, 'no_mask_channel_overlap': False, 'freeze_finetune_updates': 10000, 
        'feature_grad_mult': 0.0, 'layerdrop': 0.1, 'mask_channel_min_space': 1, 'mask_channel_before': False, 'normalize': True, 
        'data': '/workspace/librispeech_model/am/fairseq_audio_data2', 'w2v_args': None, 'checkpoint_activations': False, 
        'offload_activations': False, 'min_params_to_wrap': 100000000, 'ddp_backend': 'legacy_ddp', 'blank_weight': 0.0, 'blank_mode': 'add'}
        '''
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
            number_of_classes = logits.size(-1)
            masking_tensor = torch.ones(
                number_of_classes, device=logits.device
            ) * float("-inf")
            masking_tensor[0] = 0
            logits[net_output["padding_mask"].T] = masking_tensor.type_as(logits)

        if normalize:
            logits = utils.log_softmax(logits.float(), dim=-1)

        return logits

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = self.get_logits(net_output)

        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    def forward(self, inter_ctc = False, **kwargs):
        out = self.w2v_encoder(**kwargs)
        # dict_keys(['encoder_out', 'padding_mask', 'layer_results'])
        if inter_ctc:
            num_layer = len(out["layer_results"])
            inter_encoder_out = out["layer_results"][num_layer//2][0]

            if self.w2v_encoder.w2v_model.encoder.layer_norm_first:
                inter_encoder_out = inter_encoder_out.transpose(0,1)
                inter_encoder_out = self.w2v_encoder.w2v_model.encoder.layer_norm(inter_encoder_out)
                inter_encoder_out = inter_encoder_out.transpose(0,1)
            inter_encoder_out = self.w2v_encoder.final_dropout(inter_encoder_out)
            inter_logits = self.w2v_encoder.proj(inter_encoder_out)
            inter_out = {
                'encoder_out' : inter_logits,
                'padding_mask' : out['padding_mask'],
                'layer_results' : out['layer_results'],
            }
            # Tra()
            return out, inter_out
        else:
            return out


#################################################################
########################     Seq2seq     ########################
#################################################################

# @dataclass
# class Wav2Vec2CtcForS2TConfig(Wav2Vec2CtcConfig):
#     s2t_src_joint_ctc: bool = II("task.s2t_src_joint_ctc")

@register_model("wav2vec_ctc_for_s2t", dataclass=Wav2Vec2CtcConfig)
class Wav2VecCtcForS2T(Wav2VecCtc):
    def __init__(self, cfg: Wav2Vec2CtcConfig, w2v_encoder: BaseFairseqModel):
        super().__init__(cfg, w2v_encoder)

    @classmethod
    def build_model(cls, cfg: Wav2Vec2CtcConfig, task: FairseqTask):
        # Tra()
        # w2v_encoder = Wav2VecEncoder(cfg, len(task.target_dictionary), len(task.target_dictionary))
        if task.cfg.s2t_src_joint_ctc:
            w2v_encoder = Wav2VecEncoder(cfg, len(task.source_dictionary), len(task.source_dictionary))
        else:
            w2v_encoder = Wav2VecEncoder(cfg, len(task.target_dictionary), len(task.target_dictionary))

        return cls(cfg, w2v_encoder)


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


    ########################### added ###########################

    ## for bart initialized decoder
    load_pretrained_w2v_ctc_from: Optional[str] = field(
        default=None, metadata={"help": "model to take decoder weights from (for initialization)"}
    )

    load_pretrained_decoder_from: Optional[str] = field(
        default=None, metadata={"help": "model to take decoder weights from (for initialization)"}
    )

    ## for ctc joint
    # ctc_weight: float = II("criterion.ctc_weight")
    # ctc_weight: float = field(default=1.0, metadata={"help": "weight for CTC loss"})
    s2t_src_joint_ctc: bool = II("task.s2t_src_joint_ctc")

    ## for soft input training
    soft_input_training: bool = field(
        default=False, metadata={"help": "tmp"}
    )
    soft_input_training_updates: int = field(
        default=50000, metadata={"help": "tmp"}
    )

    # ## for rnn decoder
    # use_srupp_decoder: bool = field(
    #     default=False, metadata={"help": "tmp"}
    # ) 

def need_finetuning(ft_params, param_name):
    if ft_params == "all":
        return True
    ft_params_list = ft_params.split(",")
    for ft_param in ft_params_list:
        if ft_param in param_name:
            return True
    return False


@register_model("wav2vec_seq2seq", dataclass=Wav2Vec2Seq2SeqConfig)
class Wav2Vec2Seq2SeqModel(FairseqEncoderDecoderModel):
    def __init__(self, cfg:Wav2Vec2Seq2SeqConfig, encoder, decoder):
        super().__init__(encoder, decoder)
        self.cfg = cfg
        self.soft_input_training = cfg.soft_input_training
        self.soft_input_training_updates = cfg.soft_input_training_updates

    @classmethod
    def build_model(cls, cfg: Wav2Vec2Seq2SeqConfig, task: FairseqTask):
        """Build a new model instance."""

        assert (
            cfg.autoregressive
        ), "Please set task.autoregressive=true for seq2seq asr models"

        # Tra()

        src_dict = task.source_dictionary if task.cfg.s2t_src_joint_ctc else None
        tgt_dict = task.target_dictionary

        '''
        (Pdb) len(src_dict)
        97
        (Pdb) len(tgt_dict)
        2562
        '''

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            return emb

        decoder_embed_tokens = build_embedding(tgt_dict, cfg.decoder_embed_dim)

        encoder = cls.build_encoder(cfg, src_dict if task.cfg.s2t_src_joint_ctc and src_dict else tgt_dict, task.cfg)
        decoder = cls.build_decoder(cfg, tgt_dict, decoder_embed_tokens)

        return Wav2Vec2Seq2SeqModel(cfg, encoder, decoder)

    @classmethod
    def build_encoder(cls, cfg: Wav2Vec2AsrConfig, tgt_dict, task_cfg=None):

        # load_pretrained_w2v_ctc_from

        if cfg.load_pretrained_w2v_ctc_from:
            logger.info("| loading pretrained w2v2-ctc model from {}".format(cfg.load_pretrained_w2v_ctc_from))
            import os
            import copy
            
            model_dir_path, checkpoint = os.path.split(cfg.load_pretrained_w2v_ctc_from)

            w2v_arg_overrides = {
                    "encoder_layerdrop": cfg.layerdrop,
                    "dropout": cfg.dropout,
                    "activation_dropout": cfg.activation_dropout,
                    "dropout_input": cfg.dropout_input,
                    "attention_dropout": cfg.attention_dropout,
                    "mask_length": cfg.mask_length,
                    "mask_prob": cfg.mask_prob,
                    "require_same_masks": getattr(cfg, "require_same_masks", True),
                    "pct_holes": getattr(cfg, "mask_dropout", 0),
                    "mask_selection": cfg.mask_selection,
                    "mask_other": cfg.mask_other,
                    "no_mask_overlap": cfg.no_mask_overlap,
                    "mask_channel_length": cfg.mask_channel_length,
                    "mask_channel_prob": cfg.mask_channel_prob,
                    "mask_channel_before": cfg.mask_channel_before,
                    "mask_channel_selection": cfg.mask_channel_selection,
                    "mask_channel_other": cfg.mask_channel_other,
                    "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
                    "feature_grad_mult": cfg.feature_grad_mult,
                    "checkpoint_activations": cfg.checkpoint_activations,
                    "offload_activations": cfg.offload_activations,
                    "min_params_to_wrap": cfg.min_params_to_wrap,
            }

            # Tra()
            arg_overrides = {
                "task": {
                    "_name":'audio_finetuning', 
                    # "data": model_dir_path,
                    "data": task_cfg.data,
                    "s2t_src_joint_ctc": cfg.s2t_src_joint_ctc,
                    "s2t_src_data": task_cfg.s2t_src_data,
                    },
                "model": {
                    "_name": 'wav2vec_ctc_for_s2t',
                    "freeze_finetune_updates": cfg.freeze_finetune_updates,
                    "w2v_args" : {
                        "model": w2v_arg_overrides,
                        "task": {
                            "data": task_cfg.data if not cfg.s2t_src_joint_ctc else task_cfg.s2t_src_data
                            }
                        }
                    }
            }

            w2v_ctc_models, w2v_ctc_cfg, w2v_ctc_task = checkpoint_utils.load_model_ensemble_and_task(
                utils.split_paths(cfg.load_pretrained_w2v_ctc_from, separator="\\"),
                arg_overrides=arg_overrides,
                strict=False,
            )

            w2v_ctc = w2v_ctc_models[0].w2v_encoder
            encoder_out_dim = w2v_ctc.proj.weight.size(1)
            # Tra()

            # for joint ctc
            w2v_ctc._modules.pop('ctc_proj')
            og_ctc_proj = w2v_ctc._modules.pop('proj')

            if cfg.s2t_src_joint_ctc:
                w2v_ctc._modules['ctc_proj'] = og_ctc_proj

            # for encoder-decoder matching
            if cfg.decoder_embed_dim != encoder_out_dim:
                proj = Linear(encoder_out_dim, cfg.decoder_embed_dim)
                w2v_ctc._modules['proj'] = proj
            # Tra()

            '''
            (Pdb) w2v_ctc.w2v_model.encoder.layerdrop
            0.0
            (Pdb) w2v_ctc.freeze_finetune_updates
            6000
            '''

            return w2v_ctc
        else:
            return Wav2VecEncoder(cfg, ctc_proj_dim = len(tgt_dict))

    @classmethod
    def build_decoder(cls, cfg: Wav2Vec2Seq2SeqConfig, tgt_dict, embed_tokens):

        # if cfg.use_srupp_decoder:
        #     decoder = SRUppDecoder(cfg, tgt_dict, embed_tokens)
        #     decoder.reset_parameters()
        #     return decoder
        # else:

        if cfg.load_pretrained_decoder_from:
            _cfg = copy.deepcopy(cfg)
            # if cfg.adaptor_proj or cfg.encoder_embed_dim:  # not V0 arch
            _cfg.encoder_embed_dim = cfg.decoder_embed_dim
            _cfg.dropout = cfg.decoder_dropout
            _cfg.attention_dropout = cfg.decoder_attention_dropout
            _cfg.activation_dropout = cfg.decoder_activation_dropout

            # from fairseq.models.transformer import TransformerDecoder
            decoder = TransformerDecoder(_cfg, tgt_dict, embed_tokens)

            try:
                decoder = checkpoint_utils.load_pretrained_component_from_model(component=decoder, checkpoint=cfg.load_pretrained_decoder_from)
            except RuntimeError as e:
                '''
                *** RuntimeError: Error(s) in loading state_dict for TransformerDecoder:
                        Missing key(s) in state_dict: "embed_out", "embed_positions._float_tensor". 
                        Unexpected key(s) in state_dict: "version", "layernorm_embedding.weight", "layernorm_embedding.bias", "embed_positions.weight". 
                        size mismatch for embed_tokens.weight: copying a param with shape torch.Size([51201, 768]) from checkpoint, the shape in current model is torch.Size([10001, 768]).
                '''
                logger.warning(e)
                # decoder = checkpoint_utils.load_pretrained_component_from_model(component=decoder, checkpoint=cfg.load_pretrained_decoder_from, strict=False)

                from collections import OrderedDict
                state = checkpoint_utils.load_checkpoint_to_cpu(cfg.load_pretrained_decoder_from)
                component_type = "decoder"
                component_state_dict = OrderedDict()
                for key in state["model"].keys():
                    if key.startswith(component_type):
                        # encoder.input_layers.0.0.weight --> input_layers.0.0.weight
                        component_subkey = key[len(component_type) + 1 :]
                        component_state_dict[component_subkey] = state["model"][key]

                component_state_dict['embed_tokens.weight'] = decoder.embed_tokens.weight

                decoder.load_state_dict(component_state_dict, strict=False)

            for k, p in decoder.named_parameters():
                # p.requires_grad = need_finetuning(cfg.finetune_decoder_params, k)
                p.requires_grad = True
            return decoder
        else:
            # Tra()
            return TransformerDecoder(cfg, tgt_dict, embed_tokens)

    def forward(self, **kwargs):
        encoder_out = self.encoder(**kwargs)
        # Tra()
        if (self.soft_input_training) and (self.soft_input_training_updates <= self.encoder.num_updates):
            with torch.no_grad():
                decoder_out = self.decoder(encoder_out=encoder_out, **kwargs)
            # Tra()
            decoder_out = self.decoder(encoder_out=encoder_out, soft_input=decoder_out[0], **kwargs)
        else:
            decoder_out = self.decoder(encoder_out=encoder_out, **kwargs)

        return (decoder_out[0], decoder_out[1], encoder_out)

    def get_ctc_target(self, sample: Optional[Dict[str, Tensor]], s2t_src_joint_ctc=False):
        if s2t_src_joint_ctc:
            return sample["s2t_src_target"], sample["s2t_src_target_lengths"]
        else:
            return sample["target"], sample["target_lengths"]

    def get_ctc_output(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        sample: Optional[Dict[str, Tensor]],
        inter_ctc=False,
        only_inter_ctc=False,
    ):
        # This is for speech translation
        if only_inter_ctc:
            num_layer = len(net_output[2]["layer_results"])
            encoder_out = net_output[2]["layer_results"][num_layer//2][0]
            if self.encoder.w2v_model.encoder.layer_norm_first:
                encoder_out = encoder_out.transpose(0,1)
                encoder_out = self.encoder.w2v_model.encoder.layer_norm(encoder_out)
                encoder_out = encoder_out.transpose(0,1)

            encoder_out = self.encoder.final_dropout(encoder_out)
            # if 'proj.weight' in self.encoder.state_dict().keys():
            #     encoder_out = self.encoder.proj(encoder_out)

            logits = self.encoder.ctc_proj(encoder_out)  # T x B x C
            out = utils.log_softmax(logits.float(), dim=-1)      

            padding_mask = net_output[2]["padding_mask"]
            lens = out.new_full((out.shape[1],), out.shape[0]).long()

            if padding_mask is not None :
                if len(padding_mask) > 0:
                    lens -= padding_mask[0].sum(dim=-1)
                    
            # return out, None, lens
            return None, out, lens

        else:
            # encoder_out = net_output[1]["encoder_out"]["encoder_out"][0]
            encoder_out = net_output[2]["encoder_out_before_proj"]
            logits = self.encoder.ctc_proj(encoder_out)  # T x B x C
            out = utils.log_softmax(logits.float(), dim=-1)

            if inter_ctc:
                num_layer = len(net_output[2]["layer_results"])
                # Tra()
                inter_encoder_out = net_output[2]["layer_results"][num_layer//2][0]
                if self.encoder.w2v_model.encoder.layer_norm_first:
                    inter_encoder_out = inter_encoder_out.transpose(0,1)
                    inter_encoder_out = self.encoder.w2v_model.encoder.layer_norm(inter_encoder_out)
                    inter_encoder_out = inter_encoder_out.transpose(0,1)

                inter_encoder_out = self.encoder.final_dropout(inter_encoder_out)
                # if 'proj.weight' in self.encoder.state_dict().keys():
                #     inter_encoder_out = self.encoder.proj(inter_encoder_out)

                inter_logits = self.encoder.ctc_proj(inter_encoder_out)
                inter_out = utils.log_softmax(inter_logits.float(), dim=-1)

            # padding_mask = net_output[1]["encoder_out"]["encoder_padding_mask"]
            padding_mask = net_output[2]["padding_mask"]
            lens = out.new_full((out.shape[1],), out.shape[0]).long()

            if padding_mask is not None :
                if len(padding_mask) > 0:
                    lens -= padding_mask[0].sum(dim=-1)

            if inter_ctc :
                return out, inter_out, lens 
            else :
                return out, None, lens
        
    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict


####################################################################
########################     Transducer     ########################
####################################################################


@dataclass
class Wav2Vec2TransducerConfig(Wav2Vec2AsrConfig):
    blank_weight: float = 0 # tmp

@register_model("wav2vec_transducer", dataclass=Wav2Vec2TransducerConfig)
class Wav2Vec2Transducer(BaseFairseqModel):
    def __init__(self, cfg: Wav2Vec2CtcConfig, task, w2v_encoder: BaseFairseqModel, rnnt_model):
        super().__init__()
        self.cfg = cfg
        self.task = task
        self.w2v_encoder = w2v_encoder
        self.rnnt_model = rnnt_model
        self.blank_idx = 0

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: Wav2Vec2CtcConfig, task: FairseqTask):
        w2v_encoder = Wav2VecEncoder(cfg)

        transcriber = _Transcriber(
            input_dim=80, # Emformer usually use mel spectrogram as input
            output_dim=cfg.encoder_embed_dim,
            segment_length=16, #
            right_context_length=4, #
            time_reduction_input_dim=128, #
            time_reduction_stride=4, # 
            transformer_num_heads=8,
            transformer_ffn_dim=2048,
            transformer_num_layers=20,
            transformer_dropout=0.1,
            transformer_activation="gelu",
            transformer_left_context_length=30,
            transformer_max_memory_size=0,
            transformer_weight_init_scale_strategy="depthwise",
            transformer_tanh_on_mem=True,
            pretrained_encoder=w2v_encoder, # if it exists, dont care above factors
        )

        predictor = _Predictor(
            len(task.target_dictionary),
            cfg.w2v_args.model.encoder_embed_dim,
            symbol_embedding_dim=cfg.encoder_embed_dim,
            num_lstm_layers=2,
            lstm_layer_norm=True,
            lstm_layer_norm_epsilon=1e-3,
            lstm_dropout=0.1,
        )

        joiner = _Joiner(
            cfg.w2v_args.model.encoder_embed_dim, 
            len(task.target_dictionary)
            )
        
        rnnt_model = RNNT(transcriber, predictor, joiner)

        return cls(cfg, task, w2v_encoder, rnnt_model)

    def forward(self, sample):

        # Transcriber
        encoder_out = self.rnnt_model.transcriber.transformer(**sample['net_input'])
        
        source_encodings = encoder_out['encoder_out'].transpose(0, 1)
        if encoder_out['padding_mask'] is None:
            source_lengths = torch.LongTensor([encoder_out['encoder_out'].size(0)]*encoder_out['encoder_out'].size(1))
        else:
            source_lengths = (~encoder_out['padding_mask']).sum(-1)

        targets = sample['target']
        target_lengths = sample['target_lengths']

        prepended_targets = targets.new_empty([targets.size(0), targets.size(1) + 1])
        prepended_targets[:, 1:] = targets
        prepended_targets[:, 0] = self.blank_idx

        targets = prepended_targets
        target_lengths = target_lengths + 1 

        # Predictior
        target_encodings, target_lengths, predictor_state = self.rnnt_model.predictor(
            input=targets,
            lengths=target_lengths,
        )

        # Joiner
        output, source_lengths, target_lengths = self.rnnt_model.joiner(
            source_encodings=source_encodings,
            source_lengths=source_lengths,
            target_encodings=target_encodings,
            target_lengths=target_lengths,
        )
        
        return source_encodings.cuda(), output.cuda(), source_lengths.cuda()


####################################################################
#########################     Modules     ##########################
####################################################################


class Wav2VecEncoder(FairseqEncoder):
    def __init__(self, cfg: Wav2Vec2AsrConfig, output_size=None, ctc_proj_dim=None):
        self.apply_mask = cfg.apply_mask

        self.cfg = cfg

        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "require_same_masks": getattr(cfg, "require_same_masks", True),
            "pct_holes": getattr(cfg, "mask_dropout", 0),
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
            "checkpoint_activations": cfg.checkpoint_activations,
            "offload_activations": cfg.offload_activations,
            "min_params_to_wrap": cfg.min_params_to_wrap,
        }

        # Why w2v2 seq2seq didnt save w2v_args? -> latest version has it

        if cfg.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(cfg.w2v_path, arg_overrides) # /mnt/hdd~~
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            w2v_args.criterion = None
            w2v_args.lr_scheduler = None
            cfg.w2v_args = w2v_args

            logger.info(w2v_args)

        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(w2v_args)

        model_normalized = w2v_args.task.get(
            "normalize", w2v_args.model.get("normalize", False)
        )
        assert cfg.normalize == model_normalized, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for both pre-training and here"
        )

        if hasattr(cfg, "checkpoint_activations") and cfg.checkpoint_activations:
            with open_dict(w2v_args):
                w2v_args.model.checkpoint_activations = cfg.checkpoint_activations

        # w2v_args.task.data = cfg.data
        # if 'eval_wer' in w2v_args.task.keys():
        #     w2v_args.task._name = 'audio_finetuning'

        task = tasks.setup_task(w2v_args.task)
        model = task.build_model(w2v_args.model, from_checkpoint=True)
        # Tra()

        model.remove_pretraining_modules()

        if state is not None and not cfg.no_pretrained_weights:
            self.load_model_weights(state, model, cfg)

        super().__init__(task.source_dictionary)

        '''
        (Pdb) task.cfg
        {'_name': 'audio_pretraining', 'data': '~', 
        'labels': None, 'binarized_dataset': False, 'sample_rate': 16000, 'normalize': False, 'enable_padding': False, 'max_sample_size': 320000, 
        'min_sample_size': 32000, 'num_batch_buckets': 0, 'precompute_mask_indices': False, 'inferred_w2v_config': None, 'tpu': False, 'text_compression_level': none}
        (Pdb) task.source_dictionary
        (Pdb) task.target_dictionary
        '''

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

        # Tra()

        self.ctc_proj = None
        if ctc_proj_dim is not None:
            self.ctc_proj = nn.Linear(d, ctc_proj_dim)
            # self.ctc_proj = nn.Linear(cfg.encoder_embed_dim, ctc_proj_dim)

        # self.ctc_proj = None
        # if cfg.ctc_weight > 0.0:
        #     self.ctc_proj = nn.Linear(cfg.encoder_embed_dim, ctc_proj_dim)

    def load_model_weights(self, state, model, cfg):
        if cfg.ddp_backend == "fully_sharded":
            from fairseq.distributed import FullyShardedDataParallel

            for name, module in model.named_modules():
                if "encoder.layers" in name and len(name.split(".")) == 3:
                    # Only for layers, we do a special handling and load the weights one by one
                    # We dont load all weights together as that wont be memory efficient and may
                    # cause oom
                    new_dict = {
                        k.replace(name + ".", ""): v
                        for (k, v) in state["model"].items()
                        if name + "." in k
                    }
                    assert isinstance(module, FullyShardedDataParallel)
                    with module.summon_full_params():
                        module.load_state_dict(new_dict, strict=True)
                    module._reset_lazy_init()

            # Once layers are loaded, filter them out and load everything else.
            r = re.compile("encoder.layers.\d.")
            filtered_list = list(filter(r.match, state["model"].keys()))

            new_big_dict = {
                k: v for (k, v) in state["model"].items() if k not in filtered_list
            }

            model.load_state_dict(new_big_dict, strict=False)
        else:
            if "_ema" in state["model"]:
                del state["model"]["_ema"]
            model.load_state_dict(state["model"], strict=True)

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

        x_before_proj = x

        if self.proj:
            x = self.proj(x)

        '''
        (Pdb) x.size()
        torch.Size([538, 7, 768])
        (Pdb) self.proj
        Linear(in_features=1024, out_features=768, bias=True)
        '''

        return {
            "encoder_out": x,  # T x B x C
            "padding_mask": padding_mask,  # B x T,
            "layer_results": res["layer_results"],
            "encoder_out_before_proj": x_before_proj,
        }

    def forward_torchscript(self, net_input):
        if torch.jit.is_scripting():
            return self.forward(net_input["source"], net_input["padding_mask"])
        else:
            return self.forward_non_torchscript(net_input)

    def reorder_encoder_out(self, encoder_out, new_order):

        # tmp_dict = {}
        # for k,v in encoder_out.items():
        #     if type(v) == torch.Tensor:
        #         tmp_dict[k]=v.clone()
        #     else:
        #         tmp_dict[k]=v
        # encoder_out = tmp_dict

        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = encoder_out["encoder_out"].index_select(
                1, new_order
            )
        if encoder_out["padding_mask"] is not None:
            encoder_out["padding_mask"] = encoder_out["padding_mask"].index_select(
                0, new_order
            )
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
            nn.init.normal_(self.embed_out, mean=0, std=self.output_embed_dim**-0.5)

        if transformer_cfg.decoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

    def forward(
        self, prev_output_tokens, encoder_out=None, incremental_state=None, soft_input=None, **unused
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
        # if type(prev_output_tokens)==list:
        prev_output_tokens = prev_output_tokens.long()
        max_len = prev_output_tokens.size(1)

        if soft_input is not None:
            bos = prev_output_tokens[0,0]
            tmp = torch.zeros(soft_input.size(0),1,soft_input.size(-1))
            tmp[:,:,bos] = 1
            soft_input = torch.cat((tmp.type_as(soft_input), soft_input[:,:,:]), 1)
            soft_input = soft_input[:,:max_len,:]

            # input_padding_mask = (prev_output_tokens == self.padding_idx).unsqueeze(-1).expand(soft_input.size())
            padding_start_idx = torch.sum((prev_output_tokens != self.padding_idx),1)
            padding_vector = torch.zeros(1,soft_input.size(-1))
            padding_vector[:,self.padding_idx] = 1
            padding_vector = padding_vector.type_as(soft_input)

            for i, idx in enumerate(padding_start_idx): 
                soft_input[i][idx:] = padding_vector.repeat(max_len-idx,1)

            soft_input = F.softmax(soft_input, dim = -1)

        x, extra = self.extract_features(
            prev_output_tokens, encoder_out, incremental_state, soft_input
        )
        x = self.output_layer(x)
        return x, extra

    def extract_features(
        self, prev_output_tokens, encoder_out=None, incremental_state=None, soft_input=None, **unused
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
        if soft_input is None:
            x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        else:
            x = self.embed_scale * F.linear(soft_input, self.embed_tokens.weight.T)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None

        inner_states = [x]

        '''
        (Pdb) prev_output_tokens.eq(self.padding_idx)
        tensor([[False, False, False,  ...,  True,  True,  True],
                [False, False, False,  ...,  True,  True,  True],
                [False, False, False,  ..., False, False, False],
                ...,
                [False, False, False,  ...,  True,  True,  True],
                [False, False, False,  ...,  True,  True,  True],
                [False, False, False,  ...,  True,  True,  True]], device='cuda:0')
        (Pdb) prev_output_tokens.eq(self.padding_idx).size()
        torch.Size([7, 209])
        (Pdb) x.size()
        torch.Size([209, 7, 768])
        (Pdb) encoder_out["encoder_out"].size()
        torch.Size([538, 7, 768])
        (Pdb) encoder_out["padding_mask"]
        (Pdb) self_attn_mask=self.buffered_future_mask(x) if incremental_state is None else None
        (Pdb) self_attn_mask
        tensor([[0., -inf, -inf,  ..., -inf, -inf, -inf],
                [0., 0., -inf,  ..., -inf, -inf, -inf],
                [0., 0., 0.,  ..., -inf, -inf, -inf],
                ...,
                [0., 0., 0.,  ..., 0., -inf, -inf],
                [0., 0., 0.,  ..., 0., 0., -inf],
                [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', dtype=torch.float16)
        '''
        # decoder layers
        self_attn_padding_mask = None
        if prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)
        for layer in self.layers:
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.layerdrop):
                '''
                (Pdb) x.size(); encoder_out["encoder_out"].size();
                torch.Size([116, 16, 768])
                torch.Size([424, 16, 768])
                '''

                x, attn, _ = layer(
                    x,
                    encoder_out["encoder_out"] if encoder_out is not None else None,
                    encoder_out["padding_mask"] if encoder_out is not None else None,
                    incremental_state,
                    self_attn_mask=self.buffered_future_mask(x) if incremental_state is None else None,
                    self_attn_padding_mask=self_attn_padding_mask,
                    need_attn=True,
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


# from fairseq.models import FairseqIncrementalDecoder
# from fairseq.models.lstm import LSTMDecoder

# from torch.nn.utils.rnn import PackedSequence
# import torch.nn.functional as F
# from torch import Tensor

# @dataclass
# class SRUppDecoderConfig(FairseqDataclass):
#     ema_decay: float = field(
#         default=0.9999, metadata={"help": "decay for exponential moving average model"}
#     )
#     ema_fp32: bool = field(
#         default=False,
#         metadata={"help": "If true, store EMA model in fp32 even if model is in fp16"},
#     )
#     input_size: int = field(
#         default=None, metadata={"help": "tmp"}
#     )
#     hidden_size: int = field(
#         default=768, metadata={"help": "tmp"}
#     )
#     proj_size: int = field(
#         default=256, metadata={"help": "tmp"}
#     )
#     num_layers: int = field(
#         default=2, metadata={"help": "tmp"}
#     )
#     dropout: float = field(
#         default=0.0, metadata={"help": "tmp"}
#     )
#     attn_dropout: float = field(
#         default=0.0, metadata={"help": "tmp"}
#     )
#     num_heads: int = field(
#         default=1, metadata={"help": "tmp"}
#     )
#     bidirectional: bool = field(
#         default=False, metadata={"help": "tmp"}
#     )
#     layer_norm: bool = field(
#         default=False, metadata={"help": "tmp"}
#     )
#     normalize_after: bool = field(
#         default=False, metadata={"help": "tmp"}
#     )
#     attn_layer_norm: bool = field(
#         default=False, metadata={"help": "tmp"}
#     )
#     highway_bias: float = field(
#         default=-2.0, metadata={"help": "tmp"}
#     )
#     attention_every_n_layers: int = field(
#         default=1, metadata={"help": "tmp"}
#     )
#     attention_last_n_layers: int = field(
#         default=-1, metadata={"help": "tmp"}
#     )
#     rescale: bool = field(
#         default=False, metadata={"help": "tmp"}
#     )
#     nn_rnn_compatible_return: bool = field(
#         default=False, metadata={"help": "tmp"}
#     )
#     proj_input_to_hidden_first: bool = field(
#         default=False, metadata={"help": "tmp"}
#     )
#     weight_c_init: float = field(
#         default=1.0, metadata={"help": "tmp"}
#     )
#     dropout_out: float = field(
#         default=0.0, metadata={"help": "tmp"}
#     )

#     unroll_size: int = field(
#         default=64, metadata={"help": "tmp"}
#     )
    

# class SRUppDecoder(LSTMDecoder):
#     def __init__(
#         self,
#         cfg: Wav2Vec2Seq2SeqConfig,
#         dictionary,
#         embed_tokens,
#         ):
#         super().__init__(dictionary)

#         try:
#             from sru import SRUppCell, SRUppAttention, SRUppProjectedLinear
#         except ImportError as e:
#             print("You have to install sru (e.g. pip install sru) check https://github.com/asappresearch/sru/tree/3.0.0-dev")

#         self.cfg = SRUppDecoderConfig(
#             input_size = embed_tokens.embedding_dim,
#             hidden_size = cfg.decoder_embed_dim,
#             num_layers = cfg.decoder_layers,
#             dropout = cfg.decoder_dropout,
#             attn_dropout = cfg.decoder_attention_dropout,
#         )

#         self.dictionary = dictionary
#         self.num_embeddings = len(dictionary)
#         self.embed_tokens = embed_tokens
#         self.padding_idx = embed_tokens.padding_idx
#         self.share_input_output_embed = cfg.share_decoder_input_output_embed

#         self.input_size = self.cfg.input_size # the number of input features
#         self.hidden_size = self.cfg.hidden_size # the number of features in the hidden state *for each direction*
#         self.proj_size = self.cfg.proj_size # the number of features used for attention
#         self.output_size = self.hidden_size
#         self.num_layers = self.cfg.num_layers # the number of stacked SRU++ layers

#         self.dropout = self.cfg.dropout # dropout probability applied between sub-layers
#         self.bidirectional = self.cfg.bidirectional
#         self.num_directions = 2 if self.bidirectional else 1

#         self.normalize_after = self.cfg.normalize_after

#         self.use_layer_norm = self.cfg.layer_norm # whether to apply layer normalization to each SRU++ layer
#         self.layer_norm = self.cfg.layer_norm
#         self.highway_bias = self.cfg.highway_bias # the initial value of the bias used in the highway (sigmoid) gate (default=-1.0)

#         self.nn_rnn_compatible_return = self.cfg.nn_rnn_compatible_return 
#         self.input_to_hidden: Optional[nn.Module] = None

#         self.num_heads = self.cfg.num_heads
#         self.attn_dropout = self.cfg.attn_dropout
#         self.attn_layer_norm = self.cfg.attn_layer_norm # whether to apply layer norm in the attention module or projected linear module if attention is disabled (default=True).
#         self.attention_last_n_layers = self.cfg.attention_last_n_layers
#         self.attention_every_n_layers = self.cfg.attention_every_n_layers
#         self.unroll_size = self.cfg.unroll_size

#         self.rescale = self.cfg.rescale
#         self.proj_input_to_hidden_first = self.cfg.proj_input_to_hidden_first

#         self.weight_c_init = self.cfg.weight_c_init

#         if self.proj_input_to_hidden_first and self.input_size != self.output_size:
#             first_layer_input_size = self.output_size
#             self.input_to_hidden = nn.Linear(self.input_size, self.output_size, bias=False)
#             nn.init.xavier_uniform_(self.input_to_hidden.weight)
#         else:
#             first_layer_input_size = self.input_size

#         # attention configuration
#         if self.attention_last_n_layers != -1:
#             use_attention = lambda ind: self.num_layers - ind <= self.attention_last_n_layers  # noqa
#         else:
#             use_attention = lambda ind: (ind + 1) % self.attention_every_n_layers == 0  # noqa

#         self.layers = nn.ModuleList()
#         for i in range(self.num_layers):
#             # create the i-th SRU layer
#             in_features = first_layer_input_size if i == 0 else self.output_size
#             proj_features = self.proj_size
#             out_features = self.output_size * (3 if in_features == self.output_size else 4)
#             custom_m: Optional[nn.Module] = None

#             if use_attention(i):
#                 custom_m = SRUppAttention(
#                     in_features,
#                     out_features,
#                     proj_features,
#                     dropout=self.dropout,
#                     attn_dropout=self.attn_dropout,
#                     num_heads=self.num_heads,
#                     layer_norm=self.attn_layer_norm,
#                 )
#             else:
#                 custom_m = SRUppProjectedLinear(
#                     in_features,
#                     out_features,
#                     proj_features,
#                     dropout=self.dropout,
#                     layer_norm=self.attn_layer_norm,
#                 )

#             layer = SRUppCell(
#                 in_features,
#                 self.hidden_size,
#                 dropout=self.dropout if i + 1 != self.num_layers else 0,
#                 bidirectional=self.bidirectional,
#                 layer_norm=self.layer_norm,
#                 normalize_after=self.normalize_after,
#                 highway_bias=self.highway_bias,
#                 rescale=self.rescale,
#                 transform_module=custom_m,
#                 weight_c_init=self.weight_c_init,
#             )
#             self.layers.append(layer)

#         if not self.share_input_output_embed:
#             self.embed_out = nn.Parameter(
#                 torch.Tensor(self.num_embeddings, self.output_size)
#             )
#             nn.init.normal_(self.embed_out, mean=0, std=self.output_size**-0.5)

#         self.dropout_out = torch.nn.Dropout(p=self.cfg.dropout_out)

#         '''
#         (Pdb) self.layers
#         ModuleList(
#         (0): SRUppCell(768, 768, dropout=0.1, highway_bias=-2.0,
#             transform_module=SRUppAttention(
#             (dropout): Dropout(p=0.1, inplace=False)
#             (linear1): Linear(in_features=768, out_features=256, bias=False)
#             (linear2): Linear(in_features=256, out_features=512, bias=False)
#             (linear3): Linear(in_features=256, out_features=2304, bias=False)
#         )
#         )
#         (1): SRUppCell(768, 768, dropout=0.1, highway_bias=-2.0,
#             transform_module=SRUppAttention(
#             (dropout): Dropout(p=0.1, inplace=False)
#             (linear1): Linear(in_features=768, out_features=256, bias=False)
#             (linear2): Linear(in_features=256, out_features=512, bias=False)
#             (linear3): Linear(in_features=256, out_features=2304, bias=False)
#         )
#         )
#         )
#         '''

#         # Tra()

#     def forward(
#         self,
#         prev_output_tokens,
#         encoder_out: Optional[Tuple[Tensor, Tensor, Tensor, Tensor]] = None,
#         incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
#         src_lengths: Optional[Tensor] = None,
#         **unused
#     ):
#         x, attn_scores = self.extract_features(
#             prev_output_tokens, encoder_out, incremental_state
#         )
#         return self.output_layer(x), attn_scores

#     def extract_features(
#         self,
#         prev_output_tokens,
#         encoder_out: Optional[Tuple[Tensor, Tensor, Tensor, Tensor]] = None,
#         incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
#         hidden: Optional[Tensor] = None,
#         memory: Optional[List[Optional[Tensor]]] = None,
#         memory_mask_pad: Optional[Tensor] = None,
#         **unused
#     ):

#         """
#         The forward method of SRUpp module should be like ...

#         Inputs
#         ----------
#         input: Tensor
#             the input feature. shape: (length, batch_size, input_size)
#         c0: Tensor, optional
#             the initial internal hidden state. shape: (num_layers,
#             batch_size, output_size) where
#             output_size = hidden_size * num_direction
#         mask_pad: Tensor, optional
#             the mask where a non-zero value indicates if an input token
#             is pad token that should be ignored in forward and backward
#             computation. shape: (length, batch_size)
#         memory: a list of optional tensors, optional
#             a list of memory tensors as additional inputs for the attention
#             to attend to. the size of the list is equal to the number of layers
#             of SRUpp module. memory[i] is the memory tensor for the (i+1)-th
#             layer and its second dimension (batch size) and third dimension
#             (hidden size) must be compatible with the input tensor to the
#             (i+1)-th layer.
#         memory_mask_pad: tensor, optional
#             the mask tensor indicate if a position in the memory tensors is
#             an invalid / pad token that should be ignored in attention.
#             shape: (memory_length, batch_size)

#         Returns
#         ----------
#         h: Tensor
#             the output hidden state. shape: (length, batch_size,
#             output_size) where
#             output_size = hidden_size * num_direction
#         c: Tensor
#             the last internal hidden state. shape: (num_layers,
#             batch_size, output_size), or (num_layers * num_directions,
#             batch_size, hidden_size) if `nn_rnn_compatible_return` is
#             set `True`
#         memory_bank: Dict[str, List[Tensor]]
#             a dictionary that stores various internal states indexed
#             by state names. each value is a list of tensors in which
#             the i-th element is the state tensor of the (i+1)-th layer.
#             these internal states can be reused for attention for the
#             next forward call during training and decoding.
#         """

#         if incremental_state is not None and len(incremental_state) > 0:
#             prev_output_tokens = prev_output_tokens[:, -1:]

#         input = self.embed_tokens(prev_output_tokens).transpose(0,1)
#         mask_pad = (prev_output_tokens==self.padding_idx).transpose(0,1)

#         length = input.size(0)
#         bsz = input.size(1)
#         input_size = input.size(2)
#         num_layers = self.num_layers
#         output_size = self.output_size

#         c0 = self.init_hidden(bsz) if hidden is None else hidden
#         memory = [None] * self.num_layers if memory is None else memory

#         mem_len = 0 if memory[0] is None else memory[0].size(0)
#         attn_mask = self.get_attn_mask(self.unroll_size, same_length=False)
#         attn_mask = attn_mask[mem_len:mem_len + length, :mem_len + length]
#         attn_mask = attn_mask.to(input.device)

#         orig_input = input
#         if isinstance(orig_input, PackedSequence):
#             input, lengths = nn.utils.rnn.pad_packed_sequence(input)
#             max_length = lengths.max().item()
#             mask_pad = torch.ByteTensor([[0] * length + [1] * (max_length - length)
#                                         for length in lengths.tolist()])
#             mask_pad = mask_pad.to(input.device).transpose(0, 1).contiguous()

#         # The dimensions of `input` should be: `(sequence_length, batch_size, input_size)`.
#         if input.dim() != 3:
#             raise ValueError("There must be 3 dimensions for (length, batch_size, input_size)")

#         # # initialize previous states (or get from cache during incremental generation)
#         # if incremental_state is not None and len(incremental_state) > 0:
#         #     prev_hiddens, prev_cells, input_feed = self.get_cached_state(
#         #         incremental_state
#         #     )


#         if input_size != self.input_size:
#             raise ValueError("Input has size (*, *, {}) but expect a last dimension of {}".format(
#                 input_size, self.input_size
#             ))

#         if c0 is None:
#             zeros = torch.zeros(bsz, output_size, dtype=input.dtype, device=input.device)
#             c0_ = [zeros for i in range(num_layers)]
#         else:
#             if list(c0.size()) != [num_layers, bsz, output_size]:
#                 raise ValueError("c0 has size {} but expect {}.".format(
#                     list(c0.size()),
#                     [num_layers, bsz, output_size]
#                 ))
#             c0_ = [x.squeeze(0) for x in c0.chunk(self.num_layers, 0)]

#         if mask_pad is not None and list(mask_pad.size()) != [length, bsz]:
#             raise ValueError("mask_pad has size {} but expect {}.".format(
#                 list(mask_pad.size()),
#                 [length, bsz]
#             ))

#         if memory is not None and not isinstance(memory, list):
#             raise ValueError("memory has type {} but expect List[Tensor].".format(
#                 type(memory)
#             ))

#         if memory is not None and len(memory) != num_layers:
#             raise ValueError("memory has size {} but expect {}.".format(
#                 len(memory),
#                 num_layers
#             ))

#         if self.input_to_hidden is None:
#             x = input
#         else:
#             x = self.input_to_hidden(input)

#         # Tra()

#         prev_inputs = []
#         lstc = []
#         i = 0
#         x = x.contiguous()
#         for layer in self.layers:
#             prev_inputs.append(x)
#             memory_i = memory[i] if memory is not None else None
#             h, c = layer(x, c0_[i],
#                        mask_pad=mask_pad,
#                        attn_mask=attn_mask,
#                        memory=memory_i,
#                        memory_mask_pad=memory_mask_pad)
#             x = h
#             lstc.append(c)
#             i += 1
#             '''
#             (Pdb) x.size(); lstc[0].size(); 
#             torch.Size([50, 16, 768])
#             torch.Size([16, 768])
#             '''
#             # Tra()

#         lstc_stack = torch.stack(lstc)
#         if self.nn_rnn_compatible_return:
#             lstc_stack = lstc_stack.view(num_layers, bsz, self.num_directions, self.hidden_size)
#             lstc_stack = lstc_stack.transpose(1, 2).contiguous()
#             lstc_stack = lstc_stack.view(num_layers * self.num_directions, bsz, self.hidden_size)

#         if isinstance(orig_input, PackedSequence):
#             h = nn.utils.rnn.pack_padded_sequence(h, lengths, enforce_sorted=False)

#         Tra()

#         # # Stack all the necessary tensors together and store
#         # prev_hiddens_tensor = torch.stack(prev_hiddens)
#         # prev_cells_tensor = torch.stack(prev_cells)
#         # cache_state = torch.jit.annotate(
#         #     Dict[str, Optional[Tensor]],
#         #     {
#         #         "prev_hiddens": prev_hiddens_tensor,
#         #         "prev_cells": prev_cells_tensor,
#         #         "input_feed": input_feed,
#         #     },
#         # )
#         # self.set_incremental_state(incremental_state, "cached_state", cache_state)

#         out = self.dropout_out(h)
#         out = self.output_layer(out).transpose(0,1)
#         attn_scores = None

#         return out, (lstc_stack, {'saved_inputs': prev_inputs}, attn_scores)

#     def init_hidden(self, batch_size):
#         weight = next(self.parameters()).data
#         zeros = weight.new(self.num_layers, batch_size, self.hidden_size).zero_()
#         return zeros

#     def get_attn_mask(self, mem_length, same_length=False):
#         weight = next(self.parameters())
#         ones = weight.new_ones(mem_length * 2, mem_length * 2)
#         if same_length:
#             ''' example: mem_length = 3
#                 0 1 1 1 1 1
#                 0 0 1 1 1 1
#                 0 0 0 1 1 1
#                 0 0 0 0 1 1
#                 1 0 0 0 0 1
#                 1 1 0 0 0 0
#             '''
#             attn_mask = (torch.triu(ones, diagonal=1) +
#                               torch.tril(ones, diagonal=-1-mem_length)) * -10000.0
#         else:
#             attn_mask = torch.triu(ones, diagonal=1) * -10000.0
        
#         return attn_mask

#     def output_layer(self, x):
#         """Project features to the vocabulary size."""
#         if self.share_input_output_embed:
#             x = F.linear(x, self.embed_tokens.weight)
#         else:
#             x = F.linear(x, self.embed_out)
#         return x

#     def reset_parameters(self):
#         for layer in self.layers:
#             layer.reset_parameters()
#         if self.input_to_hidden is not None:
#             nn.init.xavier_uniform_(self.input_to_hidden.weight)


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


########################################################################
########################     for Transducer     ########################
########################################################################

'''
This is from torchaudio,

https://github.com/pytorch/audio/blob/release/0.11/torchaudio/models/rnnt.py
'''

class _TimeReduction(torch.nn.Module):
    r"""Coalesces frames along time dimension into a
    fewer number of frames with higher feature dimensionality.
    Args:
        stride (int): number of frames to merge for each output frame.
    """

    def __init__(self, stride: int) -> None:
        super().__init__()
        self.stride = stride

    def forward(self, input: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Forward pass.
        B: batch size;
        T: maximum input sequence length in batch;
        D: feature dimension of each input sequence frame.
        Args:
            input (torch.Tensor): input sequences, with shape `(B, T, D)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``input``.
        Returns:
            (torch.Tensor, torch.Tensor):
                torch.Tensor
                    output sequences, with shape
                    `(B, T  // stride, D * stride)`
                torch.Tensor
                    output lengths, with shape `(B,)` and i-th element representing
                    number of valid frames for i-th batch element in output sequences.
        """
        B, T, D = input.shape
        num_frames = T - (T % self.stride)
        input = input[:, :num_frames, :]
        lengths = lengths.div(self.stride, rounding_mode="trunc")
        T_max = num_frames // self.stride

        output = input.reshape(B, T_max, D * self.stride)
        output = output.contiguous()
        return output, lengths

class _Transcriber(torch.nn.Module):
    r"""Recurrent neural network transducer (RNN-T) transcription network.
    Args:
        input_dim (int): feature dimension of each input sequence element.
        output_dim (int): feature dimension of each output sequence element.
        segment_length (int): length of input segment expressed as number of frames.
        right_context_length (int): length of right context expressed as number of frames.
        time_reduction_input_dim (int): dimension to scale each element in input sequences to
            prior to applying time reduction block.
        time_reduction_stride (int): factor by which to reduce length of input sequence.
        transformer_num_heads (int): number of attention heads in each Emformer layer.
        transformer_ffn_dim (int): hidden layer dimension of each Emformer layer's feedforward network.
        transformer_num_layers (int): number of Emformer layers to instantiate.
        transformer_left_context_length (int): length of left context.
        transformer_dropout (float, optional): transformer dropout probability. (Default: 0.0)
        transformer_activation (str, optional): activation function to use in each Emformer layer's
            feedforward network. Must be one of ("relu", "gelu", "silu"). (Default: "relu")
        transformer_max_memory_size (int, optional): maximum number of memory elements to use. (Default: 0)
        transformer_weight_init_scale_strategy (str, optional): per-layer weight initialization scaling
            strategy. Must be one of ("depthwise", "constant", ``None``). (Default: "depthwise")
        transformer_tanh_on_mem (bool, optional): if ``True``, applies tanh to memory elements. (Default: ``False``)
    """

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        segment_length: int,
        right_context_length: int,
        time_reduction_input_dim: int,
        time_reduction_stride: int,
        transformer_num_heads: int,
        transformer_ffn_dim: int,
        transformer_num_layers: int,
        transformer_left_context_length: int,
        transformer_dropout: float = 0.0,
        transformer_activation: str = "relu",
        transformer_max_memory_size: int = 0,
        transformer_weight_init_scale_strategy: str = "depthwise",
        transformer_tanh_on_mem: bool = False,
        pretrained_encoder = None,
    ) -> None:
        super().__init__()


        self.pretrained_encoder = pretrained_encoder
        if pretrained_encoder :
            self.transformer = pretrained_encoder
        else : 
            self.input_linear = torch.nn.Linear(
                input_dim,
                time_reduction_input_dim,
                bias=False,
            )
            self.time_reduction = _TimeReduction(time_reduction_stride)
            transformer_input_dim = time_reduction_input_dim * time_reduction_stride
            from torchaudio.models import Emformer
            self.transformer = Emformer(
                transformer_input_dim,
                transformer_num_heads,
                transformer_ffn_dim,
                transformer_num_layers,
                segment_length // time_reduction_stride,
                dropout=transformer_dropout,
                activation=transformer_activation,
                left_context_length=transformer_left_context_length,
                right_context_length=right_context_length // time_reduction_stride,
                max_memory_size=transformer_max_memory_size,
                weight_init_scale_strategy=transformer_weight_init_scale_strategy,
                tanh_on_mem=transformer_tanh_on_mem,
            )
            self.output_linear = torch.nn.Linear(transformer_input_dim, output_dim)
            self.layer_norm = torch.nn.LayerNorm(output_dim)

    def forward(self, input: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Forward pass for training.
        B: batch size;
        T: maximum input sequence length in batch;
        D: feature dimension of each input sequence frame (input_dim).
        Args:
            input (torch.Tensor): input frame sequences right-padded with right context, with
                shape `(B, T + right context length, D)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``input``.
        Returns:
            (torch.Tensor, torch.Tensor):
                torch.Tensor
                    output frame sequences, with
                    shape `(B, T // time_reduction_stride, output_dim)`.
                torch.Tensor
                    output input lengths, with shape `(B,)` and i-th element representing
                    number of valid elements for i-th batch element in output frame sequences.
        """
        
        if self.pretrained_encoder is not None : 
            encoder_out = self.encoder(input)
        else :
            input_linear_out = self.input_linear(input)
            time_reduction_out, time_reduction_lengths = self.time_reduction(input_linear_out, lengths)
            transformer_out, transformer_lengths = self.transformer(time_reduction_out, time_reduction_lengths)
            output_linear_out = self.output_linear(transformer_out)
            layer_norm_out = self.layer_norm(output_linear_out)
            return layer_norm_out, transformer_lengths

    @torch.jit.export
    def infer(
        self,
        input: torch.Tensor,
        lengths: torch.Tensor,
        states: Optional[List[List[torch.Tensor]]],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        r"""Forward pass for inference.
        B: batch size;
        T: maximum input sequence segment length in batch;
        D: feature dimension of each input sequence frame (input_dim).
        Args:
            input (torch.Tensor): input frame sequence segments right-padded with right context, with
                shape `(B, T + right context length, D)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``input``.
            state (List[List[torch.Tensor]] or None): list of lists of tensors
                representing internal state generated in preceding invocation
                of ``infer``.
        Returns:
            (torch.Tensor, torch.Tensor, List[List[torch.Tensor]]):
                torch.Tensor
                    output frame sequences, with
                    shape `(B, T // time_reduction_stride, output_dim)`.
                torch.Tensor
                    output input lengths, with shape `(B,)` and i-th element representing
                    number of valid elements for i-th batch element in output.
                List[List[torch.Tensor]]
                    output states; list of lists of tensors
                    representing internal state generated in current invocation
                    of ``infer``.
        """

        raise NotImplementedError("!!! Not supported streaming mode. This is for batch transducer.")

        input_linear_out = self.input_linear(input)
        time_reduction_out, time_reduction_lengths = self.time_reduction(input_linear_out, lengths)
        (
            transformer_out,
            transformer_lengths,
            transformer_states,
        ) = self.transformer.infer(time_reduction_out, time_reduction_lengths, states)
        output_linear_out = self.output_linear(transformer_out)
        layer_norm_out = self.layer_norm(output_linear_out)
        return layer_norm_out, transformer_lengths, transformer_states



class _CustomLSTM(torch.nn.Module):
    r"""Custom long-short-term memory (LSTM) block that applies layer normalization
    to internal nodes.
    Args:
        input_dim (int): input dimension.
        hidden_dim (int): hidden dimension.
        layer_norm (bool, optional): if ``True``, enables layer normalization. (Default: ``False``)
        layer_norm_epsilon (float, optional):  value of epsilon to use in
            layer normalization layers (Default: 1e-5)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        layer_norm: bool = False,
        layer_norm_epsilon: float = 1e-5,
    ) -> None:
        super().__init__()
        self.x2g = torch.nn.Linear(input_dim, 4 * hidden_dim, bias=(not layer_norm))
        self.p2g = torch.nn.Linear(hidden_dim, 4 * hidden_dim, bias=False)
        if layer_norm:
            self.c_norm = torch.nn.LayerNorm(hidden_dim, eps=layer_norm_epsilon)
            self.g_norm = torch.nn.LayerNorm(4 * hidden_dim, eps=layer_norm_epsilon)
        else:
            self.c_norm = torch.nn.Identity()
            self.g_norm = torch.nn.Identity()

        self.hidden_dim = hidden_dim

    def forward(
        self, input: torch.Tensor, state: Optional[List[torch.Tensor]]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        r"""Forward pass.
        B: batch size;
        T: maximum sequence length in batch;
        D: feature dimension of each input sequence element.
        Args:
            input (torch.Tensor): with shape `(T, B, D)`.
            state (List[torch.Tensor] or None): list of tensors
                representing internal state generated in preceding invocation
                of ``forward``.
        Returns:
            (torch.Tensor, List[torch.Tensor]):
                torch.Tensor
                    output, with shape `(T, B, hidden_dim)`.
                List[torch.Tensor]
                    list of tensors representing internal state generated
                    in current invocation of ``forward``.
        """
        if state is None:
            B = input.size(1)
            h = torch.zeros(B, self.hidden_dim, device=input.device, dtype=input.dtype)
            c = torch.zeros(B, self.hidden_dim, device=input.device, dtype=input.dtype)
        else:
            h, c = state

        gated_input = self.x2g(input)
        outputs = []
        for gates in gated_input.unbind(0):
            gates = gates + self.p2g(h)
            gates = self.g_norm(gates)
            input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, 1)
            input_gate = input_gate.sigmoid()
            forget_gate = forget_gate.sigmoid()
            cell_gate = cell_gate.tanh()
            output_gate = output_gate.sigmoid()
            c = forget_gate * c + input_gate * cell_gate
            c = self.c_norm(c)
            h = output_gate * c.tanh()
            outputs.append(h)

        output = torch.stack(outputs, dim=0)
        state = [h, c]

        return output, state

class _Predictor(torch.nn.Module):
    r"""Recurrent neural network transducer (RNN-T) prediction network.
    Args:
        num_symbols (int): size of target token lexicon.
        output_dim (int): feature dimension of each output sequence element.
        symbol_embedding_dim (int): dimension of each target token embedding.
        num_lstm_layers (int): number of LSTM layers to instantiate.
        lstm_layer_norm (bool, optional): if ``True``, enables layer normalization
            for LSTM layers. (Default: ``False``)
        lstm_layer_norm_epsilon (float, optional): value of epsilon to use in
            LSTM layer normalization layers. (Default: 1e-5)
        lstm_dropout (float, optional): LSTM dropout probability. (Default: 0.0)
    """

    def __init__(
        self,
        num_symbols: int,
        output_dim: int,
        symbol_embedding_dim: int,
        num_lstm_layers: int,
        lstm_layer_norm: bool = False,
        lstm_layer_norm_epsilon: float = 1e-5,
        lstm_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(num_symbols, symbol_embedding_dim)
        self.input_layer_norm = torch.nn.LayerNorm(symbol_embedding_dim)
        self.lstm_layers = torch.nn.ModuleList(
            [
                _CustomLSTM(
                    symbol_embedding_dim,
                    symbol_embedding_dim,
                    layer_norm=lstm_layer_norm,
                    layer_norm_epsilon=lstm_layer_norm_epsilon,
                )
                for idx in range(num_lstm_layers)
            ]
        )
        self.dropout = torch.nn.Dropout(p=lstm_dropout)
        self.linear = torch.nn.Linear(symbol_embedding_dim, output_dim)
        self.output_layer_norm = torch.nn.LayerNorm(output_dim)

        self.lstm_dropout = lstm_dropout

    def forward(
        self,
        input: torch.Tensor,
        lengths: torch.Tensor,
        state: Optional[List[List[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        r"""Forward pass.
        B: batch size;
        U: maximum sequence length in batch;
        D: feature dimension of each input sequence element.
        Args:
            input (torch.Tensor): target sequences, with shape `(B, U)` and each element
                mapping to a target symbol, i.e. in range `[0, num_symbols)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``input``.
            state (List[List[torch.Tensor]] or None, optional): list of lists of tensors
                representing internal state generated in preceding invocation
                of ``forward``. (Default: ``None``)
        Returns:
            (torch.Tensor, torch.Tensor, List[List[torch.Tensor]]):
                torch.Tensor
                    output encoding sequences, with shape `(B, U, output_dim)`
                torch.Tensor
                    output lengths, with shape `(B,)` and i-th element representing
                    number of valid elements for i-th batch element in output encoding sequences.
                List[List[torch.Tensor]]
                    output states; list of lists of tensors
                    representing internal state generated in current invocation of ``forward``.
        """
        input_tb = input.permute(1, 0)
        embedding_out = self.embedding(input_tb)
        input_layer_norm_out = self.input_layer_norm(embedding_out)

        lstm_out = input_layer_norm_out
        state_out: List[List[torch.Tensor]] = []
        for layer_idx, lstm in enumerate(self.lstm_layers):
            lstm_out, lstm_state_out = lstm(lstm_out, None if state is None else state[layer_idx])
            lstm_out = self.dropout(lstm_out)
            state_out.append(lstm_state_out)

        linear_out = self.linear(lstm_out)
        output_layer_norm_out = self.output_layer_norm(linear_out)
        return output_layer_norm_out.permute(1, 0, 2), lengths, state_out


class _Joiner(torch.nn.Module):
    r"""Recurrent neural network transducer (RNN-T) joint network.
    Args:
        input_dim (int): source and target input dimension.
        output_dim (int): output dimension.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=True)
        self.relu = torch.nn.ReLU()

    def forward(
        self,
        source_encodings: torch.Tensor,
        source_lengths: torch.Tensor,
        target_encodings: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Forward pass for training.
        B: batch size;
        T: maximum source sequence length in batch;
        U: maximum target sequence length in batch;
        D: dimension of each source and target sequence encoding.
        Args:
            source_encodings (torch.Tensor): source encoding sequences, with
                shape `(B, T, D)`.
            source_lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                valid sequence length of i-th batch element in ``source_encodings``.
            target_encodings (torch.Tensor): target encoding sequences, with shape `(B, U, D)`.
            target_lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                valid sequence length of i-th batch element in ``target_encodings``.
        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor):
                torch.Tensor
                    joint network output, with shape `(B, T, U, output_dim)`.
                torch.Tensor
                    output source lengths, with shape `(B,)` and i-th element representing
                    number of valid elements along dim 1 for i-th batch element in joint network output.
                torch.Tensor
                    output target lengths, with shape `(B,)` and i-th element representing
                    number of valid elements along dim 2 for i-th batch element in joint network output.
        """
        joint_encodings = source_encodings.unsqueeze(2).contiguous() + target_encodings.unsqueeze(1).contiguous()
        relu_out = self.relu(joint_encodings)
        output = self.linear(relu_out)
        return output, source_lengths, target_lengths



class RNNT(torch.nn.Module):
    r"""torchaudio.models.RNNT()
    Recurrent neural network transducer (RNN-T) model.
    Note:
        To build the model, please use one of the factory functions.
    Args:
        transcriber (torch.nn.Module): transcription network.
        predictor (torch.nn.Module): prediction network.
        joiner (torch.nn.Module): joint network.
    """

    def __init__(self, transcriber: _Transcriber, predictor: _Predictor, joiner: _Joiner) -> None:
        super().__init__()
        self.transcriber = transcriber
        self.predictor = predictor
        self.joiner = joiner

    def forward(
        self,
        sources: torch.Tensor,
        source_lengths: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
        predictor_state: Optional[List[List[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        r"""Forward pass for training.
        B: batch size;
        T: maximum source sequence length in batch;
        U: maximum target sequence length in batch;
        D: feature dimension of each source sequence element.
        Args:
            sources (torch.Tensor): source frame sequences right-padded with right context, with
                shape `(B, T, D)`.
            source_lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``sources``.
            targets (torch.Tensor): target sequences, with shape `(B, U)` and each element
                mapping to a target symbol.
            target_lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``targets``.
            predictor_state (List[List[torch.Tensor]] or None, optional): list of lists of tensors
                representing prediction network internal state generated in preceding invocation
                of ``forward``. (Default: ``None``)
        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor, List[List[torch.Tensor]]):
                torch.Tensor
                    joint network output, with shape
                    `(B, max output source length, max output target length, output_dim (number of target symbols))`.
                torch.Tensor
                    output source lengths, with shape `(B,)` and i-th element representing
                    number of valid elements along dim 1 for i-th batch element in joint network output.
                torch.Tensor
                    output target lengths, with shape `(B,)` and i-th element representing
                    number of valid elements along dim 2 for i-th batch element in joint network output.
                List[List[torch.Tensor]]
                    output states; list of lists of tensors
                    representing prediction network internal state generated in current invocation
                    of ``forward``.
        """
        source_encodings, source_lengths = self.transcriber(
            input=sources,
            lengths=source_lengths,
        )
        target_encodings, target_lengths, predictor_state = self.predictor(
            input=targets,
            lengths=target_lengths,
            state=predictor_state,
        )
        output, source_lengths, target_lengths = self.joiner(
            source_encodings=source_encodings,
            source_lengths=source_lengths,
            target_encodings=target_encodings,
            target_lengths=target_lengths,
        )

        return (
            output,
            source_lengths,
            target_lengths,
            predictor_state,
        )

    @torch.jit.export
    def transcribe_streaming(
        self,
        sources: torch.Tensor,
        source_lengths: torch.Tensor,
        state: Optional[List[List[torch.Tensor]]],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        r"""Applies transcription network to sources in streaming mode.
        B: batch size;
        T: maximum source sequence segment length in batch;
        D: feature dimension of each source sequence frame.
        Args:
            sources (torch.Tensor): source frame sequence segments right-padded with right context, with
                shape `(B, T + right context length, D)`.
            source_lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``sources``.
            state (List[List[torch.Tensor]] or None): list of lists of tensors
                representing transcription network internal state generated in preceding invocation
                of ``transcribe_streaming``.
        Returns:
            (torch.Tensor, torch.Tensor, List[List[torch.Tensor]]):
                torch.Tensor
                    output frame sequences, with
                    shape `(B, T // time_reduction_stride, output_dim)`.
                torch.Tensor
                    output lengths, with shape `(B,)` and i-th element representing
                    number of valid elements for i-th batch element in output.
                List[List[torch.Tensor]]
                    output states; list of lists of tensors
                    representing transcription network internal state generated in current invocation
                    of ``transcribe_streaming``.
        """
        return self.transcriber.infer(sources, source_lengths, state)

    @torch.jit.export
    def transcribe(
        self,
        sources: torch.Tensor,
        source_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Applies transcription network to sources in non-streaming mode.
        B: batch size;
        T: maximum source sequence length in batch;
        D: feature dimension of each source sequence frame.
        Args:
            sources (torch.Tensor): source frame sequences right-padded with right context, with
                shape `(B, T + right context length, D)`.
            source_lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``sources``.
        Returns:
            (torch.Tensor, torch.Tensor):
                torch.Tensor
                    output frame sequences, with
                    shape `(B, T // time_reduction_stride, output_dim)`.
                torch.Tensor
                    output lengths, with shape `(B,)` and i-th element representing
                    number of valid elements for i-th batch element in output frame sequences.
        """
        return self.transcriber(sources, source_lengths)

    @torch.jit.export
    def predict(
        self,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
        state: Optional[List[List[torch.Tensor]]],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        r"""Applies prediction network to targets.
        B: batch size;
        U: maximum target sequence length in batch;
        D: feature dimension of each target sequence frame.
        Args:
            targets (torch.Tensor): target sequences, with shape `(B, U)` and each element
                mapping to a target symbol, i.e. in range `[0, num_symbols)`.
            target_lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``targets``.
            state (List[List[torch.Tensor]] or None): list of lists of tensors
                representing internal state generated in preceding invocation
                of ``predict``.
        Returns:
            (torch.Tensor, torch.Tensor, List[List[torch.Tensor]]):
                torch.Tensor
                    output frame sequences, with shape `(B, U, output_dim)`.
                torch.Tensor
                    output lengths, with shape `(B,)` and i-th element representing
                    number of valid elements for i-th batch element in output.
                List[List[torch.Tensor]]
                    output states; list of lists of tensors
                    representing internal state generated in current invocation of ``predict``.
        """
        return self.predictor(input=targets, lengths=target_lengths, state=state)

    @torch.jit.export
    def join(
        self,
        source_encodings: torch.Tensor,
        source_lengths: torch.Tensor,
        target_encodings: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Applies joint network to source and target encodings.
        B: batch size;
        T: maximum source sequence length in batch;
        U: maximum target sequence length in batch;
        D: dimension of each source and target sequence encoding.
        Args:
            source_encodings (torch.Tensor): source encoding sequences, with
                shape `(B, T, D)`.
            source_lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                valid sequence length of i-th batch element in ``source_encodings``.
            target_encodings (torch.Tensor): target encoding sequences, with shape `(B, U, D)`.
            target_lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                valid sequence length of i-th batch element in ``target_encodings``.
        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor):
                torch.Tensor
                    joint network output, with shape `(B, T, U, output_dim)`.
                torch.Tensor
                    output source lengths, with shape `(B,)` and i-th element representing
                    number of valid elements along dim 1 for i-th batch element in joint network output.
                torch.Tensor
                    output target lengths, with shape `(B,)` and i-th element representing
                    number of valid elements along dim 2 for i-th batch element in joint network output.
        """
        output, source_lengths, target_lengths = self.joiner(
            source_encodings=source_encodings,
            source_lengths=source_lengths,
            target_encodings=target_encodings,
            target_lengths=target_lengths,
        )
        return output, source_lengths, target_lengths