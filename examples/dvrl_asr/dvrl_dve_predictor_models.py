
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

logger = logging.getLogger(__name__)

from fairseq.models.wav2vec.wav2vec2_asr import (
    Wav2Vec2AsrConfig,
    Wav2VecCtc,
    Wav2Vec2CtcConfig
)

from torch.distributions import Categorical, Bernoulli


@dataclass
class DVRLCtcConfig(Wav2Vec2CtcConfig):
    ## configs for w2v_ctc (refer to Wav2Vec2AsrConfig)
    tmp: float = 0
    moving_average_window: int = field(
        default=10,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )

@register_model("dvrl_ctc", dataclass=DVRLCtcConfig)
class DVRLforASR(BaseFairseqModel):
    def __init__(self, cfg: DVRLCtcConfig, w2v_ctc, dve, sampler):
        super().__init__()
        self.cfg = cfg
        self.blank_weight = cfg.blank_weight
        self.blank_mode = cfg.blank_mode
        self.w2v_ctc = w2v_ctc
        self.dve = dve
        self.sampler = sampler

        self.moving_average_window = cfg.moving_average_window
        self.moving_average_previous_loss = 0

    def update_moving_average_previous_loss(self, reward):
        self.moving_average_previous_loss = (
            ((self.moving_average_window - 1)/self.moving_average_window) 
            + (1/self.moving_average_window) * reward
            )

    @classmethod
    def build_model(cls, cfg: DVRLCtcConfig, task: FairseqTask):
        """Build a new model instance."""
        w2v_ctc, w2v_encoder = cls.build_asr_model(cfg, task)
        dve = cls.build_dve_model(cfg, w2v_encoder)
        sampler = cls.build_sampler(cfg)
        return DVRLforASR(cfg, w2v_ctc, dve, sampler)

    ## Task Specific Predictor, for ASR, w2v2-CTC 
    @classmethod
    def build_asr_model(cls, cfg, task):
        w2v_encoder = Wav2VecEncoder(cfg, len(task.target_dictionary)) ## build w2v2-encoder first 
        w2v_ctc = Wav2VecCtc(cfg, w2v_encoder) ## then, build w2v2-ctc using w2v2-encoder
        return w2v_ctc, w2v_encoder

    ## Data Value Evaluator
    @classmethod
    def build_dve_model(cls, cfg, w2v_encoder):
        dve = DVE(cfg, w2v_encoder)
        return dve

    @classmethod
    def build_sampler(cls, cfg):
        return Sampler(cfg.encoder_embed_dim)

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

    def forward(self, **kwargs):
        x = self.w2v_ctc(**kwargs)
        # dict_keys(['encoder_out', 'padding_mask', 'layer_results'])
        return x


## output multinomial distribution, selection prob
class DVE(BaseFairseqModel):
    def __init__(self, cfg, w2v_encoder):
        super().__init__()
        self.cfg = cfg
        self.w2v_encoder = w2v_encoder

        self.layer = Linear(w2v_encoder.encoder_embed_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

        self.proj = Linear(w2v_encoder.encoder_embed_dim, 1)

    def forward(self, **kwargs):
        x = self.w2v_encoder.w2v_model.extract_features(**kwargs) # (B, T) -> (T, B, C) 
        cnn_output = x['x'] # (B, T, C) 
        padding_mask = x['padding_mask'] # (B, T)

        ## attention pooling score = w2 * tanh ( w1 * H' )
        score = (self.layer(cnn_output)).squeeze(dim=-1) # (B, T, 1), projection to scalar score
        if padding_mask is not None:
            score.data.masked_fill_(padding_mask, -float("inf"))
        attn_weights = self.softmax(score) # compute score

        context = torch.bmm(attn_weights.unsqueeze(dim=1), cnn_output) # (B, 1, C) # weighted sum
        out = context.squeeze(dim=1) # (B, C)

        out = self.proj(out)
        return out


## output binomial distribution using DVE value output, select or not
class Sampler(nn.Module):
     def __init__(self, input_dim):
        super(Sampler, self).__init__()

     def forward(self, logit):
        prob = torch.sigmoid(logit) # 1-dim prob
        prob_dist = Bernoulli(prob) # binary prob distribution
        '''
        torch.distributions example)
        probs = torch.Tensor([ [0.1, 0.2, 0.7], [0.1, 0.8, 0.1], [0.4, 0.5, 0.1] ])
        prob_dist = torch.distributions.Categorical(probs) # probs should be of size batch x classes
        prob_dist.sample() # tensor([2, 1, 1])
        '''
        return prob_dist.sample()


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m

class Wav2VecEncoder(FairseqEncoder):
    def __init__(self, cfg: Wav2Vec2AsrConfig, output_size=None, ctc_proj_dim=None):
        self.apply_mask = cfg.apply_mask

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

        if cfg.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(cfg.w2v_path, arg_overrides)
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

        task = tasks.setup_task(w2v_args.task)
        model = task.build_model(w2v_args.model, from_checkpoint=True)

        model.remove_pretraining_modules()

        if state is not None and not cfg.no_pretrained_weights:
            self.load_model_weights(state, model, cfg)

        super().__init__(task.source_dictionary)

        d = w2v_args.model.encoder_embed_dim

        ## added
        self.encoder_embed_dim = d

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

        self.ctc_proj = None
        if ctc_proj_dim is not None:
            self.ctc_proj = nn.Linear(cfg.encoder_embed_dim, ctc_proj_dim)

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
            encoder_out["padding_mask"] = encoder_out["padding_mask"].index_select(
                0, new_order
            )
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return None

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict
