
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
    # FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
)
from fairseq.models.wav2vec.wav2vec2 import MASKING_DISTRIBUTION_CHOICES
from fairseq.modules import LayerNorm, PositionalEmbedding, TransformerDecoderLayer
from fairseq.tasks import FairseqTask

logger = logging.getLogger(__name__)

from fairseq.models.wav2vec import (
    Wav2Vec2AsrConfig,
    Wav2Vec2CtcConfig,
    Wav2VecCtc,
    Wav2VecEncoder
)

# from examples.data2vec import (
#     Data2VecTextConfig,
#     Data2VecTextModel
# )

from examples.speech_recognition.w2l_decoder import (
    W2lViterbiDecoder
)

from fairseq.models.nat import (
    FairseqNATModel,
    FairseqNATEncoder,
    FairseqNATDecoder
)


import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from fairseq import utils
from fairseq.distributed import fsdp_wrap
from fairseq.models import FairseqIncrementalDecoder
from fairseq.models.transformer import TransformerConfig
from fairseq.modules import (
    AdaptiveSoftmax,
    BaseLayer,
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    transformer_layer,
)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_

from fairseq.data.data_utils import post_process

from transformers import RobertaTokenizer


############################################
############ bimodal data2vec model #############
############ cross attention of audio and text #############
############################################

@dataclass
class Data2VecAudioTextConfig(Wav2Vec2AsrConfig):
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

    w2v_ctc_path: str = field(
        default=MISSING, metadata={"help": "path to wav2vec ctc model"}
    )

    roberta_path: str = field(
        default=MISSING, metadata={"help": "path to roberta model"}
    )


from examples.speech_recognition.new.decoders.decoder_config import (
    DecoderConfig,
    FlashlightDecoderConfig,
)
from typing import Any, Dict, List, Optional, Tuple, Union
from examples.speech_recognition.new.decoders.decoder import Decoder

@dataclass
class DecodingConfig(DecoderConfig, FlashlightDecoderConfig):
    unique_wer_file: bool = field(
        default=False,
        metadata={"help": "If set, use a unique file for storing WER"},
    )
    results_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "If set, write hypothesis and reference sentences into this directory"
        },
    )



@register_model("data2vec_bimodal", dataclass=Data2VecAudioTextConfig)
class Data2VecAudioTextModel(BaseFairseqModel):
    def __init__(self, audio_encoder, asr_eval_decoder, tgt_dict, text_encoder, roberta_src_dict, decoder):
        super().__init__()

        self.device = torch.device("cuda")
        self.audio_encoder = audio_encoder.to(self.device)
        # self.device = next(self.audio_encoder.parameters()).device
        self.asr_eval_decoder = asr_eval_decoder
        self.asr_tgt_dict = tgt_dict

        self.text_encoder = text_encoder.to(self.device)
        self.roberta_src_dict = roberta_src_dict

        from fairseq.data import encoders
        self.bpe = encoders.build_bpe("gpt2")
        
        self.decoder = decoder.to(self.device)

        # import pdb; pdb.set_trace()

        # self.roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    @classmethod
    def build_model(cls, cfg: Data2VecAudioTextConfig, task: FairseqTask):
        """Build a new model instance."""

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            return emb
        
        # import pdb; pdb.set_trace()

        decoder_embed_tokens = build_embedding(tgt_dict, cfg.decoder_embed_dim)

        audio_encoder, w2v_encoder_embed_dim = cls.build_audio_encoder(cfg)
        asr_eval_decoder = cls.build_asr_eval_decoder(tgt_dict)
        # import pdb; pdb.set_trace()
        text_encoder, roberta_src_dict = cls.build_text_encoder(cfg)
        decoder = cls.build_decoder(cfg, w2v_encoder_embed_dim, tgt_dict, decoder_embed_tokens)
        
        # import pdb; pdb.set_trace()

        return Data2VecAudioTextModel(audio_encoder, asr_eval_decoder, tgt_dict, text_encoder, roberta_src_dict, decoder)


    @classmethod
    def build_audio_encoder(cls, cfg):

        # state = checkpoint_utils.load_checkpoint_to_cpu(cfg.w2v_ctc_path)
        # w2v_args = state.get("cfg", None)
        # cfg.w2v_args = w2v_args
        # import pdb; pdb.set_trace()
        # task = tasks.setup_task(state['args'])
        # model = task.build_model(state['args'], from_checkpoint=True)
        # model.remove_pretraining_modules()
        # import pdb; pdb.set_trace()

        # w2v_encoder = Wav2VecEncoder(cfg, len(task.target_dictionary))
        # import pdb; pdb.set_trace()

        overrides = {
            "criterion": 'ctc',
            "data": '/workspace/librispeech_model/am/fairseq_audio_data2', 
            "post_process": 'letter', 
            "scoring": 'wer', 
            "task": 'audio_finetuning'
        }

        # overrides = {
        #     "dropout": cfg.dropout,
        #     "activation_dropout": cfg.activation_dropout,
        #     "dropout_input": cfg.dropout_input,
        #     "attention_dropout": cfg.attention_dropout,
        #     "mask_length": cfg.mask_length,
        #     "mask_prob": cfg.mask_prob,
        #     "require_same_masks": getattr(cfg, "require_same_masks", True),
        #     "pct_holes": getattr(cfg, "mask_dropout", 0),
        #     "mask_selection": cfg.mask_selection,
        #     "mask_other": cfg.mask_other,
        #     "no_mask_overlap": cfg.no_mask_overlap,
        #     "mask_channel_length": cfg.mask_channel_length,
        #     "mask_channel_prob": cfg.mask_channel_prob,
        #     "mask_channel_before": cfg.mask_channel_before,
        #     "mask_channel_selection": cfg.mask_channel_selection,
        #     "mask_channel_other": cfg.mask_channel_other,
        #     "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
        #     "encoder_layerdrop": cfg.layerdrop,
        #     "feature_grad_mult": cfg.feature_grad_mult,
        #     "checkpoint_activations": cfg.checkpoint_activations,
        #     "offload_activations": cfg.offload_activations,
        #     "min_params_to_wrap": cfg.min_params_to_wrap,
        #     "criterion": 'ctc',
        #     "data": '/workspace/librispeech_model/am/fairseq_audio_data2', 
        #     "post_process": 'letter', 
        #     "scoring": 'wer', 
        #     "task": 'audio_finetuning'
        # }

        # Load ensemble
        logger.info("| loading audio model from {}".format(cfg.w2v_ctc_path))
        models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            utils.split_paths(cfg.w2v_ctc_path, separator="\\"),
            arg_overrides=overrides,
            strict=True
        )
        model = models[0]

        w2v_encoder_embed_dim = saved_cfg.model.w2v_args.model.encoder_embed_dim

        # import pdb; pdb.set_trace()
            
        return model, w2v_encoder_embed_dim

    @classmethod
    def build_asr_eval_decoder(cls, tgt_dict):
        decoding = DecodingConfig()
        # return W2lViterbiDecoder(args, tgt_dict)
        return Decoder(decoding, tgt_dict)

    @classmethod
    def build_text_encoder(cls, cfg):

        overrides = {
            "task": 'language_modeling',
            "data": '/workspace/data2vec/roberta.base'
        }

        from fairseq.data import encoders

        # Load ensemble
        logger.info("| loading text model from {}".format(cfg.roberta_path))
        models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            utils.split_paths(cfg.roberta_path, separator="\\"),
            arg_overrides=overrides,
            strict=True,
            # state=state
        )
        model = models[0]

        # import pdb; pdb.set_trace()

        return model, task.source_dictionary

    @classmethod
    def build_decoder(cls, cfg: Data2VecAudioTextConfig, w2v_encoder_embed_dim, tgt_dict, embed_tokens):
        # return TransformerDecoder(cfg, tgt_dict, embed_tokens)
        # return FairseqNATDecoder(cfg, tgt_dict, embed_tokens)
        return TransformerDecoderBase(cfg, w2v_encoder_embed_dim, tgt_dict, embed_tokens)

    def forward(self, **kwargs):
        res = self.audio_encoder.w2v_encoder.w2v_model.extract_features(**kwargs)
        # import pdb; pdb.set_trace()
        x, padding_mask = res['x'], res['padding_mask']
        acoustic_representation = x

        # B x T x C -> T x B x C

        x = x.transpose(0, 1)
        x = self.audio_encoder.w2v_encoder.final_dropout(x) # torch.Size([740, 1, 1024])
        ctc_emissions = self.audio_encoder.w2v_encoder.proj(x)

        ctc_emissions = self.audio_encoder.w2v_encoder.w2v_model.get_normalized_probs(ctc_emissions,log_probs=True)


        ctc_emissions = ctc_emissions.transpose(0, 1) # torch.Size([1, 740, 32])

        ctc_output = self.asr_eval_decoder.decode(ctc_emissions)

        ctc_output = [
            self.process_sentence(ctc_output[b][0])
            for b in range(ctc_emissions.size(0))
        ]


        # # greedy ctc outputs
        # ctc_probs, ctc_ids = torch.exp(ctc_emissions).max(dim=-1)
        # y_hat = torch.stack([x[0] for x in groupby(ctc_ids[0])])
        # y_idx = torch.nonzero(y_hat != 0).squeeze(-1)

        # import pdb; pdb.set_trace()

        # trellis = get_trellis(ctc_emissions, tokens)

        # path = backtrack(trellis, emission, tokens)
        # print(path)

        # segments = merge_repeats(path)
        # for seg in segments:
        #     print(seg)

        # word_segments = merge_words(segments)
        # for word in word_segments:
        #     print(word)

        # self.roberta_tokenizer(ctc_output) # huggingface roberta tokenizer
        roberta_input, _, _ = self.encode_batch_sent_for_roberta(ctc_output) # fairseq roberta tokenizer
        roberta_input = roberta_input.to(self.device)

        # import pdb; pdb.set_trace()

        text_representation = self.text_encoder(roberta_input, features_only=True)[0] # torch.Size([1, 44, 768])

        # import pdb; pdb.set_trace()

        decoder_out, _ = self.decoder(prev_output_tokens = text_representation, encoder_out = x)
        # encoder-side attention, should be of size T x B x C

        import pdb; pdb.set_trace()

        return decoder_out

    def process_sentence(self, hypo):
        # Processes hypothesis.
        # hyp_pieces = self.tgt_dict.string(hypo["tokens"].int().cpu())
        hyp_pieces = self.asr_tgt_dict.string(hypo["tokens"].int())
        if "words" in hypo:
            hyp_words = " ".join(hypo["words"])
        else:
            # hyp_words = post_process(hyp_pieces, self.cfg.common_eval.post_process)
            hyp_words = post_process(hyp_pieces, 'letter')
        return hyp_words.lower()

    def encode_batch_sent_for_roberta(self, sents, *addl_sentences, no_separator=False):

        tmp = []
        for sent in sents :
            # import pdb; pdb.set_trace()
            # sent = "<s> " + sent[0].upper() + sent[1:] + " </s>"
            # tokens = self.roberta_src_dict.encode_line(sent, append_eos=False, add_if_not_exist=False)

            bpe_sentence = "<s> " + self.bpe.encode(sent[0].upper() + sent[1:]) + " </s>"
            # for s in addl_sentences:
            #     bpe_sentence += " </s>" if not no_separator else ""
            #     bpe_sentence += " " + self.bpe.encode(s) + " </s>"
            tokens = self.roberta_src_dict.encode_line(bpe_sentence, append_eos=False, add_if_not_exist=False)

            # import pdb; pdb.set_trace()

            '''
            (Pdb) tokens
            tensor([    0, 32251,    56,    57,  3518,     7,    28,    14,     9,   475,
                    4742,  4410,   642,  3121,    18, 21023,  1354,    54,    56,   962,
                    23, 32382,     8,     9,    10, 16747,  2199,     5,  5442,    58,
                    27922,     8,   818,    11,  5247,  4748,     7,    41, 23089, 15664,
                    1720,  5567,   150,   167,     9,    39,  1150,    58,   933,     8,
                    11693,     2], dtype=torch.int32)
                    
            (Pdb) self.roberta_src_dict.string(tokens)
            '13828 550 587 4385 284 307 326 286 285 1694 1357 79 1706 338 34331 4957 508 550 3724 379 27913 290 286 257 28528 
            4369 262 6846 547 18107 290 2048 287 3885 856 284 281 8593 974 1417 1027 981 883 286 465 2988 547 4081 290 7310'

            (Pdb) self.bpe.decode(bpe_sentence[3:len(bpe_sentence)-4])
            "Which had been supposed to be that of mister dempster's eldest daughter who had died at sixteen and of a lingering disease 
            the latter were faint and almost inaudible to an unpractised ear while those of his father were firm and distinct"
            '''

            tmp.append(tokens)

        # import pdb; pdb.set_trace()

        batch, batch_mask, length = batch_pad(tmp, self.roberta_src_dict.pad_index)

        return batch, batch_mask, length

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    def max_positions(self):
        """Maximum length supported by the model."""
        return (self.audio_encoder.max_positions(), self.decoder.max_positions())

    def max_decoder_positions(self):
        """Maximum length supported by the decoder."""
        return self.decoder.max_positions()


def pad(input, max_len, pad_id):
    pad_needed = max_len - len(input) 
    assert pad_needed >= 0

    # always assumes right padding

    if type(input) == torch.Tensor:
        padded = torch.cat((input, torch.Tensor([pad_id]).expand(pad_needed)))
    else:
        padded = input + torch.LongTensor([pad_id]) * pad_needed
    mask = [1] * len(input) + [0] * pad_needed
    return padded, mask, len(input) -1


def batch_pad(input, pad_id):
    max_len = max(len(t) for t in input) 
    padded = []
    mask = []
    length = []

    for i in input:
        p, m, l = pad(i, max_len, pad_id)
        padded.append(p)
        mask.append(m)
        length.append(l)
    
    # import pdb; pdb.set_trace()

    return torch.stack(padded).long(), torch.LongTensor(mask), torch.LongTensor(length)

# rewrite name for backward compatibility in `make_generation_fast_`
def module_name_fordropout(module_name: str) -> str:
    if module_name == "TransformerDecoderBase":
        return "TransformerDecoder"
    else:
        return module_name


class TransformerDecoderBase(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *cfg.decoder.layers* layers. Each layer
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
        cfg,
        encoder_output_dim,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        self.cfg = TransformerConfig.from_namespace(cfg)
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self._future_mask = torch.empty(0)

        self.dropout_module = FairseqDropout(
            self.cfg.dropout, module_name=module_name_fordropout(self.__class__.__name__)
        )
        self.decoder_layerdrop = self.cfg.decoder.layerdrop
        self.share_input_output_embed = self.cfg.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim # 768
        embed_dim = self.cfg.decoder.embed_dim # 768
        self.embed_dim = embed_dim
        self.output_embed_dim = self.cfg.decoder.output_dim

        self.padding_idx = embed_tokens.padding_idx # 1
        self.max_target_positions = self.cfg.max_target_positions

        self.encoder_output_proj = (
            Linear(encoder_output_dim, embed_dim, bias=False)
            if encoder_output_dim != embed_dim
            else None
        )            

        # import pdb; pdb.set_trace()

        # self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if self.cfg.no_scale_embedding else math.sqrt(embed_dim)

        if not self.cfg.adaptive_input and self.cfg.quant_noise.pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                self.cfg.quant_noise.pq,
                self.cfg.quant_noise.pq_block_size,
            )
        else:
            self.quant_noise = None

        self.project_in_dim = (
            Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )

        self.embed_positions = None
        self.layernorm_embedding = None

        # self.embed_positions = (
        #     PositionalEmbedding(
        #         self.max_target_positions,
        #         embed_dim,
        #         self.padding_idx,
        #         learned=self.cfg.decoder.learned_pos,
        #     )
        #     if not self.cfg.no_token_positional_embeddings
        #     else None
        # )

        # if self.cfg.layernorm_embedding:
        #     self.layernorm_embedding = LayerNorm(embed_dim, export=self.cfg.export)
        # else:
        #     self.layernorm_embedding = None

        self.cross_self_attention = self.cfg.cross_self_attention

        if self.decoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.decoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_decoder_layer(self.cfg, no_encoder_attn)
                for _ in range(self.cfg.decoder.layers)
            ]
        )
        self.num_layers = len(self.layers)

        if self.cfg.decoder.normalize_before and not self.cfg.no_decoder_final_norm:
            self.layer_norm = LayerNorm(embed_dim, export=self.cfg.export)
        else:
            self.layer_norm = None

        self.project_out_dim = (
            Linear(embed_dim, self.output_embed_dim, bias=False)
            if embed_dim != self.output_embed_dim and not self.cfg.tie_adaptive_weights
            else None
        )

        self.adaptive_softmax = None

        ## dont need embedding layer

        # self.output_projection = output_projection
        # if self.output_projection is None:
        #     self.build_output_projection(self.cfg, dictionary, embed_tokens)

    # def build_output_projection(self, cfg, dictionary, embed_tokens):
    #     if cfg.adaptive_softmax_cutoff is not None:
    #         self.adaptive_softmax = AdaptiveSoftmax(
    #             len(dictionary),
    #             self.output_embed_dim,
    #             utils.eval_str_list(cfg.adaptive_softmax_cutoff, type=int),
    #             dropout=cfg.adaptive_softmax_dropout,
    #             adaptive_inputs=embed_tokens if cfg.tie_adaptive_weights else None,
    #             factor=cfg.adaptive_softmax_factor,
    #             tie_proj=cfg.tie_adaptive_proj,
    #         )
    #     elif self.share_input_output_embed:
    #         self.output_projection = nn.Linear(
    #             self.embed_tokens.weight.shape[1],
    #             self.embed_tokens.weight.shape[0],
    #             bias=False,
    #         )
    #         self.output_projection.weight = self.embed_tokens.weight
    #     else:
    #         self.output_projection = nn.Linear(
    #             self.output_embed_dim, len(dictionary), bias=False
    #         )
    #         nn.init.normal_(
    #             self.output_projection.weight, mean=0, std=self.output_embed_dim**-0.5
    #         )
    #     num_base_layers = cfg.base_layers
    #     for i in range(num_base_layers):
    #         self.layers.insert(
    #             ((i + 1) * cfg.decoder.layers) // (num_base_layers + 1),
    #             BaseLayer(cfg),
    #         )

    def build_decoder_layer(self, cfg, no_encoder_attn=False):
        layer = transformer_layer.TransformerDecoderLayerBase(cfg, no_encoder_attn)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def forward(
        self,
        prev_output_tokens,
        encoder_out,
        # encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention, should be of size T x B x C
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """

        # need only cross attention
        # not PE, embedding, output projection
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )

        # if not features_only:
        #     x = self.output_layer(x)

        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out,
        # encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )

    """
    A scriptable subclass of this class has an extract_features method and calls
    super().extract_features, but super() is not supported in torchscript. A copy of
    this function is made to be used in the subclass instead.
    """

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out,
        # encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.
        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).
        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).
        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """

        # bs, slen = prev_output_tokens.size()

        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None

        # if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
        #     enc = encoder_out["encoder_out"][0]
        #     assert (
        #         enc.size()[1] == bs
        #     ), f"Expected enc.shape == (t, {bs}, c) got {enc.shape}"
        # if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
        #     padding_mask = encoder_out["encoder_padding_mask"][0]


        if self.encoder_output_proj:
            enc = self.encoder_output_proj(encoder_out)
        else:
            enc = encoder_out


        ##################################################
        ######## PE + embedding 부분 노필요 ################
        ##################################################

        # # embed positions
        # positions = None
        # if self.embed_positions is not None:
        #     positions = self.embed_positions(
        #         prev_output_tokens, incremental_state=incremental_state
        #     )

        # if incremental_state is not None:
        #     prev_output_tokens = prev_output_tokens[:, -1:]
        #     if positions is not None:
        #         positions = positions[:, -1:]

        # # embed tokens and positions
        # x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        # if self.quant_noise is not None:
        #     x = self.quant_noise(x)

        # if self.project_in_dim is not None:
        #     x = self.project_in_dim(x)

        # if positions is not None:
        #     x += positions

        # if self.layernorm_embedding is not None:
        #     x = self.layernorm_embedding(x)

        # x = self.dropout_module(x)

        x = prev_output_tokens

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        # if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
        #     self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # import pdb; pdb.set_trace()

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            # import pdb; pdb.set_trace()

            x, layer_attn, _ = layer(
                x,
                enc,
                padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )

            # import pdb; pdb.set_trace()

            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        ##
        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}

    # def output_layer(self, features):
    #     """Project features to the vocabulary size."""
    #     if self.adaptive_softmax is None:
    #         # project back to size of vocabulary
    #         return self.output_projection(features)
    #     else:
    #         return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.max_target_positions
        # if self.embed_positions is None:
        #     return self.max_target_positions
        # return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)

        if f"{name}.output_projection.weight" not in state_dict:
            if self.share_input_output_embed:
                embed_out_key = f"{name}.embed_tokens.weight"
            else:
                embed_out_key = f"{name}.embed_out"
            if embed_out_key in state_dict:
                state_dict[f"{name}.output_projection.weight"] = state_dict[
                    embed_out_key
                ]
                if not self.share_input_output_embed:
                    del state_dict[embed_out_key]

        for i in range(self.num_layers):
            # update layer norms
            layer_norm_map = {
                "0": "self_attn_layer_norm",
                "1": "encoder_attn_layer_norm",
                "2": "final_layer_norm",
            }
            for old, new in layer_norm_map.items():
                for m in ("weight", "bias"):
                    k = "{}.layers.{}.layer_norms.{}.{}".format(name, i, old, m)
                    if k in state_dict:
                        state_dict[
                            "{}.layers.{}.{}.{}".format(name, i, new, m)
                        ] = state_dict[k]
                        del state_dict[k]

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])

        return state_dict


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m



################################################################################################
########################        Implementation of forced alinger      ##########################
########################   need for token (word) - wise probability   ##########################
################################################################################################

def get_trellis(emission, tokens, blank_id=0):
  num_frame = emission.size(0)
  num_tokens = len(tokens)

  # Trellis has extra diemsions for both time axis and tokens.
  # The extra dim for tokens represents <SoS> (start-of-sentence)
  # The extra dim for time axis is for simplification of the code.
  trellis = torch.full((num_frame+1, num_tokens+1), -float('inf'))
  trellis[:, 0] = 0
  for t in range(num_frame):
    trellis[t+1, 1:] = torch.maximum(
        # Score for staying at the same token
        trellis[t, 1:] + emission[t, blank_id],
        # Score for changing to the next token
        trellis[t, :-1] + emission[t, tokens],
    )
  return trellis


@dataclass
class Point:
  token_index: int
  time_index: int
  score: float


def backtrack(trellis, emission, tokens, blank_id=0):
  # Note:
  # j and t are indices for trellis, which has extra dimensions
  # for time and tokens at the beginning.
  # When refering to time frame index `T` in trellis,
  # the corresponding index in emission is `T-1`.
  # Similarly, when refering to token index `J` in trellis,
  # the corresponding index in transcript is `J-1`.
  j = trellis.size(1) - 1
  t_start = torch.argmax(trellis[:, j]).item()

  path = []
  for t in range(t_start, 0, -1):
    # 1. Figure out if the current position was stay or change
    # Note (again):
    # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
    # Score for token staying the same from time frame J-1 to T.
    stayed = trellis[t-1, j] + emission[t-1, blank_id]
    # Score for token changing from C-1 at T-1 to J at T.
    changed = trellis[t-1, j-1] + emission[t-1, tokens[j-1]]

    # 2. Store the path with frame-wise probability.
    prob = emission[t-1, tokens[j-1] if changed > stayed else 0].exp().item()
    # Return token index and time index in non-trellis coordinate.
    path.append(Point(j-1, t-1, prob))

    # 3. Update the token
    if changed > stayed:
      j -= 1
      if j == 0:
        break
  else:
    raise ValueError('Failed to align')
  return path[::-1]


# Merge the labels
@dataclass
class Segment:
  label: str
  start: int
  end: int
  score: float

  def __repr__(self):
    return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

  @property
  def length(self):
    return self.end - self.start

def merge_repeats(path):
  i1, i2 = 0, 0
  segments = []
  while i1 < len(path):
    while i2 < len(path) and path[i1].token_index == path[i2].token_index:
      i2 += 1
    score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
    segments.append(Segment(transcript[path[i1].token_index], path[i1].time_index, path[i2-1].time_index + 1, score))
    i1 = i2
  return segments


# Merge words
def merge_words(segments, separator='|'):
  words = []
  i1, i2 = 0, 0
  while i1 < len(segments):
    if i2 >= len(segments) or segments[i2].label == separator:
      if i1 != i2:
        segs = segments[i1:i2]
        word = ''.join([seg.label for seg in segs])
        score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
        words.append(Segment(word, segments[i1].start, segments[i2-1].end, score))
      i1 = i2 + 1
      i2 = i1
    else:
      i2 += 1
  return words