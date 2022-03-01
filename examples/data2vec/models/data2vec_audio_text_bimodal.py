
import contextlib
import copy
from curses import keyname
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
    Wav2VecEncoder,
    Wav2Vec2Config
)

# from . import Data2VecAudioConfig

from itertools import groupby

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
from fairseq.data import encoders

from transformers import RobertaTokenizer


from fairseq.modules import EMAModule, EMAModuleConfig
import torch.distributed as dist
import time

# from fairseq.models.roberta.alignment_utils import align_bpe_to_words


############################################
############ bimodal data2vec model #############
############ cross attention of audio and text #############
############################################

@dataclass
# class Data2VecAudioTextConfig(Wav2Vec2Config):
class Data2VecAudioTextConfig(Wav2Vec2AsrConfig):

    ## 1. -> Wav2Vec2AsrConfig for Wav2Vec-CTC Encdoer

    ## 2. for Transformer Decoder Config
    decoder_embed_dim: int = field(
        default=768, metadata={"help": "decoder embedding dimension"}
    )
    decoder_output_dim: int = field(
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
        default=512, metadata={"help": "max target positions"}
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

    adaptive_softmax_cutoff: Optional[str] = field(
        default=None,
        metadata={
            "help": "comma separated list of adaptive softmax cutoff points. "
            "Must be used with adaptive_loss criterion"
        },
    )
    adaptive_softmax_dropout: float = field(
        default=0,
        metadata={"help": "sets adaptive softmax dropout for the tail projections"},
    )
    adaptive_softmax_factor: float = field(
        default=4, metadata={"help": "adaptive input factor"}
    )

    ### 3. for EMA Training configs

    # l1 smooth loss의 하이퍼 파라메터
    loss_beta: float = field(
        default=0, metadata={"help": "beta for smooth l1 loss. 0 means use l2 loss"}
    )
    loss_scale: Optional[float] = field(
        default=None,
        metadata={
            "help": "scale the reconstruction loss by this constant. if None then scales by 1/sqrt(dim)"
        },
    )

    # 상위 k 개 레이어의 output을 averaging해서 타겟으로 사용함. 
    average_top_k_layers: int = field(
        default=4, metadata={"help": "how many layers to average"}
    )

    layer_norm_target_layer: bool = False
    instance_norm_target_layer: bool = False
    instance_norm_targets: bool = False
    layer_norm_targets: bool = False
    batch_norm_target_layer: bool = False
    group_norm_target_layer: bool = False

    # EMA decay 하이퍼 파라메터
    ema_decay: float = field(default=0.999, metadata={"help": "initial ema decay rate"})
    ema_end_decay: float = field(
        default=0.9999, metadata={"help": "final ema decay rate"}
    )

    # when to finish annealing ema decay rate
    ema_anneal_end_step: int = II("optimization.max_update")

    ema_transformer_only: bool = field(
        default=True,
        metadata={"help": "whether to momentum update only the transformer"},
    )
    ema_layers_only: bool = field(
        default=True,
        metadata={"help": "whether to momentum update only the transformer layers"},
    )

    max_update: int = II("optimization.max_update")

    min_target_var: float = field(
        default=0.1, metadata={"help": "stop training if target var falls below this"}
    )
    min_pred_var: float = field(
        default=0.01,
        metadata={"help": "stop training if prediction var falls below this"},
    )

    ema_pretraining: bool = field(
        default=False,
        metadata={"help": "tmp"},
    )

    head_layers: int = 2

    # load_checkpoint_heads: bool = field(
    #     default=False,
    #     metadata={"help": "(re-)register and load heads when loading checkpoints"},
    # )

    # ema_transformer_layers_only: bool = field(
    #     default=True,
    #     metadata={"help": "whether to momentum update only the transformer layers"},
    # )


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


def get_annealed_rate(start, end, curr_step, total_steps):
    r = end - start
    pct_remaining = 1 - curr_step / total_steps
    return end - r * pct_remaining


def freeze_module_params(m):
    if m is not None:
        for p in m.parameters():
            p.requires_grad = False


def check_model_freezed(m):
    if m is not None:
        for n, p in m.named_parameters():
            print(n,p.requires_grad)


@register_model("data2vec_bimodal", dataclass=Data2VecAudioTextConfig)
class Data2VecAudioTextModel(BaseFairseqModel):
    def __init__(self, cfg: Data2VecAudioTextConfig, audio_encoder_ctc, asr_eval_decoder, asr_embedding_layer, asr_tgt_dict, text_encoder, roberta_src_dict, decoder):
        super().__init__()

        self.cfg = cfg

        ## Wav2Vec2 Encoder
        self.device = torch.device("cuda")
        self.audio_encoder_ctc = audio_encoder_ctc.to(self.device)
        # self.audio_encoder = self.audio_encoder_ctc.w2v_encoder.w2v_model.to(self.device)
        # self.device = next(self.audio_encoder.parameters()).device
        self.asr_eval_decoder = asr_eval_decoder
        self.asr_embedding_layer = asr_embedding_layer
        self.asr_tgt_dict = asr_tgt_dict
        self.mask_token = self.asr_tgt_dict.bos_index

        ## RoBERTa Encoder
        self.text_encoder = text_encoder.to(self.device)
        self.roberta_src_dict = roberta_src_dict
        self.roberta_mask_idx = self.roberta_src_dict.index("<mask>")
        assert self.roberta_mask_idx != self.roberta_src_dict.unk(), self.roberta_src_dict.symbols
        self.roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.bpe = encoders.build_bpe("gpt2")
        
        ## Transformer Decoder for bimodal training
        self.decoder = decoder.to(self.device)

        ## ema training
        self.ema = None

        self.average_top_k_layers = cfg.average_top_k_layers # 8

        self.loss_beta = cfg.loss_beta # 0.0
        self.loss_scale = cfg.loss_scale # None

        self.mask_prob = cfg.mask_prob # 0.65
        self.mask_selection = cfg.mask_selection # static
        self.mask_other = cfg.mask_other # 0.0
        self.mask_length = cfg.mask_length # 10
        self.no_mask_overlap = cfg.no_mask_overlap # False
        self.mask_min_space = cfg.mask_min_space # 1

        self.mask_channel_prob = cfg.mask_channel_prob # 0.0
        self.mask_channel_before = cfg.mask_channel_before # False
        self.mask_channel_selection = cfg.mask_channel_selection # static
        self.mask_channel_other = cfg.mask_channel_other # 0.0
        self.mask_channel_length = cfg.mask_channel_length # 10
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap # False
        self.mask_channel_min_space = cfg.mask_channel_min_space # 1

        self.num_updates = 0

        # self.final_proj = nn.Linear(self.cfg.decoder_embed_dim, self.cfg.decoder_embed_dim) 

        self.ema_pretraining = cfg.ema_pretraining

        embed_dim = self.cfg.decoder_embed_dim
        curr_dim = embed_dim
        projs = []
        for i in range(self.cfg.head_layers - 1):
            next_dim = embed_dim * 2 if i == 0 else curr_dim
            projs.append(nn.Linear(curr_dim, next_dim))
            projs.append(nn.GELU())
            curr_dim = next_dim

        projs.append(nn.Linear(curr_dim, embed_dim))
        self.regression_head = nn.Sequential(*projs)

        if self.ema_pretraining:
            pass
        else:
            logger.info("===============================================================")
            logger.info("| EMA Pretraining mode? {}".format(self.ema_pretraining))
            logger.info("| freezing audio encoder and text encoder...")
            freeze_module_params(self.audio_encoder_ctc)
            freeze_module_params(self.text_encoder)
            logger.info("| Done !")
            logger.info("===============================================================")
        

    def make_ema_teacher(self):
        ema_config = EMAModuleConfig(
            ema_decay=self.cfg.ema_decay, # 0.999 default
            ema_fp32=True,
        )

        skip_keys = set()

        for k, v in self.decoder.named_parameters():
            # print('name:',k)
            if 'layers' not in k :
                # skip_keys.add(k.split('.weight')[0])
                skip_keys.add(k)
            else:
                if 'norm' in k:
                    skip_keys.add(k)

        self.ema = EMAModule(
            self.decoder,
            ema_config,
            skip_keys=skip_keys
        )

        # self.ema = EMAModule(
        #     self,
        #     ema_config,
        #     skip_keys=skip_keys,
        # )

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)

        # import pdb; pdb.set_trace()

        if self.ema_pretraining:
            # if self.ema is None and self.final_proj is not None:
            #     logger.info(f"making ema teacher")
            #     self.make_ema_teacher()
            if self.ema is None and self.regression_head is not None:
                logger.info(f"making ema teacher")
                self.make_ema_teacher()
            elif self.training and self.ema is not None:
                if self.cfg.ema_decay != self.cfg.ema_end_decay:
                    if num_updates >= self.cfg.ema_anneal_end_step:
                        decay = self.cfg.ema_end_decay
                    else:
                        decay = get_annealed_rate(
                            self.cfg.ema_decay,
                            self.cfg.ema_end_decay,
                            num_updates,
                            self.cfg.ema_anneal_end_step,
                        )
                    self.ema.set_decay(decay)
                if self.ema.get_decay() < 1:
                    # self.ema.step(self.encoder if self.cfg.ema_transformer_only else self)
                    self.ema.step(self.decoder)
        else:
            pass

        self.num_updates = num_updates

    def remove_pretraining_modules(self, last_layer=None):
        self.regression_head = None
        self.ema = None


        # if last_layer is not None:
        #     self.encoder.sentence_encoder.layers = nn.ModuleList(
        #         l
        #         for i, l in enumerate(self.encoder.sentence_encoder.layers)
        #         if i <= last_layer
        #     )
        #     self.encoder.sentence_encoder.layer_norm = None

    @classmethod
    def build_model(cls, cfg: Data2VecAudioTextConfig, task: FairseqTask):
        """Build a new model instance."""

        # import pdb; pdb.set_trace()

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            return emb

        ## 1. Wav2Vec2 Encoder
        audio_encoder_ctc, w2v_encoder_embed_dim, asr_tgt_dict = cls.build_audio_encoder(cfg)
        asr_eval_decoder = cls.build_asr_eval_decoder(asr_tgt_dict)
        asr_embedding_layer = build_embedding(asr_tgt_dict, cfg.decoder_embed_dim)

        ## 2. RoBERTa Encoder
        text_encoder, roberta_src_dict = cls.build_text_encoder(cfg)

        # import pdb; pdb.set_trace()

        ## 3. Transformer Decoder for bimodal training
        decoder = cls.build_decoder(cfg, w2v_encoder_embed_dim, tgt_dict, text_encoder.encoder.sentence_encoder.embed_tokens)
        # decoder = cls.build_decoder(cfg, len(asr_tgt_dict), tgt_dict, text_encoder.encoder.sentence_encoder.embed_tokens)

        return Data2VecAudioTextModel(cfg, audio_encoder_ctc, asr_eval_decoder, asr_embedding_layer, asr_tgt_dict, text_encoder, roberta_src_dict, decoder)


    @classmethod
    def build_audio_encoder(cls, cfg):
        overrides = {
            "criterion": 'ctc',
            "data": '/workspace/data2vec/only_dict_for_w2v2', 
            "post_process": 'letter', 
            "scoring": 'wer', 
            "task": 'audio_finetuning'
        }
        logger.info("| loading audio model from {}".format(cfg.w2v_ctc_path))
        models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            utils.split_paths(cfg.w2v_ctc_path, separator="\\"),
            arg_overrides=overrides,
            strict=True
        )
        model = models[0]
        w2v_encoder_embed_dim = saved_cfg.model.w2v_args.model.encoder_embed_dim
        return model, w2v_encoder_embed_dim, task.target_dictionary

    @classmethod
    def build_asr_eval_decoder(cls, tgt_dict):
        decoding = DecodingConfig()
        return Decoder(decoding, tgt_dict)

    @classmethod
    def build_text_encoder(cls, cfg):
        overrides = {
            "task": 'language_modeling',
            "data": '/workspace/data2vec/roberta.base'
        }
        logger.info("| loading text model from {}".format(cfg.roberta_path))
        models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            utils.split_paths(cfg.roberta_path, separator="\\"),
            arg_overrides=overrides,
            strict=True,
            # state=state
        )
        model = models[0]
        return model, task.source_dictionary

    def get_greedy_decoding_results(self, ctc_emissions):
        """
        Args:
            ctc_emissions, 

        Returns:
            greedy decoding results
        """
        transcripts = self.asr_eval_decoder.decode(ctc_emissions)
        # <examples.speech_recognition.new.decoders.viterbi_decoder.ViterbiDecoder object at 0x7f793c32d520>
        transcripts = [
            self.process_sentence(transcripts[b][0])
            for b in range(ctc_emissions.size(0))
        ]

        # import pdb; pdb.set_trace()
        return transcripts


    def get_ctc_emissions(self, input):
        """
        Args:
            wav_input, 
            target 

        Returns:
            w2v2 cnn outputs,
            ctc_emissions,
            greedy decoding results
        """

        # res = self.audio_encoder_ctc.w2v_encoder.w2v_model.extract_features(**input)
        cnn_outputs = self.audio_encoder_ctc.w2v_encoder.w2v_model.extract_features(**input, cnn_features_only=True, mask=False)
        x, layer_results = self.audio_encoder_ctc.w2v_encoder.w2v_model.encoder(cnn_outputs['x'],cnn_outputs['padding_mask'])

        # import pdb; pdb.set_trace()

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        x = self.audio_encoder_ctc.w2v_encoder.final_dropout(x) # torch.Size([740, 1, 1024])
        ctc_emissions = self.audio_encoder_ctc.w2v_encoder.proj(x) # torch.Size([1, 740, 32])

        w2v2_encoder_results = {
            "encoder_out": x,
            "padding_mask": cnn_outputs['padding_mask'],
            "layer_results": layer_results,
        }

        ctc_emissions = self.audio_encoder_ctc.w2v_encoder.w2v_model.get_normalized_probs(ctc_emissions,log_probs=True)
        ctc_emissions = ctc_emissions.transpose(0, 1) # torch.Size([1, 740, 32])

        # import pdb; pdb.set_trace()

        return cnn_outputs, ctc_emissions, w2v2_encoder_results

    def get_mask_indices_for_audio_and_text(self, ctc_emissions, transcripts, cnn_outputs):
        """
        Args:
            ctc_emissions, 
            target,
            cnn_outputs

        Returns:
            mask indices for roberta
            mask indices for cnn outputs
            greedy decoding results (transcripts)
        """

        mask_indices = []
        cnn_output_mask_indices = []

        for idx, (emission, transcript, cnn_output) in enumerate(zip(ctc_emissions, transcripts, cnn_outputs)):
            transcript = transcript.replace(' ','|') + ('|')
            tokens = [self.asr_tgt_dict.indices[c] for c in transcript]
            emission = emission.cpu()
            sent = transcript
            trellis = get_trellis(emission, tokens)
            path = backtrack(trellis, emission, tokens)
            segments = merge_repeats(path, sent)
            word_segments = merge_words(segments)

            # mask_indice = []
            # for seg in segments:
            #     if seg.score <0.95:
            #         mask_indice.append(self.mask_token)
            #     else:
            #         mask_indice.append(self.asr_tgt_dict.index(seg.label))

            mask_indice = []
            dummy=10
            cnn_output_mask = torch.zeros(cnn_output.size(0))
            for seg in word_segments:
                if seg.score <0.8:
                    mask_indice += [0]*len(seg.label)
                    cnn_output_mask[seg.start:seg.end]=1
                else:
                    mask_indice += [dummy]*len(seg.label)
                mask_indice.append(0)
            mask_indices.append(mask_indice)
            cnn_output_mask_indices.append(cnn_output_mask)

        # import pdb; pdb.set_trace()

        cnn_output_mask_indices = (torch.stack(cnn_output_mask_indices).to(self.device)==1)
        return mask_indices, cnn_output_mask_indices

    def apply_mask_for_audio(self, cnn_outputs, cnn_output_mask_indices):
        masked_cnn_output, padding_mask = self.audio_encoder_ctc.w2v_encoder.w2v_model.apply_mask(cnn_outputs['x'],cnn_outputs['padding_mask'],mask_indices=cnn_output_mask_indices)
        return masked_cnn_output, padding_mask

    def apply_mask_for_text(self, transcripts, roberta_input, roberta_input_length, mask_indices):
        post_processed_transcript = self.lower_and_punc(transcripts)
        bpe_mask = torch.zeros_like(roberta_input, dtype=torch.long)

        for idx, (bpe_sent, bpe_length, sent) in enumerate(zip(roberta_input, roberta_input_length, post_processed_transcript)):
            bpe_sent = bpe_sent[:bpe_length]
            alignment = align_bpe_to_words(self.roberta_src_dict, self.bpe, bpe_sent, sent)
            alignment_mask = (torch.Tensor(mask_indices[idx]) == 0)
            # import pdb; pdb.set_trace()
            masked_alignment = torch.LongTensor(alignment).masked_fill(alignment_mask,0)
            _ = torch.stack([x[0] for x in groupby(masked_alignment)])
            confident_idx = torch.unique(_)[1:]
            bpe_mask[idx][confident_idx]=1

        roberta_input = roberta_input.masked_fill(torch.logical_not(bpe_mask),self.roberta_mask_idx).to(self.device)

        return roberta_input


    @classmethod
    def build_decoder(cls, cfg: Data2VecAudioTextConfig, w2v_encoder_embed_dim, tgt_dict, embed_tokens):
        # return TransformerDecoder(cfg, tgt_dict, embed_tokens)
        # return FairseqNATDecoder(cfg, tgt_dict, embed_tokens)
        # return TransformerDecoderBase(cfg, w2v_encoder_embed_dim, tgt_dict, embed_tokens)
        return TransformerDecoderBase(cfg, w2v_encoder_embed_dim, tgt_dict, embed_tokens)


    def forward(self, sample):

        target = sample['target']
        target_padding_mask = (target==self.roberta_src_dict.pad())
        target_bos_marker=(sample['target']==self.roberta_src_dict.bos())
        target_eos_marker=(sample['target']==self.roberta_src_dict.eos())
        target_len = sample['target_lengths']
        input = sample['net_input']

        if self.ema is None:
            ##############################################################################################################
            ###### for ASR finetuning, we hv to use gold transcription because of compute loss non-autoregressively ######
            ##############################################################################################################
            cnn_outputs, ctc_emissions, w2v2_encoder_results = self.get_ctc_emissions(input)
            transcripts = None
            if self.training:
                # transcripts = self.get_greedy_decoding_results(ctc_emissions)
                uniform_mask = torch.FloatTensor(target.size()).uniform_() > 0.65
                target = target.masked_fill(uniform_mask.to(self.device),self.roberta_mask_idx)
                target = target.masked_fill(target_padding_mask,self.roberta_src_dict.pad())
                target = target.masked_fill(target_bos_marker,self.roberta_src_dict.bos()).masked_fill(target_eos_marker,self.roberta_src_dict.eos()) 
                
                roberta_output = self.text_encoder(target.masked_fill(uniform_mask.to(self.device),self.roberta_mask_idx), features_only=True)[0]
                decoder_out, _ = self.decoder(prev_output_tokens = roberta_output, encoder_out = w2v2_encoder_results, decoder_input_mask=target_padding_mask)
                # import pdb; pdb.set_trace()
            else:
                # get ctc output and make bpe roberta sentence
                transcripts = self.get_greedy_decoding_results(ctc_emissions)
                roberta_input, roberta_input_mask, roberta_input_length = self.encode_batch_sent_for_roberta(transcripts)

                # padding mask for roberta input
                mask_indices, _ = self.get_mask_indices_for_audio_and_text(ctc_emissions, transcripts, cnn_outputs['x'])
                target_padding_mask = torch.logical_not(roberta_input_mask).to(self.device)
                target_bos_marker=(roberta_input==self.roberta_src_dict.bos()).to(self.device)
                target_eos_marker=(roberta_input==self.roberta_src_dict.eos()).to(self.device)

                masked_roberta_input = self.apply_mask_for_text(transcripts, roberta_input, roberta_input_length, mask_indices)
                masked_roberta_input = masked_roberta_input.masked_fill(target_padding_mask,self.roberta_src_dict.pad())
                masked_roberta_input = masked_roberta_input.masked_fill(target_bos_marker,self.roberta_src_dict.bos()).masked_fill(target_eos_marker,self.roberta_src_dict.eos()) 

                num_iter = 10
                for i in range(num_iter):
                    roberta_output = self.text_encoder(masked_roberta_input, features_only=True)[0]
                    decoder_out, _ = self.decoder(prev_output_tokens = roberta_output, encoder_out = w2v2_encoder_results, decoder_input_mask=torch.logical_not(roberta_input_mask).to(self.device))
                    prob, pred=F.softmax(decoder_out,dim=-1).max(dim=-1)
                    # import pdb; pdb.set_trace()
                    logger.info("| {} th average prob of decoder outs are {}".format(i, prob.masked_fill(target_padding_mask,0).sum()/torch.sum(roberta_input_length)))
                    if i==(num_iter-1):
                        break;
                    confident_idx = prob > 0.7
                    pred = pred.masked_fill(torch.logical_not(confident_idx),self.roberta_mask_idx).to(self.device)
                    masked_roberta_input = pred.masked_fill(target_padding_mask,self.roberta_src_dict.pad())
                    masked_roberta_input = masked_roberta_input.masked_fill(target_bos_marker,self.roberta_src_dict.bos()).masked_fill(target_eos_marker,self.roberta_src_dict.eos()) 
        else:
            ############################################################
            ###### for EMA training, we need to use masked inputs ######
            ############################################################

            # 1. Get CTC emission and greedy output
            cnn_outputs, ctc_emissions, w2v2_encoder_results = self.get_ctc_emissions(input)
            transcripts = self.get_greedy_decoding_results(ctc_emissions)

            # 2. Get BPE roberta sentence input
            roberta_input, roberta_input_mask, roberta_input_length = self.encode_batch_sent_for_roberta(transcripts)

            # 3. Get Masked indices for Roberta Inputs (text)
            mask_indices, cnn_output_mask_indices = self.get_mask_indices_for_audio_and_text(ctc_emissions, transcripts, cnn_outputs['x'])
            target_padding_mask = torch.logical_not(roberta_input_mask).to(self.device)
            target_bos_marker=(roberta_input==self.roberta_src_dict.bos()).to(self.device)
            target_eos_marker=(roberta_input==self.roberta_src_dict.eos()).to(self.device)

            masked_roberta_input = self.apply_mask_for_text(transcripts, roberta_input, roberta_input_length, mask_indices)
            masked_roberta_input = masked_roberta_input.masked_fill(target_padding_mask,self.roberta_src_dict.pad())
            masked_roberta_input = masked_roberta_input.masked_fill(target_bos_marker,self.roberta_src_dict.bos()).masked_fill(target_eos_marker,self.roberta_src_dict.eos()) 

            # 4. Get Masked Inputs for w2v2 (audio)
            masked_cnn_output, padding_mask = self.apply_mask_for_audio(cnn_outputs, cnn_output_mask_indices)

            # 5. feeding masked inputs to w2v2 and roberta each
            x_audio, layer_results = self.audio_encoder_ctc.w2v_encoder.w2v_model.encoder(masked_cnn_output, padding_mask)
            x_audio = x_audio.transpose(0, 1)
            x_audio = self.audio_encoder_ctc.w2v_encoder.final_dropout(x_audio)
            masked_w2v2_encoder_results = {
                "encoder_out": x_audio,
                "padding_mask": cnn_outputs['padding_mask'],
                "layer_results": layer_results,
            }
            masked_roberta_output = self.text_encoder(masked_roberta_input, features_only=True)[0]
            x, _ = self.decoder(prev_output_tokens = masked_roberta_output, encoder_out = masked_w2v2_encoder_results, decoder_input_mask=target_padding_mask, features_only=True)


            # 5. EMA teacher inference
            with torch.no_grad():
                # use EMA parameter as the teacher
                self.ema.model.eval()

                decoder_out = self.ema.model( # just decoder
                    prev_output_tokens = masked_roberta_output,
                    encoder_out = masked_w2v2_encoder_results,
                    return_all_hiddens=True,
                    features_only=True,
                )

                '''
                (Pdb) self.decoder.project_out_dim
                Linear(in_features=768, out_features=512, bias=False)
                (Pdb) decoder_out[0].size()
                torch.Size([1, 23, 512])
                (Pdb) len(decoder_out[-1]['inner_states'])
                7
                (Pdb) decoder_out[-1]['inner_states'][-1].size()
                torch.Size([23, 1, 768])
                '''

                import pdb; pdb.set_trace()

                # y = decoder_out["fc_results"]
                y = decoder_out[-1]['inner_states']

                y = y[-self.average_top_k_layers :]

                # target normalizer
                # 1. instance norm -> self.cfg.instance_norm_target_layer
                # 2. batch norm -> self.cfg.batch_norm_target_layer
                # 3. layer norm -> self.cfg.layer_norm_target_layer

                # self.cfg.layer_norm_targets
                # self.cfg.instance_norm_targets

                permuted = False
                if self.cfg.instance_norm_target_layer or self.cfg.batch_norm_target_layer:
                    y = [tl.permute(1, 2, 0) for tl in y]  # TBC -> BCT
                    permuted = True

                if self.cfg.batch_norm_target_layer:
                    y = [
                        F.batch_norm(
                            tl.float(), running_mean=None, running_var=None, training=True
                        )
                        for tl in y
                    ]

                if self.cfg.instance_norm_target_layer:
                    y = [F.instance_norm(tl.float()) for tl in y]

                if permuted:
                    y = [tl.transpose(1, 2) for tl in y]  # BCT -> BTC

                if self.cfg.layer_norm_target_layer:
                    y = [F.layer_norm(tl.float(), tl.shape[-1:]) for tl in y]

                y = sum(y) / len(y)

                if not permuted:
                    y = y.transpose(0, 1)

                if self.cfg.layer_norm_targets:
                    y = F.layer_norm(y.float(), y.shape[-1:])

                if self.cfg.instance_norm_targets:
                    y = F.instance_norm(y.transpose(1, 2)).transpose(1, 2)

            masked_indices = masked_roberta_input.eq(self.roberta_mask_idx)

            # x is decoder representation outputs from masked inputs
            # y is decoder representation outputs from non-masked inputs
            x = x[masked_indices]
            y = y[masked_indices]

            x = self.regression_head(x)

            sz = x.size(-1)
            if self.cfg.loss_beta == 0:
                loss = F.mse_loss(x.float(), y.float(), reduction="none").sum(dim=-1)
            else:
                loss = F.smooth_l1_loss(
                    x.float(), y.float(), reduction="none", beta=self.cfg.loss_beta
                ).sum(dim=-1)

            result = {
                "losses": {
                    "main": loss.sum() / math.sqrt(sz)
                    if self.loss_scale <= 0
                    else loss.sum() * self.loss_scale,
                },
                "sample_size": loss.numel(),
            }

            # logging other values
            other_logs = {
                "ema_decay": self.ema.get_decay() * 1000
            }
            result["logs"] = other_logs
            return result

        return decoder_out, transcripts, target_padding_mask


    @staticmethod
    def compute_var(y):
        y = y.view(-1, y.size(-1))
        if dist.is_initialized():
            zc = torch.tensor(y.size(0)).cuda()
            zs = y.sum(dim=0)
            zss = (y ** 2).sum(dim=0)

            dist.all_reduce(zc)
            dist.all_reduce(zs)
            dist.all_reduce(zss)

            var = zss / (zc - 1) - (zs ** 2) / (zc * (zc - 1))
            return torch.sqrt(var + 1e-6).mean()
        else:
            return torch.sqrt(y.var(dim=0) + 1e-6).mean()


    def get_targets(self, sample, net_output):
        return sample['target'].long()


    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output.float()
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)


    def process_sentence(self, hypo):
        # Processes hypothesis.
        # hyp_pieces = self.tgt_dict.string(hypo["tokens"].int().cpu())
        hyp_pieces = self.asr_tgt_dict.string(hypo["tokens"].int())
        if "words" in hypo:
            hyp_words = " ".join(hypo["words"])
        else:
            # hyp_words = post_process(hyp_pieces, self.cfg.common_eval.post_process)
            hyp_words = post_process(hyp_pieces, 'letter')
        # import pdb; pdb.set_trace()
        hyp_words = re.sub(" +", " ", hyp_words).strip()
        # output = hyp_words.lower()
        output = hyp_words
        return output

    def lower_and_punc(self, sents):
        return [ (sent[0].upper() + sent[1:].lower() + '.') for sent in sents]

    def encode_batch_sent_for_roberta(self, sents, *addl_sentences, no_separator=False):

        sents = self.lower_and_punc(sents)

        tmp = []
        for sent in sents :
            # import pdb; pdb.set_trace()
            # sent = "<s> " + sent[0].upper() + sent[1:] + " </s>"
            # tokens = self.roberta_src_dict.encode_line(sent, append_eos=False, add_if_not_exist=False)

            bpe_sentence = "<s> " + self.bpe.encode(sent) + " </s>"
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
        return (self.audio_encoder_ctc.max_positions(), self.decoder.max_positions())

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
    return padded, mask, len(input)


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

        self.embed_tokens = embed_tokens
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

        # import pdb; pdb.set_trace()

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

        # output projection layer
        self.adaptive_softmax = None
        self.output_projection = output_projection
        if self.output_projection is None:
            self.build_output_projection(self.cfg, dictionary, embed_tokens)


    def build_output_projection(self, cfg, dictionary, embed_tokens):
        if cfg.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                utils.eval_str_list(cfg.adaptive_softmax_cutoff, type=int),
                dropout=cfg.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if cfg.tie_adaptive_weights else None,
                factor=cfg.adaptive_softmax_factor,
                tie_proj=cfg.tie_adaptive_proj,
            )
        elif self.share_input_output_embed:
            self.output_projection = nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            self.output_projection.weight = self.embed_tokens.weight
        else:
            self.output_projection = nn.Linear(
                self.output_embed_dim, len(dictionary), bias=False
            )
            nn.init.normal_(
                self.output_projection.weight, mean=0, std=self.output_embed_dim**-0.5
            )
        num_base_layers = cfg.base_layers
        for i in range(num_base_layers):
            self.layers.insert(
                ((i + 1) * cfg.decoder.layers) // (num_base_layers + 1),
                BaseLayer(cfg),
            )

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
        decoder_input_mask=None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = True,
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

        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            decoder_input_mask=decoder_input_mask,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )

        if not features_only:
            x = self.output_layer(x)

        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out,
        decoder_input_mask,
        # encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = True,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            decoder_input_mask,
            incremental_state, # should None
            full_context_alignment, # should True
            alignment_layer,
            alignment_heads,
        )

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out,
        decoder_input_mask,
        # encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None, # should None
        full_context_alignment: bool = True, # should True
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

        # import pdb; pdb.set_trace()

        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        # import pdb; pdb.set_trace()

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None

        # if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
        #     enc = encoder_out["encoder_out"][0]
        #     assert (
        #         enc.size()[1] == bs
        #     ), f"Expected enc.shape == (t, {bs}, c) got {enc.shape}"
        # if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
        #     padding_mask = encoder_out["encoder_padding_mask"][0]

        # import pdb; pdb.set_trace()

        if self.encoder_output_proj:
            enc = self.encoder_output_proj(encoder_out['encoder_out'])
        else:
            enc = encoder_out['encoder_out']
        padding_mask = encoder_out["padding_mask"]

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
        self_attn_padding_mask = decoder_input_mask

        # import pdb; pdb.set_trace()


        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]

        # s1 = time.time()

        for idx, layer in enumerate(self.layers):

            # This should be None if you want to use NAR Decoding
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None
            
            # import pdb; pdb.set_trace()

            x, layer_attn, _ = layer(
                x, # roberta hidden # torch.Size([51, 8, 768])
                enc, # w2v2 hidden # torch.Size([634, 8, 768])
                padding_mask, # None
                incremental_state, # None
                self_attn_mask=self_attn_mask, # None
                self_attn_padding_mask=self_attn_padding_mask, # torch.Size([8, 51])
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )

            inner_states.append(x)

            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        # s2 = time.time()
        # print('full_context_alignment : {}, time : {}'.format(full_context_alignment,s2-s1))

        # AR vs NAR
        '''
        [2022-02-26 20:39:15,262][fairseq_cli.train][INFO] - Start iterating over samples
        full_context_alignment : True, time : 0.03482556343078613
        full_context_alignment : True, time : 0.016609668731689453
        full_context_alignment : True, time : 0.007838964462280273
        full_context_alignment : True, time : 0.007956743240356445
        full_context_alignment : True, time : 0.008926153182983398
        full_context_alignment : True, time : 0.006220340728759766
        full_context_alignment : True, time : 0.006015777587890625
        full_context_alignment : True, time : 0.005965232849121094
        full_context_alignment : True, time : 0.0075855255126953125
        full_context_alignment : True, time : 0.00720524787902832
        full_context_alignment : True, time : 0.005980491638183594
        full_context_alignment : True, time : 0.00596308708190918
        full_context_alignment : True, time : 0.007592916488647461
        full_context_alignment : True, time : 0.00717926025390625
        full_context_alignment : True, time : 0.007151603698730469
        full_context_alignment : True, time : 0.007178544998168945
        full_context_alignment : True, time : 0.007654428482055664
        full_context_alignment : True, time : 0.0061762332916259766
        full_context_alignment : True, time : 0.006176948547363281
        full_context_alignment : True, time : 0.0074460506439208984
        full_context_alignment : True, time : 0.006490230560302734
        full_context_alignment : True, time : 0.008759737014770508
        full_context_alignment : True, time : 0.006211042404174805
        full_context_alignment : True, time : 0.007407188415527344
        full_context_alignment : True, time : 0.007780551910400391
        full_context_alignment : True, time : 0.007874727249145508
        full_context_alignment : True, time : 0.005155801773071289

        [2022-02-26 20:42:57,648][fairseq_cli.train][INFO] - Start iterating over samples
        full_context_alignment : False, time : 0.04191136360168457
        full_context_alignment : False, time : 0.010588645935058594
        full_context_alignment : False, time : 0.008368730545043945
        full_context_alignment : False, time : 0.008075475692749023
        full_context_alignment : False, time : 0.009489297866821289
        full_context_alignment : False, time : 0.00696873664855957
        full_context_alignment : False, time : 0.007178783416748047
        full_context_alignment : False, time : 0.00745844841003418
        full_context_alignment : False, time : 0.008281469345092773
        full_context_alignment : False, time : 0.010922670364379883
        full_context_alignment : False, time : 0.006968975067138672
        full_context_alignment : False, time : 0.006543636322021484
        full_context_alignment : False, time : 0.008106231689453125
        full_context_alignment : False, time : 0.0077364444732666016
        full_context_alignment : False, time : 0.007710695266723633
        full_context_alignment : False, time : 0.007706403732299805
        full_context_alignment : False, time : 0.008033990859985352
        full_context_alignment : False, time : 0.006588459014892578
        full_context_alignment : False, time : 0.0076906681060791016
        full_context_alignment : False, time : 0.007881641387939453
        full_context_alignment : False, time : 0.0068128108978271484
        full_context_alignment : False, time : 0.008458614349365234
        full_context_alignment : False, time : 0.0065059661865234375
        full_context_alignment : False, time : 0.007749795913696289
        full_context_alignment : False, time : 0.008281946182250977
        full_context_alignment : False, time : 0.007920503616333008
        '''
        

        # import pdb; pdb.set_trace()

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}

    def output_layer(self, features):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            return self.output_projection(features)
        else:
            return features

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
#########################        Align bpe tokens to word tokens      ##########################
################################################################################################


def align_bpe_to_words(roberta_dict, bpe, bpe_tokens: torch.LongTensor, other_tokens: List[str]):

    assert bpe_tokens.dim() == 1
    assert bpe_tokens[0] == 0

    def clean(text):
        return text.strip()

    # remove whitespaces to simplify alignment
    bpe_tokens = [roberta_dict.string([x]) for x in bpe_tokens]
    bpe_tokens = [
        clean(bpe.decode(x) if x not in {"<s>", ""} else x) for x in bpe_tokens
    ]
    other_tokens = [clean(str(o)) for o in other_tokens]

    # strip leading <s>
    bpe_tokens = bpe_tokens[1:]
    assert "".join(bpe_tokens) == "".join(other_tokens)

    # create alignment from every word to a list of BPE tokens
    alignment = []
    bpe_toks = filter(lambda item: item[1] != "", enumerate(bpe_tokens, start=1))
    j, bpe_tok = next(bpe_toks)
    for other_tok in other_tokens:
        bpe_indices = []
        while True:
            if other_tok.startswith(bpe_tok):
                bpe_indices.append(j)
                other_tok = other_tok[len(bpe_tok) :]
                try:
                    j, bpe_tok = next(bpe_toks)
                except StopIteration:
                    j, bpe_tok = None, None
            elif bpe_tok.startswith(other_tok):
                # other_tok spans multiple BPE tokens
                bpe_indices.append(j)
                bpe_tok = bpe_tok[len(other_tok) :]
                other_tok = ""
            else:
                raise Exception('Cannot align "{}" and "{}"'.format(other_tok, bpe_tok))
            if other_tok == "":
                break
        assert len(bpe_indices) > 0

        alignment.append(bpe_indices[0])
    assert len(alignment) == len(other_tokens)

    return alignment




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
    j = trellis.size(1) - 1
    t_start = torch.argmax(trellis[:, j]).item()

    path = []
    for t in range(t_start, 0, -1):
        # 1. Figure out if the current position was stay or change
        stayed = trellis[t-1, j] + emission[t-1, blank_id]
        changed = trellis[t-1, j-1] + emission[t-1, tokens[j-1]]

        # 2. Store the path with frame-wise probability.
        prob = emission[t-1, tokens[j-1] if changed > stayed else 0].exp().item()
        path.append(Point(j-1, t-1, prob))

        # 3. Update the token
        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        raise ValueError('Failed to align')
    return path[::-1]


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


def merge_repeats(path, transcript):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(Segment(transcript[path[i1].token_index], path[i1].time_index, path[i2-1].time_index + 1, score))
        i1 = i2
    return segments


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