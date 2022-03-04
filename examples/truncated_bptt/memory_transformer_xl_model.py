# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from fairseq.dataclass import FairseqDataclass
from fairseq.models import (
    FairseqIncrementalDecoder,
    FairseqLanguageModel,
    register_model,
)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from omegaconf import II


logger = logging.getLogger(__name__)



'''
from https://github.com/lucidrains/memory-transformer-xl

import torch
from memory_transformer_xl import MemoryTransformerXL

model = MemoryTransformerXL(
    num_tokens = 20000,
    dim = 1024,
    heads = 8,
    depth = 8,
    seq_len = 512,
    mem_len = 256,            # short term memory (the memory from transformer-xl)
    lmem_len = 256,           # long term memory (memory attention network attending to short term memory and hidden activations)
    mem_write_iters = 2,      # number of iterations of attention for writing to memory
    memory_layers = [6,7,8],  # which layers to use memory, only the later layers are actually needed
    num_mem_kv = 128,         # number of memory key/values, from All-attention paper
).cuda()

x1 = torch.randint(0, 20000, (1, 512)).cuda()
logits1, mem1 = model(x1)

x2 = torch.randint(0, 20000, (1, 512)).cuda()
logits2, mem2 = model(x2, memories = mem1)
'''


@dataclass
class MemoryTransformerXLConfig(FairseqDataclass):
    checkpoint_activations: bool = False
    offload_activations: bool = False
    max_target_positions: int = II("task.max_target_positions")
    dim: int = 512
    heads: int = 8
    depth: int = 4
    emb_dim: int = 512
    seq_len: int = 512
    mem_len: int = 256
    lmem_len: int = 256
    mem_write_iters: int = 2
    memory_layers: List[int] = field(default_factory=lambda: [3,4])
    num_mem_kv: int = 128
    attn_dropout: float = 0.1
    attn_layer_dropout: float = 0.1
    ff_dropout: float = 0.1
    adaptive_input: bool = field(
        default=False, metadata={"help": "if set, uses adaptive input"}
    )
    adaptive_input_factor: float = field(
        default=4, metadata={"help": "adaptive input factor"}
    )
    adaptive_input_cutoff: Optional[str] = field(
        default=None,
        metadata={"help": "comma separated list of adaptive input cutoff points."},
    )
    adaptive_softmax: bool = field(
        default=False, metadata={"help": "if set, uses adaptive softmax"}
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
    # tie projection and embedding
    tie_adaptive_weights: bool = field(
        default=False,
        metadata={
            "help": "if set, ties the weights of adaptive softmax and adaptive input"
        },
    )
    tie_adaptive_proj: bool = field(
        default=False,
        metadata={
            "help": "if set, ties the projection weights of adaptive softmax and adaptive input"
        },
    )
    # config for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
    quant_noise_pq: float = field(
        default=0.0,
        metadata={"help": "iterative PQ quantization noise at training time"},
    )
    quant_noise_pq_block_size: int = field(
        default=8,
        metadata={"help": "block size of quantization noise at training time"},
    )
    quant_noise_scalar: float = field(
        default=0.0,
        metadata={
            "help": "scalar quantization noise and scalar quantization at training time"
        },
    )

    export: bool = field(
        default=False,
        metadata={"help": "make the layernorm exportable with torchscript."},
    )



@register_model("memory_transformer_xl", dataclass=MemoryTransformerXLConfig)
class MemoryTransformerXLLanguageModel(FairseqLanguageModel):
    @classmethod
    def build_model(cls, cfg: MemoryTransformerXLConfig, task):
        return cls(MemoryTransformerXLDecoder(cfg, task))


class MemoryTransformerXLDecoder(FairseqIncrementalDecoder):
    def __init__(self, cfg, task):

        # from memory_transformer_xl import MemoryTransformerXL
        from .memory_transformer_xl.memory_transformer_xl import MemoryTransformerXL # <class 'truncated_bptt.memory_transformer_xl.memory_transformer_xl.MemoryTransformerXL'>

        # import pdb; pdb.set_trace()

        super().__init__(task.target_dictionary)
        self.cfg = cfg

        # import pdb; pdb.set_trace()

        logger.info('model config : {}'.format(cfg))
        self.model = MemoryTransformerXL(
            target_dictionary = task.target_dictionary,
            dim = cfg.dim,
            heads = cfg.heads,
            depth = cfg.depth,
            emb_dim = cfg.emb_dim,
            seq_len = cfg.seq_len,
            mem_len = cfg.mem_len,            # short term memory (the memory from transformer-xl)
            lmem_len = cfg.lmem_len,           # long term memory (memory attention network attending to short term memory and hidden activations)
            mem_write_iters = cfg.mem_write_iters,      # number of iterations of attention for writing to memory
            memory_layers = cfg.memory_layers,  # which layers to use memory, only the later layers are actually needed
            num_mem_kv = cfg.num_mem_kv,         # number of memory key/values, from All-attention paper
            attn_dropout = cfg.attn_dropout,
            attn_layer_dropout = cfg.attn_layer_dropout,
            ff_dropout = cfg.ff_dropout,
            adaptive_input = cfg.adaptive_input,
            adaptive_softmax = cfg.adaptive_softmax,
            adaptive_input_cutoff = cfg.adaptive_input_cutoff,
            adaptive_softmax_cutoff = cfg.adaptive_softmax_cutoff,
        )

        self.adaptive_softmax = self.model.adaptive_softmax
        self.output_proj_layer = self.model.to_logits
        
        self._mems = None

    def forward(
        self,
        src_tokens,
        src_lengths=None,  # unused
        incremental_state: Optional[Dict[str, List[torch.Tensor]]] = None,
        encoder_out=None,
    ):
        if incremental_state is not None:  # used during inference
            mems = self.get_incremental_state(incremental_state, "mems")
            src_tokens = src_tokens[:, -1:]  # only keep the most recent token
        else:
            mems = self._mems

        output = self.model(
            src_tokens,
            memories=mems,
        )

        if len(output) >= 2:
            if incremental_state is not None:
                self.set_incremental_state(incremental_state, "mems", output[1])
            else:
                self._mems = output[1]

        if self.cfg.adaptive_softmax is None:
            final_output = self.output_proj_layer(output[0]) # vocab 사이즈로 -> projection
        else:
            final_output = output[0]

        return (final_output,)

    def max_positions(self):
        return self.cfg.max_target_positions

    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[torch.Tensor]]],
        new_order: torch.Tensor,
    ):
        """Reorder incremental state.

        This will be called when the order of the input has changed from the
        previous time step. A typical use case is beam search, where the input
        order changes between time steps based on the selection of beams.
        """
        mems = self.get_incremental_state(incremental_state, "mems")
        if mems is not None:
            new_mems = [mems_i.index_select(1, new_order) for mems_i in mems]
            self.set_incremental_state(incremental_state, "mems", new_mems)
