#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Flashlight decoders.
"""

import gc
import itertools as it
import os.path as osp
from typing import List
import warnings
from collections import deque, namedtuple

import numpy as np
import torch
from examples.speech_recognition.data.replabels import unpack_replabels
from fairseq import tasks
from fairseq.utils import apply_to_sample
from omegaconf import open_dict
from fairseq.dataclass.utils import convert_namespace_to_omegaconf

import pdb
import time

try:
    from flashlight.lib.text.dictionary import create_word_dict, load_words
    from flashlight.lib.sequence.criterion import CpuViterbiPath, get_data_ptr_as_bytes
    from flashlight.lib.text.decoder import (
        CriterionType,
        LexiconDecoderOptions,
        KenLM,
        LM,
        LMState,
        SmearingMode,
        Trie,
        LexiconDecoder,
    )
except:
    warnings.warn(
        "flashlight python bindings are required to use this functionality. Please install from https://github.com/facebookresearch/flashlight/tree/master/bindings/python"
    )
    LM = object
    LMState = object


class W2lDecoder(object):
    def __init__(self, args, tgt_dict):
        self.tgt_dict = tgt_dict
        self.vocab_size = len(tgt_dict)
        self.nbest = args.nbest

        # criterion-specific init
        self.criterion_type = CriterionType.CTC
        self.blank = (
            tgt_dict.index("<ctc_blank>")
            if "<ctc_blank>" in tgt_dict.indices
            else tgt_dict.bos()
        )

        print('self.tgt_dict.indices',self.tgt_dict.indices)
        print('self.vocab_size',self.vocab_size)

        # self.tgt_dict.indices {'<s>': 0, '<pad>': 1, '</s>': 2, '<unk>': 3, '|': 4, 'E': 5, 'T': 6, 'A': 7, 'O': 8, 'N': 9, 'I': 10, 'H': 11, 'S': 12, 'R': 13, 'D': 14, 'L': 15, 'U': 16, 'M': 17, 'W': 18, 'C': 19, 'F': 20, 'G': 21, 'Y': 22, 'P': 23, 'B': 24, 'V': 25, 'K': 26, "'": 27, 'X': 28, 'J': 29, 'Q': 30, 'Z': 31}
        # self.vocab_size 32

        # import pdb
        # pdb.set_trace()

        if "<sep>" in tgt_dict.indices:
            self.silence = tgt_dict.index("<sep>")
        elif "|" in tgt_dict.indices:
            self.silence = tgt_dict.index("|")
        else:
            self.silence = tgt_dict.eos()
        self.asg_transitions = None

        self.get_emission_time = 0
        self.decoding_time = 0

    def generate(self, models, sample, **unused):
        """Generate a batch of inferences."""
        # model.forward normally channels prev_output_tokens into the decoder
        # separately, but SequenceGenerator directly calls model.encoder

        start = time.time()
        encoder_input = {
            k: v for k, v in sample["net_input"].items() if k != "prev_output_tokens"
        }
        emissions = self.get_emissions(models, encoder_input)
        end = time.time()
        self.get_emission_time += (end-start)

        start = time.time()
        result = self.decode(emissions)
        end = time.time()
        self.decoding_time += (end-start)

        return result

    def get_emissions(self, models, encoder_input):
        """Run encoder and normalize emissions"""

        model = models[0]
        encoder_out = model(**encoder_input)

        if hasattr(model, "get_logits"):
            emissions = model.get_logits(encoder_out) # no need to normalize emissions
        else:
            emissions = model.get_normalized_probs(encoder_out, log_probs=True)
        
        return emissions.transpose(0, 1).float().cpu().contiguous()

    def get_tokens(self, idxs):
        """Normalize tokens by handling CTC blank, ASG replabels, etc."""
        idxs = (g[0] for g in it.groupby(idxs))
        idxs = filter(lambda x: x != self.blank, idxs)
        return torch.LongTensor(list(idxs))


class W2lViterbiDecoder(W2lDecoder):
    def __init__(self, args, tgt_dict):
        super().__init__(args, tgt_dict)

    def decode(self, emissions):
        B, T, N = emissions.size()
        hypos = []

        if self.asg_transitions is None:
            transitions = torch.FloatTensor(N, N).zero_()
        else:
            transitions = torch.FloatTensor(self.asg_transitions).view(N, N)
        
        viterbi_path = torch.IntTensor(B, T)
        workspace = torch.ByteTensor(CpuViterbiPath.get_workspace_size(B, T, N))
        CpuViterbiPath.compute(
            B,
            T,
            N,
            get_data_ptr_as_bytes(emissions),
            get_data_ptr_as_bytes(transitions),
            get_data_ptr_as_bytes(viterbi_path),
            get_data_ptr_as_bytes(workspace),
        )
        return [
            [{"tokens": self.get_tokens(viterbi_path[b].tolist()), "score": 0}]
            for b in range(B)
        ]


class W2lKenLMDecoder(W2lDecoder):
    def __init__(self, args, tgt_dict):
        super().__init__(args, tgt_dict)

        self.unit_lm = getattr(args, "unit_lm", False)

        # (Pdb) self.unit_lm
        # False

        if args.lexicon:
            self.lexicon = load_words(args.lexicon)
            '''
            'TAGISH': [['T', 'A', 'G', 'I', 'S', 'H', '|']], 
            'TAGLAT': [['T', 'A', 'G', 'L', 'A', 'T', '|']], 
            "TAGLIONI'S": [['T', 'A', 'G', 'L', 'I', 'O', 'N', 'I', "'", 'S', '|']], 
            'TAGLITZ': [['T', 'A', 'G', 'L', 'I', 'T', 'Z', '|']], 
            'TAGUS': [['T', 'A', 'G', 'U', 'S', '|']], 
            'GENTLEMANS': [['G', 'E', 'N', 'T', 'L', 'E', 'M', 'A', 'N', 'S', '|']], 
            'TAH': [['T', 'A', 'H', '|']], 
            'TAHAITI': [['T', 'A', 'H', 'A', 'I', 'T', 'I', '|']]}
            '''

            self.word_dict = create_word_dict(self.lexicon)
            self.unk_word = self.word_dict.get_index("<unk>")

            self.lm = KenLM(args.kenlm_model, self.word_dict)
            self.trie = Trie(self.vocab_size, self.silence)

            word_list = []
            score_list = []
            spelling_idxs_list = []

            start_state = self.lm.start(False)

            for i, (word, spellings) in enumerate(self.lexicon.items()):
                word_idx = self.word_dict.get_index(word)
                _, score = self.lm.score(start_state, word_idx)
                for spelling in spellings:
                    spelling_idxs = [tgt_dict.index(token) for token in spelling]

                    word_list.append(word)
                    score_list.append(score)
                    spelling_idxs_list.append(spelling_idxs)

                    # import pdb
                    # pdb.set_trace()

                    assert (
                        tgt_dict.unk() not in spelling_idxs
                    ), f"{spelling} {spelling_idxs}"
                    self.trie.insert(spelling_idxs, word_idx, score)

                    # import pdb
                    # pdb.set_trace()

            self.trie.smear(SmearingMode.MAX)


            # tmp = int(getattr(args, "beam_size_token", len(tgt_dict))) # 100
            # tmp = args.beam_size_token or len(tgt_dict)
            # import pdb
            # pdb.set_trace()


            print('kenlm weight',args.lm_weight)
            print('kenlm word penalty',args.word_score)
            print('kenlm beam size',args.beam)

            self.decoder_opts = LexiconDecoderOptions(
                beam_size=args.beam,
                # beam_size_token=int(getattr(args, "beam_size_token", len(tgt_dict))), # am vocab
                # beam_size_token = int(len(tgt_dict)), 
                beam_size_token = args.beam_size_token or len(tgt_dict),
                beam_threshold=args.beam_threshold,
                lm_weight=args.lm_weight,
                word_score=args.word_score,
                unk_score=args.unk_weight,
                sil_score=args.sil_weight,
                log_add=False,
                criterion_type=self.criterion_type,
            )

            if self.asg_transitions is None:
                N = 768
                # self.asg_transitions = torch.FloatTensor(N, N).zero_()
                self.asg_transitions = []

            self.decoder = LexiconDecoder(
                self.decoder_opts,
                self.trie, # word 
                self.lm,
                self.silence,
                self.blank,
                self.unk_word,
                self.asg_transitions,
                self.unit_lm,
            )
        else:
            assert args.unit_lm, "lexicon free decoding can only be done with a unit language model"
            from flashlight.lib.text.decoder import LexiconFreeDecoder, LexiconFreeDecoderOptions

            d = {w: [[w]] for w in tgt_dict.symbols}
            self.word_dict = create_word_dict(d)
            self.lm = KenLM(args.kenlm_model, self.word_dict)
            self.decoder_opts = LexiconFreeDecoderOptions(
                beam_size=args.beam,
                beam_size_token=int(getattr(args, "beam_size_token", len(tgt_dict))),
                beam_threshold=args.beam_threshold,
                lm_weight=args.lm_weight,
                sil_score=args.sil_weight,
                log_add=False,
                criterion_type=self.criterion_type,
            )
            self.decoder = LexiconFreeDecoder( # flashlight cpp
                self.decoder_opts, self.lm, self.silence, self.blank, []
            )

        # import pdb
        # pdb.set_trace()


    def get_timesteps(self, token_idxs: List[int]) -> List[int]:
        """Returns frame numbers corresponding to every non-blank token.

        Parameters
        ----------
        token_idxs : List[int]
            IDs of decoded tokens.

        Returns
        -------
        List[int]
            Frame numbers corresponding to every non-blank token.
        """
        timesteps = []
        for i, token_idx in enumerate(token_idxs):
            if token_idx == self.blank:
                continue
            if i == 0 or token_idx != token_idxs[i-1]:
                timesteps.append(i)
        return timesteps


    def decode(self, emissions):
        B, T, N = emissions.size()
        hypos = []
        for b in range(B):
            emissions_ptr = emissions.data_ptr() + 4 * b * emissions.stride(0)
            results = self.decoder.decode(emissions_ptr, T, N) # cpp

            nbest_results = results[: self.nbest] # nbest 가 1

            # import pdb; pdb.set_trace()

            '''
            (Pdb) nbest_results[0].amScore
            21540.401293754578
            (Pdb) nbest_results[0].lmScore
            -258.7231324315071
            '''

            hypos.append(
                [
                    {
                        "tokens": self.get_tokens(result.tokens),
                        "score": result.score,
                        "am_score": result.amScore,
                        "lm_score": result.lmScore,
                        "timesteps": self.get_timesteps(result.tokens),
                        "words": [
                            self.word_dict.get_entry(x) for x in result.words if x >= 0
                        ],
                    }
                    for result in nbest_results
                ]
            )

        return hypos



FairseqLMState = namedtuple("FairseqLMState", ["prefix", "incremental_state", "probs"])


class FairseqLM(LM):
    def __init__(self, dictionary, model):
        LM.__init__(self)
        self.dictionary = dictionary
        self.model = model
        self.unk = self.dictionary.unk()

        self.save_incremental = False  # this currently does not work properly
        self.max_cache = 20_000

        # lstm, convlm, transformer
        model.cuda()
        model.eval()
        model.make_generation_fast_()

        # for beam search
        self.states = {}
        self.stateq = deque()

    def start(self, start_with_nothing):
        state = LMState()
        prefix = torch.LongTensor([[self.dictionary.eos()]])
        incremental_state = {} if self.save_incremental else None

        with torch.no_grad():
            res = self.model(prefix.cuda(), incremental_state=incremental_state)
            probs = self.model.get_normalized_probs(res, log_probs=True, sample=None)

        if incremental_state is not None:
            incremental_state = apply_to_sample(lambda x: x.cpu(), incremental_state)
        self.states[state] = FairseqLMState(
            prefix.numpy(), incremental_state, probs[0, -1].cpu().numpy()
        )
        self.stateq.append(state)

        return state

    def score(self, state: LMState, token_index: int, no_cache: bool = False):
        """
        Evaluate language model based on the current lm state and new word
        Parameters:
        -----------
        state: current lm state
        token_index: index of the word
                     (can be lexicon index then you should store inside LM the
                      mapping between indices of lexicon and lm, or lm index of a word)

        Returns:
        --------
        (LMState, float): pair of (new state, score for the current word)
        """
        curr_state = self.states[state]

        def trim_cache(targ_size):
            while len(self.stateq) > targ_size:
                rem_k = self.stateq.popleft()
                rem_st = self.states[rem_k]
                rem_st = FairseqLMState(rem_st.prefix, None, None)
                self.states[rem_k] = rem_st
        
        # import pdb; pdb.set_trace()

        if curr_state.probs is None:
            new_incremental_state = (
                curr_state.incremental_state.copy()
                if curr_state.incremental_state is not None
                else None
            )
            with torch.no_grad():
                if new_incremental_state is not None:
                    new_incremental_state = apply_to_sample(
                        lambda x: x.cuda(), new_incremental_state
                    )
                elif self.save_incremental:
                    new_incremental_state = {}

                res = self.model(
                    torch.from_numpy(curr_state.prefix).cuda(),
                    incremental_state=new_incremental_state,
                )
                probs = self.model.get_normalized_probs(
                    res, log_probs=True, sample=None
                )

                if new_incremental_state is not None:
                    new_incremental_state = apply_to_sample(
                        lambda x: x.cpu(), new_incremental_state
                    )

                curr_state = FairseqLMState(
                    curr_state.prefix, new_incremental_state, probs[0, -1].cpu().numpy()
                )

            if not no_cache:
                self.states[state] = curr_state
                self.stateq.append(state)

        score = curr_state.probs[token_index].item()

        trim_cache(self.max_cache)

        outstate = state.child(token_index)
        if outstate not in self.states and not no_cache:
            prefix = np.concatenate(
                [curr_state.prefix, torch.LongTensor([[token_index]])], -1
            )
            incr_state = curr_state.incremental_state

            self.states[outstate] = FairseqLMState(prefix, incr_state, None)

        if token_index == self.unk:
            score = float("-inf")

        # import pdb; pdb.set_trace()

        return outstate, score

    def finish(self, state: LMState):
        """
        Evaluate eos for language model based on the current lm state

        Returns:
        --------
        (LMState, float): pair of (new state, score for the current word)
        """
        return self.score(state, self.dictionary.eos())

    def empty_cache(self):
        self.states = {}
        self.stateq = deque()
        gc.collect()


class W2lFairseqLMDecoder(W2lDecoder):
    def __init__(self, args, tgt_dict):
        super().__init__(args, tgt_dict)

        self.unit_lm = getattr(args, "unit_lm", False)

        self.lexicon = load_words(args.lexicon) if args.lexicon else None
        self.idx_to_wrd = {}

        checkpoint = torch.load(args.kenlm_model, map_location="cpu")

        if "cfg" in checkpoint and checkpoint["cfg"] is not None:
            lm_args = checkpoint["cfg"]
        else:
            lm_args = convert_namespace_to_omegaconf(checkpoint["args"])

        from omegaconf import OmegaConf
        if type(lm_args) is dict:
            lm_args = OmegaConf.create(lm_args)
        # pdb.set_trace()

        try:
            with open_dict(lm_args.task):
                lm_args.task.data = osp.dirname(args.kenlm_model) # osp is os.path
        except:
            pass

        '''
        (Pdb) lm_args.task
        {'_name': 'language_modeling', 'data': '/checkpoint/vineelkpratap/librispeech', 'sample_break_mode': 'eos', 'tokens_per_sample': 256, 
        'output_dictionary_size': -1, 'self_target': False, 'future_target': False, 'past_target': False, 'add_bos_token': False, 
        'first_source_token_bos': False, 'max_target_positions': 256, 'shorten_method': 'none', 'shorten_data_split_list': '', 
        'pad_to_fixed_length': False, 'pad_to_fixed_bsz': False, 'seed': 1, 'batch_size': None, 'batch_size_valid': None, 
        'dataset_impl': None, 'data_buffer_size': 10, 'tpu': True, 'use_plasma_view': True, 'plasma_path': '/tmp/plasma'}

        (Pdb) lm_args.model
        {'_name': 'transformer_lm_gbw', 'activation_fn': 'relu', 'dropout': 0.1, 'attention_dropout': 0.1, 'activation_dropout': 0.1, 
        'relu_dropout': 0.0, 'decoder_embed_dim': 1280, 'decoder_output_dim': 1280, 'decoder_input_dim': 1280, 'decoder_ffn_embed_dim': 6144, 
        'decoder_layers': 20, 'decoder_attention_heads': 16, 'decoder_normalize_before': True, 'no_decoder_final_norm': True, 
        'adaptive_softmax_cutoff': '60000,160000', 'adaptive_softmax_dropout': 0.0, 'adaptive_softmax_factor': 4.0, 'no_token_positional_embeddings': False, 
        'share_decoder_input_output_embed': False, 'character_embeddings': False, 'character_filters': '[(1, 64), (2, 128), (3, 192), (4, 256), (5, 256), (6, 256), (7, 256)]', 
        'character_embedding_dim': 4, 'char_embedder_highway_layers': 2, 'adaptive_input': True, 'adaptive_input_factor': 4.0, 'adaptive_input_cutoff': '60000,160000', 
        'tie_adaptive_weights': True, 'tie_adaptive_proj': False, 'decoder_learned_pos': False, 'layernorm_embedding': False, 'no_scale_embedding': False, 
        'checkpoint_activations': False, 'offload_activations': False, 'decoder_layerdrop': 0.0, 'decoder_layers_to_keep': None, 'quant_noise_pq': 0.0, 
        'quant_noise_pq_block_size': 8, 'quant_noise_scalar': 0.0, 'min_params_to_wrap': 100000000, 'base_layers': 0, 'base_sublayers': 1, 'base_shuffle': 1, 
        'scale_fc': False, 'scale_attn': False, 'scale_heads': False, 'scale_resids': False, 'add_bos_token': False, 'tokens_per_sample': 256, 
        'max_target_positions': 256, 'tpu': True}

        (Pdb) lm_args.task.data
        '/checkpoint/vineelkpratap/librispeech'
        '''

        # load
        try:
            task = tasks.setup_task(lm_args.task)
            model = task.build_model(lm_args.model)
        except:
            # task = tasks.setup_task(lm_args)
            task = tasks.setup_task(lm_args['task'])
            model = task.build_model(lm_args['model'])
        model.load_state_dict(checkpoint["model"], strict=False)

        # trie, cpp flashlight
        # if self.lexicon:
        #     self.trie = Trie(self.vocab_size, self.silence)
        self.trie = Trie(self.vocab_size, self.silence)

        # define dictionary, lm
        self.word_dict = task.dictionary # kenlm : lexicon -> lm word dict
        self.unk_word = self.word_dict.unk()
        self.lm = FairseqLM(self.word_dict, model)

        '''
        (Pdb) len(self.word_dict)
        221456
        (Pdb) len(self.lexicon)
        221449
        (Pdb) self.unit_lm
        False
        '''
        
        if self.lexicon:

            '''
            'TAGISH': [['T', 'A', 'G', 'I', 'S', 'H', '|']], 
            'TAGLAT': [['T', 'A', 'G', 'L', 'A', 'T', '|']], 
            "TAGLIONI'S": [['T', 'A', 'G', 'L', 'I', 'O', 'N', 'I', "'", 'S', '|']], 
            'TAGLITZ': [['T', 'A', 'G', 'L', 'I', 'T', 'Z', '|']], 
            'TAGUS': [['T', 'A', 'G', 'U', 'S', '|']], 
            'GENTLEMANS': [['G', 'E', 'N', 'T', 'L', 'E', 'M', 'A', 'N', 'S', '|']], 
            'TAH': [['T', 'A', 'H', '|']], 
            'TAHAITI': [['T', 'A', 'H', 'A', 'I', 'T', 'I', '|']]}
            '''

            start_state = self.lm.start(False)
            for i, (word, spellings) in enumerate(self.lexicon.items()):
                # pdb.set_trace()
                if self.unit_lm:
                    word_idx = i
                    self.idx_to_wrd[i] = word
                    score = 0
                else:
                    word_idx = self.word_dict.index(word)
                    _, score = self.lm.score(start_state, word_idx, no_cache=True)

                for spelling in spellings:
                    spelling_idxs = [tgt_dict.index(token) for token in spelling]
                    # pdb.set_trace()

                    '''
                    (Pdb) (word, word_idx, score)
                    ('zwilling', 221451, -16.700355529785156)
                    (Pdb) (spelling_idxs, word_idx, score)
                    ([31, 18, 10, 15, 15, 10, 9, 21, 4], 221451, -16.700355529785156)
                    (Pdb) (self.tgt_dict.string(spelling_idxs))
                    'Z W I L L I N G |'

                    (Pdb) (word, word_idx, score)
                    ('zuylestein', 221448, -17.481908798217773)
                    (Pdb) (spelling_idxs, word_idx, score)
                    ([31, 16, 22, 15, 5, 12, 6, 5, 10, 9, 4], 221448, -17.481908798217773)
                    (Pdb) (self.tgt_dict.string(spelling_idxs))
                    'Z U Y L E S T E I N |'
                    '''

                    assert (
                        tgt_dict.unk() not in spelling_idxs
                    ), f"{spelling} {spelling_idxs}"

                    self.trie.insert(spelling_idxs, word_idx, score)

            self.trie.smear(SmearingMode.MAX)

            self.decoder_opts = LexiconDecoderOptions(
                beam_size=args.beam,
                beam_size_token=int(getattr(args, "beam_size_token", len(tgt_dict))),
                beam_threshold=args.beam_threshold,
                lm_weight=args.lm_weight,
                word_score=args.word_score,
                unk_score=args.unk_weight,
                sil_score=args.sil_weight,
                log_add=False,
                criterion_type=self.criterion_type,
            )

            self.decoder = LexiconDecoder( # C++ Decoder
                self.decoder_opts,
                self.trie,
                self.lm,
                self.silence,
                self.blank,
                self.unk_word,
                [],
                self.unit_lm,
            )
        else:
            assert args.unit_lm, "lexicon free decoding can only be done with a unit language model"
            from flashlight.lib.text.decoder import LexiconFreeDecoder, LexiconFreeDecoderOptions

            # pdb.set_trace()
            d = {w: [[w]] for w in tgt_dict.symbols}
            self.word_dict = create_word_dict(d)
            # self.ngram_lm = KenLM(args.kenlm_model, self.word_dict) # <- What Happen if i activate this line?
            # pdb.set_trace()
            self.decoder_opts = LexiconFreeDecoderOptions(
                beam_size=args.beam,
                beam_size_token=int(getattr(args, "beam_size_token", len(tgt_dict))),
                beam_threshold=args.beam_threshold,
                lm_weight=args.lm_weight,
                sil_score=args.sil_weight,
                log_add=False,
                criterion_type=self.criterion_type,
            )
            self.decoder = LexiconFreeDecoder(
                self.decoder_opts, self.lm, self.silence, self.blank, []
            )

    def decode(self, emissions):
        B, T, N = emissions.size()
        hypos = []

        def idx_to_word(idx):
            if self.unit_lm:
                return self.idx_to_wrd[idx]
            else:
                return self.word_dict[idx]

        def make_hypo(result, tgt_dict):
            # print('result',result)
            hypo = {"tokens": self.get_tokens(result.tokens), "score": result.score}
            if self.lexicon:
                hypo["words"] = [idx_to_word(x) for x in result.words if x >= 0]
            # print('hypo',hypo)
            # print(tgt_dict.string(hypo["tokens"].int().cpu()))
            return hypo

        # print('emissions.size()',emissions.size()) #torch.Size([7, 1725, 32]), B, T, N

        # print('batch beam decoding start\n')
        
        for b in range(B):

            # emissons -> CTC
            emissions_ptr = emissions.data_ptr() + 4 * b * emissions.stride(0) 
            results = self.decoder.decode(emissions_ptr, T, N)  # beam search

            # pdb.set_trace()

            # print('emissions.data_ptr()',emissions.data_ptr())
            # print('b',b)
            # print('emissions.stride(0)',emissions.stride(0))
            # print('emissions_ptr',emissions_ptr)
            # print('len(results)',len(results))
            # print('results',results)
            # print('self.nbest',self.nbest)

            # 결과 저장
            nbest_results = results[: self.nbest]
            hypos.append([make_hypo(result, self.tgt_dict) for result in nbest_results])
            self.lm.empty_cache()

            # print('\n')
        
        # print('\nbatch beam decoding done')

        return hypos



'''
emissions.size() torch.Size([7, 1725, 32])


batch beam decoding start

emissions.data_ptr() 2768996864
b 0
emissions.stride(0) 55200
emissions_ptr 2768996864
len(results) 3
results [<flashlight.lib.text.flashlight_lib_text_decoder.DecodeResult object at 0x7f31f2303670>, <flashlight.lib.text.flashlight_lib_text_decoder.DecodeResult object at 0x7f31f2303970>, <flashlight.lib.text.flashlight_lib_text_decoder.DecodeResult object at 0x7f31f2303ab0>]
self.nbest 1
result <flashlight.lib.text.flashlight_lib_text_decoder.DecodeResult object at 0x7f31f2303670>
hypo {'tokens': tensor([ 4,  4,  7, 17,  4, 18, 10, 15, 15, 10,  9, 21,  9,  5, 12, 12,  4]), 'score': 14280.519228853285, 'words': ['am']}
| | A M | W I L L I N G N E S S |


emissions.data_ptr() 2768996864
b 1
emissions.stride(0) 55200
emissions_ptr 2769217664
len(results) 3
results [<flashlight.lib.text.flashlight_lib_text_decoder.DecodeResult object at 0x7f31f1ebddf0>, <flashlight.lib.text.flashlight_lib_text_decoder.DecodeResult object at 0x7f31f1ebdbf0>, <flashlight.lib.text.flashlight_lib_text_decoder.DecodeResult object at 0x7f31f1ebdcf0>]
self.nbest 1
result <flashlight.lib.text.flashlight_lib_text_decoder.DecodeResult object at 0x7f31f1ebddf0>
hypo {'tokens': tensor([ 4,  6, 11, 10, 12,  4, 11,  5, 13, 12,  5,  9,  4, 12,  7, 16, 19,  5,
        13, 15, 10, 26,  5,  4]), 'score': inf, 'words': ['this', 'hersen']}
| T H I S | H E R S E N | S A U C E R L I K E |


emissions.data_ptr() 2768996864
b 2
emissions.stride(0) 55200
emissions_ptr 2769438464
len(results) 3
results [<flashlight.lib.text.flashlight_lib_text_decoder.DecodeResult object at 0x7f31f1ebd1f0>, <flashlight.lib.text.flashlight_lib_text_decoder.DecodeResult object at 0x7f31f1f298f0>, <flashlight.lib.text.flashlight_lib_text_decoder.DecodeResult object at 0x7f31f1f29af0>]
self.nbest 1
result <flashlight.lib.text.flashlight_lib_text_decoder.DecodeResult object at 0x7f31f1ebd1f0>
hypo {'tokens': tensor([ 4, 11,  5,  4, 14, 18,  5, 15, 15, 10,  9,  4, 19,  8,  9, 12, 10, 14,
         5, 13,  7, 24, 15,  5,  4]), 'score': inf, 'words': ['he', 'dwellin']}
| H E | D W E L L I N | C O N S I D E R A B L E |


emissions.data_ptr() 2768996864
b 3
emissions.stride(0) 55200
emissions_ptr 2769659264
len(results) 2
results [<flashlight.lib.text.flashlight_lib_text_decoder.DecodeResult object at 0x7f31f1ecf9f0>, <flashlight.lib.text.flashlight_lib_text_decoder.DecodeResult object at 0x7f31f1f294f0>]
self.nbest 1
result <flashlight.lib.text.flashlight_lib_text_decoder.DecodeResult object at 0x7f31f1ecf9f0>
hypo {'tokens': tensor([ 4, 20, 13,  7,  9, 31,  4, 18, 11,  8, 12,  5, 12,  8,  5, 25,  5, 13,
         4]), 'score': inf, 'words': ['franz']}
| F R A N Z | W H O S E S O E V E R |


emissions.data_ptr() 2768996864
b 4
emissions.stride(0) 55200
emissions_ptr 2769880064
len(results) 2
results [<flashlight.lib.text.flashlight_lib_text_decoder.DecodeResult object at 0x7f31f1f297f0>, <flashlight.lib.text.flashlight_lib_text_decoder.DecodeResult object at 0x7f31f1f295b0>]
self.nbest 1
result <flashlight.lib.text.flashlight_lib_text_decoder.DecodeResult object at 0x7f31f1f297f0>
hypo {'tokens': tensor([ 4,  7, 19, 19,  8, 13, 14, 10,  9, 21, 15, 22,  4, 12, 11,  5,  6, 15,
         7,  9, 14,  5, 13,  4]), 'score': inf, 'words': ['accordingly']}
| A C C O R D I N G L Y | S H E T L A N D E R |


emissions.data_ptr() 2768996864
b 5
emissions.stride(0) 55200
emissions_ptr 2770100864
len(results) 4
results [<flashlight.lib.text.flashlight_lib_text_decoder.DecodeResult object at 0x7f31f1e4deb0>, <flashlight.lib.text.flashlight_lib_text_decoder.DecodeResult object at 0x7f31f1e4d4f0>, <flashlight.lib.text.flashlight_lib_text_decoder.DecodeResult object at 0x7f31f1f294b0>, <flashlight.lib.text.flashlight_lib_text_decoder.DecodeResult object at 0x7f31f1f29630>]
self.nbest 1
result <flashlight.lib.text.flashlight_lib_text_decoder.DecodeResult object at 0x7f31f1e4deb0>
hypo {'tokens': tensor([ 4,  7,  9, 14,  4, 22,  5,  6,  6,  7, 27,  4]), 'score': inf, 'words': ['and']}
| A N D | Y E T T A ' |



2021-12-01 07:48:38 | INFO | __main__ | HYPO:am
2021-12-01 07:48:38 | INFO | __main__ | TARGET:I AM WILLING TO ENTER INTO COMPETITION WITH THE ANCIENTS AND FEEL ABLE TO SURPASS THEM FOR SINCE THOSE EARLY DAYS IN WHICH I MADE THE MEDALS OF POPE CLEMENT I HAVE LEARNED SO MUCH THAT I CAN NOW PRODUCE FAR BETTER PIECES OF THE KIND I THINK I CAN ALSO OUTDO THE COINS 
I STRUCK FOR DUKE ALESSANDRO WHICH ARE STILL HELD IN HIGH ESTEEM IN LIKE MANNER I COULD MAKE FOR YOU LARGE PIECES OF GOLD AND SILVER PLATE AS I DID SO OFTEN FOR THAT NOBLE MONARCH KING FRANCIS OF FRANCE THANKS TO THE GREAT CONVENIENCES HE ALLOWED ME WITHOUT EVER LOSING TIME FOR THE EXECUTION OF COLOSSAL STATUES OR
 OTHER WORKS OF THE SCULPTORS CRAFT
2021-12-01 07:48:38 | INFO | __main__ | ___________________
2021-12-01 07:48:38 | INFO | __main__ | HYPO:this hersen
2021-12-01 07:48:38 | INFO | __main__ | TARGET:THIS HE SET IN A SAUCER WETTED WITH A LITTLE WATER AND AFTER WAITING A SHORT TIME SMELT AND TASTED IT AND THEN HE TOOK OUT OF THE CHEST A BOOKLET WHEREIN HE READ AWHILE AND SAID WEEPING KNOW O YE PASSENGERS THAT IN THIS BOOK IS A MARVELLOUS MATTER DENOTING THAT WHOSO 
COMETH HITHER SHALL SURELY DIE WITHOUT HOPE OF ESCAPE FOR THAT THIS OCEAN IS CALLED THE SEA OF THE CLIME OF THE KING WHEREIN IS THE SEPULCHRE OF OUR LORD SOLOMON SON OF DAVID ON BOTH BE PEACE
2021-12-01 07:48:38 | INFO | __main__ | ___________________
2021-12-01 07:48:38 | INFO | __main__ | HYPO:he dwellin
2021-12-01 07:48:38 | INFO | __main__ | TARGET:HE DWELT WITH CONSIDERABLE FORCE AND ENERGY ON THE ALMOST MAGICAL HOSPITALITY HE HAD RECEIVED FROM THE COUNT AND THE MAGNIFICENCE OF HIS ENTERTAINMENT IN THE GROTTO OF THE THOUSAND AND ONE NIGHTS HE RECOUNTED WITH CIRCUMSTANTIAL EXACTITUDE ALL THE PARTICULARS OF THE S
UPPER THE HASHISH THE STATUES THE DREAM AND HOW AT HIS AWAKENING THERE REMAINED NO PROOF OR TRACE OF ALL THESE EVENTS SAVE THE SMALL YACHT SEEN IN THE DISTANT HORIZON DRIVING UNDER FULL SAIL TOWARD PORTO VECCHIO
2021-12-01 07:48:38 | INFO | __main__ | ___________________
2021-12-01 07:48:38 | INFO | __main__ | HYPO:franz whosesoever
2021-12-01 07:48:38 | INFO | __main__ | TARGET:FRANZ WHO SEEMED ATTRACTED BY SOME INVISIBLE INFLUENCE TOWARDS THE COUNT IN WHICH TERROR WAS STRANGELY MINGLED FELT AN EXTREME RELUCTANCE TO PERMIT HIS FRIEND TO BE EXPOSED ALONE TO THE SINGULAR FASCINATION THAT THIS MYSTERIOUS PERSONAGE SEEMED TO EXERCISE OVER HIM AN
D THEREFORE MADE NO OBJECTION TO ALBERT'S REQUEST BUT AT ONCE ACCOMPANIED HIM TO THE DESIRED SPOT AND AFTER A SHORT DELAY THE COUNT JOINED THEM IN THE SALON
2021-12-01 07:48:38 | INFO | __main__ | ___________________
2021-12-01 07:48:38 | INFO | __main__ | HYPO:accordingly
2021-12-01 07:48:38 | INFO | __main__ | TARGET:ACCORDINGLY SHE TOLD HIM ALL THAT HAD COME TO HER SINCE THEIR SEPARATION AT THE KHAN AND WHAT HAD HAPPENED TO HER WITH THE BADAWI HOW THE MERCHANT HAD BOUGHT HER OF HIM AND HAD TAKEN HER TO HER BROTHER SHARRKAN AND HAD SOLD HER TO HIM HOW HE HAD FREED HER AT THE TIME 
OF BUYING HOW HE HAD MADE A MARRIAGE CONTRACT WITH HER AND HAD GONE IN TO HER AND HOW THE KING THEIR SIRE HAD SENT AND ASKED FOR HER FROM SHARRKAN
2021-12-01 07:48:38 | INFO | __main__ | ___________________
2021-12-01 07:48:38 | INFO | __main__ | HYPO:and
2021-12-01 07:48:38 | INFO | __main__ | TARGET:AND YET THE READER WHO LIKES A COMPLETE IMAGE WHO DESIRES TO READ WITH THE SENSES AS WELL AS WITH THE REASON IS ENTREATED NOT TO FORGET THAT HE PROLONGED HIS CONSONANTS AND SWALLOWED HIS VOWELS THAT HE WAS GUILTY OF ELISIONS AND INTERPOLATIONS WHICH WERE EQUALLY UNEXP
ECTED AND THAT HIS DISCOURSE WAS PERVADED BY SOMETHING SULTRY AND VAST SOMETHING ALMOST AFRICAN IN ITS RICH BASKING TONE SOMETHING THAT SUGGESTED THE TEEMING EXPANSE OF THE COTTON FIELD
2021-12-01 07:48:38 | INFO | __main__ | ___________________
2021-12-01 07:48:38 | INFO | __main__ | HYPO:well inerney
2021-12-01 07:48:38 | INFO | __main__ | TARGET:WELL IN EARNEST IF I WERE A PRINCE THAT LADY SHOULD BE MY MISTRESS BUT I CAN GIVE NO RULE TO ANY ONE ELSE AND PERHAPS THOSE THAT ARE IN NO DANGER OF LOSING THEIR HEARTS TO HER MAY BE INFINITELY TAKEN WITH ONE I SHOULD NOT VALUE AT ALL FOR SO SAYS THE JUSTINIAN WISE PR
OVIDENCE HAS ORDAINED IT THAT BY THEIR DIFFERENT HUMOURS EVERYBODY MIGHT FIND SOMETHING TO PLEASE THEMSELVES WITHAL WITHOUT ENVYING THEIR NEIGHBOURS
2021-12-01 07:48:38 | INFO | __main__ | ___________________


Segmentation fault (core dumped)
'''


'''

Flashlight Decoder Example 참고

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} decoder_test_data_path", file=sys.stderr)
        print("  (usually: <flashlight>/flashlight/app/asr/test/decoder/data)", file=sys.stderr)
        sys.exit(1)

    data_path = sys.argv[1]

    # load test files
    # load time and number of tokens for dumped acoustic scores
    T, N = load_tn(os.path.join(data_path, "TN.bin"))
    # load emissions [Batch=1, Time, Ntokens]
    emissions = load_emissions(os.path.join(data_path, "emission.bin"))
    # load transitions (from ASG loss optimization) [Ntokens, Ntokens]
    transitions = load_transitions(os.path.join(data_path, "transition.bin"))
    # load lexicon file, which defines spelling of words
    # the format word and its tokens spelling separated by the spaces,
    # for example for letters tokens with ASG loss:
    # ann a n 1 |
    lexicon = load_words(os.path.join(data_path, "words.lst"))
    # read lexicon and store it in the w2l dictionary
    word_dict = create_word_dict(lexicon)
    # create w2l dict with tokens set (letters in this example)
    token_dict = Dictionary(os.path.join(data_path, "letters.lst"))
    # add repetition symbol as soon as we have ASG acoustic model
    token_dict.add_entry("<1>")
    # create Kenlm language model
    lm = KenLM(os.path.join(data_path, "lm.arpa"), word_dict)

    # test LM
    sentence = ["the", "cat", "sat", "on", "the", "mat"]
    # start LM with nothing, get its current state
    lm_state = lm.start(False)
    total_score = 0
    lm_score_target = [-1.05971, -4.19448, -3.33383, -2.76726, -1.16237, -4.64589]
    # iterate over words in the sentence
    for i in range(len(sentence)):
        # score lm, taking current state and index of the word
        # returns new state and score for the word
        lm_state, lm_score = lm.score(lm_state, word_dict.get_index(sentence[i]))
        assert_near(lm_score, lm_score_target[i], 1e-5)
        # add score of the current word to the total sentence score
        total_score += lm_score
    # move lm to the final state, the score returned is for eos
    lm_state, lm_score = lm.finish(lm_state)
    total_score += lm_score
    assert_near(total_score, -19.5123, 1e-5)

    # build trie
    # Trie is necessary to do beam-search decoding with word-level lm
    # We restrict our search only to the words from the lexicon
    # Trie is constructed from the lexicon, each node is a token
    # path from the root to a leaf corresponds to a word spelling in the lexicon

    # get silence index
    sil_idx = token_dict.get_index("|")

    # get unknown word index
    unk_idx = word_dict.get_index("<unk>")

    # create the trie, specifying how many tokens we have and silence index
    trie = Trie(token_dict.index_size(), sil_idx)
    start_state = lm.start(False)

    # use heuristic for the trie, called smearing:
    # predict lm score for each word in the lexicon, set this score to a leaf
    # (we predict lm score for each word as each word starts a sentence)
    # word score of a leaf is propagated up to the root to have some proxy score
    # for any intermediate path in the trie
    # SmearingMode defines the function how to process scores
    # in a node came from the children nodes:
    # could be max operation or logadd or none
    for word, spellings in lexicon.items():
        usr_idx = word_dict.get_index(word)
        _, score = lm.score(start_state, usr_idx)
        for spelling in spellings:
            # max_reps should be 1; using 0 here to match DecoderTest bug
            spelling_idxs = tkn_to_idx(spelling, token_dict, 1)
            trie.insert(spelling_idxs, usr_idx, score)

    trie.smear(SmearingMode.MAX)

    # check that trie is built in consistency with c++
    trie_score_target = [-1.05971, -2.87742, -2.64553, -3.05081, -1.05971, -3.08968]
    for i in range(len(sentence)):
        word = sentence[i]
        # max_reps should be 1; using 0 here to match DecoderTest bug
        word_tensor = tkn_to_idx([c for c in word], token_dict, 1)
        node = trie.search(word_tensor)
        assert_near(node.max_score, trie_score_target[i], 1e-5)


    # Define decoder options:
    # LexiconDecoderOptions (beam_size, token_beam_size, beam_threshold, lm_weight,
    #                 word_score, unk_score, sil_score,
    #                 log_add, criterion_type (ASG or CTC))
    opts = LexiconDecoderOptions(
        2500, 25000, 100.0, 2.0, 2.0, -math.inf, -1, False, CriterionType.ASG
    )


    # define lexicon beam-search decoder with word-level lm
    # LexiconDecoder(decoder options, trie, lm, silence index,
    #                blank index (for CTC), unk index,
    #                transitiona matrix, is token-level lm)
    decoder = LexiconDecoder(opts, trie, lm, sil_idx, -1, unk_idx, transitions, False)


    # run decoding
    # decoder.decode(emissions, Time, Ntokens)
    # result is a list of sorted hypothesis, 0-index is the best hypothesis
    # each hypothesis is a struct with "score" and "words" representation
    # in the hypothesis and the "tokens" representation
    results = decoder.decode(emissions.ctypes.data, T, N)


    print(f"Decoding complete, obtained {len(results)} results")
    print("Showing top 5 results:")
    for i in range(min(5, len(results))):
        prediction = []
        for idx in results[i].tokens:
            if idx < 0:
                break
            prediction.append(token_dict.get_entry(idx))
        prediction = " ".join(prediction)
        print(
            f"score={results[i].score} amScore={results[i].amScore} lmScore={results[i].lmScore} prediction='{prediction}'"
        )

    assert len(results) == 16
    hyp_score_target = [-284.0998, -284.108, -284.119, -284.127, -284.296]
    for i in range(min(5, len(results))):
        assert_near(results[i].score, hyp_score_target[i], 1e-3)
'''