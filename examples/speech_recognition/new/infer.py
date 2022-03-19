#!/usr/bin/env python -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import ast
import hashlib
import logging
import os
import shutil
import sys
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import editdistance
import torch
import torch.distributed as dist
from examples.speech_recognition.new.decoders.decoder_config import (
    DecoderConfig,
    FlashlightDecoderConfig,
)
from examples.speech_recognition.new.decoders.decoder import Decoder
from fairseq import checkpoint_utils, distributed_utils, progress_bar, tasks, utils
from fairseq.data.data_utils import post_process
from fairseq.dataclass.configs import (
    CheckpointConfig,
    CommonConfig,
    CommonEvalConfig,
    DatasetConfig,
    DistributedTrainingConfig,
    FairseqDataclass,
)
from fairseq.logging.meters import StopwatchMeter, TimeMeter
from fairseq.logging.progress_bar import BaseProgressBar
from fairseq.models.fairseq_model import FairseqModel
from omegaconf import OmegaConf

import hydra
from hydra.core.config_store import ConfigStore

logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config_path = Path(__file__).resolve().parent / "conf"


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


@dataclass
class InferConfig(FairseqDataclass):
    task: Any = None
    decoding: DecodingConfig = DecodingConfig()
    common: CommonConfig = CommonConfig()
    common_eval: CommonEvalConfig = CommonEvalConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    distributed_training: DistributedTrainingConfig = DistributedTrainingConfig()
    dataset: DatasetConfig = DatasetConfig()
    is_ax: bool = field(
        default=False,
        metadata={
            "help": "if true, assumes we are using ax for tuning and returns a tuple for ax to consume"
        },
    )


def reset_logging():
    root = logging.getLogger()
    for handler in root.handlers:
        root.removeHandler(handler)
    root.setLevel(os.environ.get("LOGLEVEL", "INFO").upper())
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(handler)



########################################################################
###################       for rescoring model        ###################
########################################################################

import numpy
from fairseq.data import Dictionary
from fairseq.models.fconv_lm import FConvLanguageModel
from fairseq.models.transformer_lm import TransformerLanguageModel

def load_rescoring_model(lm_path, model_type, dict_path):
    path, checkpoint = os.path.split(lm_path)
    if model_type == "convlm":
        model_handle = FConvLanguageModel.from_pretrained(
            path, checkpoint, os.path.split(dict_path)[0]
        )
    elif model_type == "transformer":
        model_handle = TransformerLanguageModel.from_pretrained(
            path, checkpoint, os.path.split(dict_path)[0]
        )
    else:
        raise Exception(
            "Unsupported language model type: use 'convlm' or 'transformer' models"
        )
    model = model_handle.models[0].decoder.cuda()
    model.eval()
    # print(model)
    return model


def predict_batch_for_rescoring(sentences, model, fairseq_dict, max_len):
    encoded_input = []
    padded_input = []
    ppls = []

    total_loss = 0.0
    nwords = 0

    # batchify
    for sentence in sentences:
        encoded_input.append([fairseq_dict.index(token) for token in sentence])
        assert (
            len(encoded_input[-1]) <= max_len
        ), "Error in the input length, it should be less than max_len {}".format(
            max_len
        )
        if len(encoded_input[-1]) < max_len:
            padded_input.append(
                [fairseq_dict.eos()]
                + encoded_input[-1]
                + [fairseq_dict.eos()] * (max_len - len(encoded_input[-1]))
            )
        else:
            padded_input.append([fairseq_dict.eos()] + encoded_input[-1])
    x = torch.LongTensor(padded_input).cuda()

    # inference
    with torch.no_grad():
        y = model.forward(x)[0]
        if model.adaptive_softmax is not None:
            logprobs = (model.adaptive_softmax.get_log_prob(y, None).detach().cpu().numpy())
        else:
            logprobs = torch.nn.functional.log_softmax(y, 2).detach().cpu().numpy()

    '''
    (Pdb) sentences[0]
    ['i', 'am', 'willing', 'to', 'enter', 'into', 'competition', 'with', 'the', 'ancients', 'and', 
    'feel', 'able', 'to', 'surpass', 'them', 'for', 'since', 'those', 'early', 'days', 'in', 
    'which', 'i', 'made', 'the', 'medals', 'of', 'pope', 'clement', 'i', 'have', 'learned', 
    'so', 'much', 'that', 'i', 'can', 'now', 'produce', 'far', 'better', 'pieces', 'of', 
    'the', 'kind', 'i', 'think', 'i', 'can', 'also', 'outdo', 'the', 'coins', 'i', 'struck', 
    'for', 'duke', 'alessandro', 'which', 'are', 'still', 'held', 'in', 'high', 'esteem', 
    'in', 'like', 'manner', 'i', 'could', 'make', 'for', 'you', 'large', 'pieces', 'of', 
    'gold', 'and', 'silver', 'plate', 'as', 'i', 'did', 'so', 'often', 'for', 'that', 'noble', 
    'monarch', 'king', 'francis', 'of', 'france', 'thanks', 'to', 'the', 'great', 'conveniences', 
    'he', 'allowed', 'me', 'without', 'ever', 'losing', 'time', 'for', 'the', 'execution', 'of', 
    'colossal', 'statues', 'or', 'other', 'works', 'of', 'the', "sculptor's", 'craft']

    (Pdb) x[0]
    tensor([    2,    10,   134,  1376,     7,  1106,    64,  7431,    16,     4,
            10432,     5,   356,   433,     7, 15349,    53,    19,   264,   130,
            519,   233,     9,    34,    10,    93,     4, 16706,     6,  3404,
            9543,    10,    29,   765,    41,   105,    12,    10,    91,    65,
            2204,   194,   209,  1380,     6,     4,   292,    10,   124,    10,
            91,   226, 30331,     4,  8978,    10,   642,    19,  1299, 19243,
            34,    47,   140,   361,     9,   300,  4266,     9,    73,   430,
            10,    62,   129,    19,    17,   325,  1380,     6,   590,     5,
            1027,  2565,    18,    10,    79,    41,   336,    19,    12,   930,
            5009,   321,  3275,     6,   903,  1804,     7,     4,    99, 18188,
            11,   933,    40,   141,   161,  3092,    71,    19,     4,  3615,
                6,  8886,  6828,    46,    88,  1199,     6,     4, 32003,  2603,
                2], device='cuda:0')
    '''

    for index, input_i in enumerate(encoded_input):
        loss = numpy.sum(logprobs[index, numpy.arange(len(input_i)), input_i])
        loss += logprobs[index, len(input_i), fairseq_dict.eos()]
        # import pdb; pdb.set_trace()
        ppls.append(loss)

        total_loss += loss
        nwords += len(input_i) + 1
    return ppls, total_loss, nwords


class InferenceProcessor:
    cfg: InferConfig

    def __init__(self, cfg: InferConfig) -> None:
        self.cfg = cfg
        self.task = tasks.setup_task(cfg.task)

        # import pdb; pdb.set_trace()

        models, saved_cfg = self.load_model_ensemble()
        self.models = models
        self.saved_cfg = saved_cfg
        self.tgt_dict = self.task.target_dictionary

        self.task.load_dataset(
            self.cfg.dataset.gen_subset,
            task_cfg=saved_cfg.task,
        )
        self.generator = Decoder(cfg.decoding, self.tgt_dict)
        self.gen_timer = StopwatchMeter()
        self.wps_meter = TimeMeter()
        self.num_sentences = 0
        self.total_errors = 0
        self.total_length = 0

        self.hypo_words_file = None
        self.hypo_units_file = None
        self.ref_words_file = None
        self.ref_units_file = None

        self.progress_bar = self.build_progress_bar()

        self.rescoring = cfg.decoding.rescoring
        self.rescoring_weight = cfg.decoding.rescoringweight
        self.rescoring_word_len_weight = cfg.decoding.rescoringwordlenweight


        # if self.rescoring:
        #     device = 'cuda'
        #     model_name = "gpt2"
        #     # model_name = "gpt2-large"
        #     from transformers import GPT2LMHeadModel, GPT2TokenizerFast, AutoModelForCausalLM, AutoTokenizer
        #     self.rescoring_model = (
        #         AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name, is_decoder=True)
        #         .to(device)
        #         .eval()
        #     )
        #     self.rescoring_tokenizer = AutoTokenizer.from_pretrained(model_name)
        #     self.rescoring_tokenizer.pad_token = self.rescoring_tokenizer.eos_token



        if self.rescoring : 
            # # my model
            # print('cfg.decoding.rescoringlmpath',cfg.decoding.rescoringlmpath)
            # path, checkpoint = os.path.split(cfg.decoding.rescoringlmpath)
            # overrides = {
            #     "task": 'language_modeling',
            #     "data": path,
            # }
            # logger.info("| loading rescoring lm model from {}".format(cfg.decoding.rescoringlmpath))
            # rescoring_models, rescoring_saved_cfg, rescoring_task = checkpoint_utils.load_model_ensemble_and_task(
            #     utils.split_paths(cfg.decoding.rescoringlmpath, separator="\\"),
            #     arg_overrides=overrides,
            #     strict=True,
            # )

            # self.rescoring_model = rescoring_models[0]
            # self.rescoring_model.eval().cuda()
            # self.rescoring_model.make_generation_fast_()
            # if rescoring_saved_cfg.common.fp16:
            #     self.rescoring_model.half()
            # self.rescoring_model = self.rescoring_model.decoder

            # dict_path = os.path.join(path,'dict.txt')
            # self.rescoring_dict = Dictionary.load(dict_path)

            # original code
            logger.info("| loading rescoring lm model from {}".format(cfg.decoding.rescoringlmpath))
            path, checkpoint = os.path.split(self.cfg.decoding.rescoringlmpath)
            dict_path = os.path.join(path,'dict.txt')
            self.rescoring_dict = Dictionary.load(dict_path)
            self.rescoring_model = load_rescoring_model(self.cfg.decoding.rescoringlmpath, 'transformer', dict_path)
            self.rescoring_model.eval().cuda()

            # # gpt model
            # model_name = "gpt2"
            # from transformers import GPT2LMHeadModel, GPT2TokenizerFast, AutoModelForCausalLM, AutoTokenizer

            # gpt_rescoring_model = (
            #     AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name, is_decoder=True)
            #     .to('cuda')
            #     .eval()
            # )
            # gpt_rescoring_tokenizer = AutoTokenizer.from_pretrained(model_name)
            # gpt_rescoring_tokenizer.pad_token = gpt_rescoring_tokenizer.eos_token


    def __enter__(self) -> "InferenceProcessor":
        if self.cfg.decoding.results_path is not None:
            self.hypo_words_file = self.get_res_file("hypo.word")
            self.hypo_units_file = self.get_res_file("hypo.units")
            self.ref_words_file = self.get_res_file("ref.word")
            self.ref_units_file = self.get_res_file("ref.units")
        return self

    def __exit__(self, *exc) -> bool:
        if self.cfg.decoding.results_path is not None:
            self.hypo_words_file.close()
            self.hypo_units_file.close()
            self.ref_words_file.close()
            self.ref_units_file.close()
        return False

    def __iter__(self) -> Any:
        for sample in self.progress_bar:
            if not self.cfg.common.cpu:
                sample = utils.move_to_cuda(sample)

            # Happens on the last batch.
            if "net_input" not in sample:
                continue
            yield sample

    def log(self, *args, **kwargs):
        self.progress_bar.log(*args, **kwargs)

    def print(self, *args, **kwargs):
        self.progress_bar.print(*args, **kwargs)

    def get_res_file(self, fname: str) -> None:
        fname = os.path.join(self.cfg.decoding.results_path, fname)
        if self.data_parallel_world_size > 1:
            fname = f"{fname}.{self.data_parallel_rank}"
        return open(fname, "w", buffering=1)

    def merge_shards(self) -> None:
        """Merges all shard files into shard 0, then removes shard suffix."""

        shard_id = self.data_parallel_rank
        num_shards = self.data_parallel_world_size

        if self.data_parallel_world_size > 1:

            def merge_shards_with_root(fname: str) -> None:
                fname = os.path.join(self.cfg.decoding.results_path, fname)
                logger.info("Merging %s on shard %d", fname, shard_id)
                base_fpath = Path(f"{fname}.0")
                with open(base_fpath, "a") as out_file:
                    for s in range(1, num_shards):
                        shard_fpath = Path(f"{fname}.{s}")
                        with open(shard_fpath, "r") as in_file:
                            for line in in_file:
                                out_file.write(line)
                        shard_fpath.unlink()
                shutil.move(f"{fname}.0", fname)

            dist.barrier()  # ensure all shards finished writing
            if shard_id == (0 % num_shards):
                merge_shards_with_root("hypo.word")
            if shard_id == (1 % num_shards):
                merge_shards_with_root("hypo.units")
            if shard_id == (2 % num_shards):
                merge_shards_with_root("ref.word")
            if shard_id == (3 % num_shards):
                merge_shards_with_root("ref.units")
            dist.barrier()

    def optimize_model(self, model: FairseqModel) -> None:
        model.make_generation_fast_()
        if self.cfg.common.fp16:
            model.half()
        if not self.cfg.common.cpu:
            model.cuda()

    def load_model_ensemble(self) -> Tuple[List[FairseqModel], FairseqDataclass]:
        arg_overrides = ast.literal_eval(self.cfg.common_eval.model_overrides)
        models, saved_cfg = checkpoint_utils.load_model_ensemble(
            utils.split_paths(self.cfg.common_eval.path, separator="\\"),
            arg_overrides=arg_overrides,
            task=self.task,
            suffix=self.cfg.checkpoint.checkpoint_suffix,
            strict=(self.cfg.checkpoint.checkpoint_shard_count == 1),
            num_shards=self.cfg.checkpoint.checkpoint_shard_count,
        )
        for model in models:
            self.optimize_model(model)
        return models, saved_cfg

    def get_dataset_itr(self, disable_iterator_cache: bool = False) -> None:
        return self.task.get_batch_iterator(
            dataset=self.task.dataset(self.cfg.dataset.gen_subset),
            max_tokens=self.cfg.dataset.max_tokens,
            max_sentences=self.cfg.dataset.batch_size,
            max_positions=(sys.maxsize, sys.maxsize),
            ignore_invalid_inputs=self.cfg.dataset.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=self.cfg.dataset.required_batch_size_multiple,
            seed=self.cfg.common.seed,
            num_shards=self.data_parallel_world_size,
            shard_id=self.data_parallel_rank,
            num_workers=self.cfg.dataset.num_workers,
            data_buffer_size=self.cfg.dataset.data_buffer_size,
            disable_iterator_cache=disable_iterator_cache,
        ).next_epoch_itr(shuffle=False)

    def build_progress_bar(
        self,
        epoch: Optional[int] = None,
        prefix: Optional[str] = None,
        default_log_format: str = "tqdm",
    ) -> BaseProgressBar:
        return progress_bar.progress_bar(
            iterator=self.get_dataset_itr(),
            log_format=self.cfg.common.log_format,
            log_interval=self.cfg.common.log_interval,
            epoch=epoch,
            prefix=prefix,
            tensorboard_logdir=self.cfg.common.tensorboard_logdir,
            default_log_format=default_log_format,
        )

    @property
    def data_parallel_world_size(self):
        if self.cfg.distributed_training.distributed_world_size == 1:
            return 1
        return distributed_utils.get_data_parallel_world_size()

    @property
    def data_parallel_rank(self):
        if self.cfg.distributed_training.distributed_world_size == 1:
            return 0
        return distributed_utils.get_data_parallel_rank()

    def process_sentence(
        self,
        sample: Dict[str, Any],
        hypo: Dict[str, Any],
        sid: int,
        batch_id: int,
    ) -> Tuple[int, int]:
        speaker = None  # Speaker can't be parsed from dataset.

        if "target_label" in sample:
            toks = sample["target_label"]
        else:
            toks = sample["target"]
        toks = toks[batch_id, :]

        # Processes hypothesis.
        hyp_pieces = self.tgt_dict.string(hypo["tokens"].int().cpu())
        if "words" in hypo:
            hyp_words = " ".join(hypo["words"])
        else:
            hyp_words = post_process(hyp_pieces, self.cfg.common_eval.post_process)
        
        # upper; necessary for LM traind with lower word 
        hyp_words = hyp_words.upper()

        # Processes target.
        target_tokens = utils.strip_pad(toks, self.tgt_dict.pad())
        tgt_pieces = self.tgt_dict.string(target_tokens.int().cpu())
        tgt_words = post_process(tgt_pieces, self.cfg.common_eval.post_process)

        # upper; necessary for LM traind with lower word 
        tgt_words = tgt_words.upper()

        if self.cfg.decoding.results_path is not None:
            print(f"{hyp_pieces} ({speaker}-{sid})", file=self.hypo_units_file)
            print(f"{hyp_words} ({speaker}-{sid})", file=self.hypo_words_file)
            print(f"{tgt_pieces} ({speaker}-{sid})", file=self.ref_units_file)
            print(f"{tgt_words} ({speaker}-{sid})", file=self.ref_words_file)

        if not self.cfg.common_eval.quiet:
            logger.info(f"HYPO : {hyp_words}")
            logger.info(f"TARG : {tgt_words}")
            logger.info("---------------------")

        hyp_words, tgt_words = hyp_words.split(), tgt_words.split()

        return editdistance.eval(hyp_words, tgt_words), len(tgt_words)

    def process_sample(self, sample: Dict[str, Any]) -> None:
        self.gen_timer.start()
        hypos = self.task.inference_step(
            generator=self.generator,
            models=self.models,
            sample=sample,
        )

        # if self.rescoring:
        #     reordered_hypos = []
        #     for nbest_hypos in hypos:
        #         inputs = []
        #         beams = []
        #         for hypo in nbest_hypos:
        #             inputs.append(' '.join(hypo['words']))
        #         encoded = self.rescoring_tokenizer(inputs, return_tensors='pt', padding=True, truncation=True)['input_ids'].to(self.rescoring_model.device)
        #         inputs_mask = (encoded != self.rescoring_tokenizer.pad_token_id)

        #         with torch.no_grad():
        #             output = self.rescoring_model(input_ids=encoded, attention_mask=inputs_mask)
        #             log_probs = torch.nn.functional.log_softmax(output.logits, dim=-1)
        #             target_log_probs = log_probs[:, :-1].gather(2, encoded[:, 1:].unsqueeze(2)).squeeze(2)
        #             neural_lm_score = torch.sum(target_log_probs * inputs_mask[:, 1:], dim=-1)
        #             rescored_results=(neural_lm_score)

        #         for hypo, rescored_result in zip(nbest_hypos, rescored_results):
        #             beams.append({
        #                 "tokens":hypo['tokens'], 
        #                 "score" : hypo['score'], 
        #                 "timesteps" : hypo['timesteps'], 
        #                 "words" : hypo['words'], 
        #                 "rescoring" : rescored_result,
        #                 "total_score" : hypo['score'] + self.rescoring_weight * rescored_result
        #                 })

        #         # Original Rescoring log P_{AM} (y|x) + \alpha1 log P_{LM1}(y) + \beta |y| + \alpha2 log P_{LM2}(y)
        #         # on the fly
        #         sorted_beams = sorted(beams, reverse=True, key=lambda x:x['total_score'])
        #         reordered_hypos.append(sorted_beams)
        #     hypos = reordered_hypos

        if self.rescoring:
            for i, nbest_hypos in enumerate(hypos):
                batch = []
                max_len = 0
                for j, n_th_hypo in enumerate(nbest_hypos):

                    sent = n_th_hypo['words']
                    score = n_th_hypo['score']
                    hypos[i][j]['wl_len'] = len(sent) + len("".join(sent))
                    sent = " ".join(sent).lower().split()

                    '''
                    (base) [irteam@adev-closer-pgpu02.clova text]$ head -10 librispeech-lm-norm.txt.lower.shuffle
                    so the reinforcements you were expecting won't get here tonight after all he remarked softly
                    why is she a saint omar illegitimate no saint omar sans reproche
                    death levels all he had no feeling that the cottagers standing at their garden gates were intruding their curiosity as was felt by susan's mother for one who thought this public tramp between a station and a church an outrage on her nobility
                    then he looked up again speaking in the same equal voice
                    he made confession of his sins and promised an entire amendment of life if the almighty would deliver him from his enemies and restore him to his throne
                    as soon as it found itself on firm ground it began to throw its legs out in all directions but toko held it fast by the halter
                    tra la chrees'mas day
                    purvis left his horse in the cool of the paraiso trees during the day and a peon brought it to the door after he had eaten a frugal dinner during which meal he attended far more to the wants of his child than to his own
                    his finger silently pointed her to withdraw
                    footnote twelve the name is corrupted as are all those handed down by the early historians

                    (Pdb) " ".join(sent)
                    "I AM WILLING TO ENTER INTO COMPETITION WITH THE ANCIENTS AND FEEL ABLE TO SURPASS THEM FOR SINCE THOSE EARLY DAYS IN WHICH 
                    I MADE THE MEDALS OF POPE CLEMENT I HAVE LEARNED SO MUCH THAT I CAN NOW PRODUCE FAR BETTER PIECES OF THE KIND I THINK I CAN ALSO 
                    OUTDO THE COINS I STRUCK FOR DUKE ALESSANDRO WHICH ARE STILL HELD IN HIGH ESTEEM IN LIKE MANNER I COULD MAKE FOR YOU LARGE PIECES 
                    OF GOLD AND SILVER PLATE AS I DID SO OFTEN FOR THAT NOBLE MONARCH KING FRANCIS OF FRANCE THANKS TO THE GREAT CONVENIENCES HE ALLOWED 
                    ME WITHOUT EVER LOSING TIME FOR THE EXECUTION OF COLOSSAL STATUES OR OTHER WORKS OF THE SCULPTOR'S CRAFT"

                    (Pdb) " ".join(sent).lower()
                    "i am willing to enter into competition with the ancients and feel able to surpass them for since those early days in which 
                    i made the medals of pope clement i have learned so much that i can now produce far better pieces of the kind i think i can also 
                    outdo the coins i struck for duke alessandro which are still held in high esteem in like manner i could make for you large pieces 
                    of gold and silver plate as i did so often for that noble monarch king francis of france thanks to the great conveniences he allowed 
                    me without ever losing time for the execution of colossal statues or other works of the sculptor's craft"

                    (Pdb) " ".join(sent).lower().split()
                    ['i', 'am', 'willing', 'to', 'enter', 'into', 'competition', 'with', 'the', 'ancients', 'and', 'feel', 'able', 'to', 'surpass', 
                    'them', 'for', 'since', 'those', 'early', 'days', 'in', 'which', 'i', 'made', 'the', 'medals', 'of', 'pope', 'clement', 'i', 
                    'have', 'learned', 'so', 'much', 'that', 'i', 'can', 'now', 'produce', 'far', 'better', 'pieces', 'of', 'the', 'kind', 'i', 
                    'think', 'i', 'can', 'also', 'outdo', 'the', 'coins', 'i', 'struck', 'for', 'duke', 'alessandro', 'which', 'are', 'still', 
                    'held', 'in', 'high', 'esteem', 'in', 'like', 'manner', 'i', 'could', 'make', 'for', 'you', 'large', 'pieces', 'of', 'gold', 
                    'and', 'silver', 'plate', 'as', 'i', 'did', 'so', 'often', 'for', 'that', 'noble', 'monarch', 'king', 'francis', 'of', 'france', 
                    'thanks', 'to', 'the', 'great', 'conveniences', 'he', 'allowed', 'me', 'without', 'ever', 'losing', 'time', 'for', 'the', 'execution', 
                    'of', 'colossal', 'statues', 'or', 'other', 'works', 'of', 'the', "sculptor's", 'craft']
                    '''

                    # import pdb; pdb.set_trace()

                    batch.append(sent)

                max_len = len(sorted(batch, key=lambda x: len(x))[-1])
                # ppls, loss_batch, nwords_batch = predict_batch_for_rescoring(batch, self.rescoring_model.decoder, self.rescoring_dict, max_len)
                ppls, loss_batch, nwords_batch = predict_batch_for_rescoring(batch, self.rescoring_model, self.rescoring_dict, max_len)

                '''
                # half()
                if hypothesis is upper case, we should lower sentence for lower word transformer
                (Pdb) ppls
                [-39.53208, -39.53209, -39.53208, -39.67231, -39.53208, -39.53209, -39.67231, -39.53209, -39.53208, -39.53209, 
                -39.53208, -39.53209, -39.53208, -39.53209, -39.53208, -39.67231, -39.53208, -39.53209, -39.53208, -39.53209, 
                -39.53208, -39.53209, -39.53208, -39.53209, -39.67231, -39.67231, -39.53208, -39.53209, -39.389668, -39.67231, 
                -39.53208, -39.53209, -39.67231, -39.53209, -39.53208, -39.53209, -39.67231, -39.67231, -39.53208, -39.53209, 
                -39.53208, -39.53209, -39.53208, -39.67231, -39.53208, -39.53209, -39.53208, -39.53209, -39.53208, -39.389675]

                (Pdb) ppls
                [-449.8855, -457.82663, -455.54514, -454.2732, -452.20465, -462.68192, -462.01917, -460.2545, -475.82095, -451.28564, 
                -462.25165, -457.00897, -455.3532, -474.80624, -459.50165, -456.44742, -454.30377, -470.56497, -463.97653, -463.7325, 
                -463.35968, -456.01425, -464.98883, -458.7563, -456.29196, -460.85138, -457.89813, -455.57092, -459.9813, -464.54758, 
                -461.90475, -454.58594, -456.65198, -470.7382, -453.04877, -471.61276, -458.97662, -458.7723, -453.94193, -464.0407, 
                -456.44324, -472.36307, -466.57474, -463.2118, -465.00317, -456.67255, -465.95905, -453.54788, -450.17545, -449.16412]

                (Pdb) predict_batch_for_rescoring(batch, self.rescoring_model.half(), self.rescoring_dict, max_len)[0]
                [-450.0, -458.0, -455.8, -454.0, -452.2, -462.8, -462.0, -460.2, -476.0, -451.5, -462.2, -457.0, -455.5, -475.0, -459.8, 
                -456.5, -454.5, -470.8, -464.0, -463.8, -463.5, -456.2, -465.0, -458.8, -456.2, -460.8, -458.0, -455.8, -460.0, -464.8, 
                -462.0, -454.5, -456.8, -470.8, -453.0, -471.8, -459.0, -458.8, -454.0, -464.2, -456.5, -472.5, -466.8, -463.2, 
                -464.8, -456.8, -466.0, -453.5, -450.2, -449.2]
                '''

                for j, (n_th_hypo, ppl) in enumerate(zip(nbest_hypos,ppls)):
                    hypos[i][j]['rescoring_lm_ppl'] = ppl
                    # hypos[i][j]['total_score'] = n_th_hypo['score'] + self.rescoring_weight * ppl
                    hypos[i][j]['total_score'] = n_th_hypo['am_score'] + self.rescoring_weight * ppl + self.rescoring_word_len_weight * n_th_hypo['wl_len']
                
                # import pdb; pdb.set_trace()

                hypos[i] = sorted(nbest_hypos, key=lambda x: -x["total_score"])
                # hypos[i] = sorted(nbest_hypos, key=lambda x: -x["rescoring_lm_ppl"])

        num_generated_tokens = sum(len(h[0]["tokens"]) for h in hypos)
        self.gen_timer.stop(num_generated_tokens)
        self.wps_meter.update(num_generated_tokens)

        for batch_id, sample_id in enumerate(sample["id"].tolist()):
            errs, length = self.process_sentence(
                sample=sample,
                sid=sample_id,
                batch_id=batch_id,
                hypo=hypos[batch_id][0],
            )
            self.total_errors += errs
            self.total_length += length

        self.log({"wps": round(self.wps_meter.avg)})
        if "nsentences" in sample:
            self.num_sentences += sample["nsentences"]
        else:
            self.num_sentences += sample["id"].numel()

    def log_generation_time(self) -> None:
        logger.info(
            "Processed %d sentences (%d tokens) in %.1fs %.2f "
            "sentences per second, %.2f tokens per second)",
            self.num_sentences,
            self.gen_timer.n,
            self.gen_timer.sum,
            self.num_sentences / (self.gen_timer.sum + 1e-6),
            1.0 / (self.gen_timer.avg + 1e-6),
        )


def parse_wer(wer_file: Path) -> float:
    with open(wer_file, "r") as f:
        return float(f.readline().strip().split(" ")[1])


def get_wer_file(cfg: InferConfig) -> Path:
    """Hashes the decoding parameters to a unique file ID."""
    base_path = "wer"
    if cfg.decoding.results_path is not None:
        base_path = os.path.join(cfg.decoding.results_path, base_path)

    if cfg.decoding.unique_wer_file:
        yaml_str = OmegaConf.to_yaml(cfg.decoding)
        fid = int(hashlib.md5(yaml_str.encode("utf-8")).hexdigest(), 16)
        return Path(f"{base_path}.{fid % 1000000}")
    else:
        return Path(base_path)


def main(cfg: InferConfig) -> float:
    """Entry point for main processing logic.

    Args:
        cfg: The inferance configuration to use.
        wer: Optional shared memory pointer for returning the WER. If not None,
            the final WER value will be written here instead of being returned.

    Returns:
        The final WER if `wer` is None, otherwise None.
    """

    yaml_str, wer_file = OmegaConf.to_yaml(cfg.decoding), get_wer_file(cfg)

    # Validates the provided configuration.
    if cfg.dataset.max_tokens is None and cfg.dataset.batch_size is None:
        cfg.dataset.max_tokens = 4000000
    if not cfg.common.cpu and not torch.cuda.is_available():
        raise ValueError("CUDA not found; set `cpu=True` to run without CUDA")

    logger.info(cfg.common_eval.path)
    # import pdb; pdb.set_trace()


    with InferenceProcessor(cfg) as processor:
        for sample in processor:
            processor.process_sample(sample)

        processor.log_generation_time()

        if cfg.decoding.results_path is not None:
            processor.merge_shards()

        errs_t, leng_t = processor.total_errors, processor.total_length

        if cfg.common.cpu:
            logger.warning("Merging WER requires CUDA.")
        elif processor.data_parallel_world_size > 1:
            stats = torch.LongTensor([errs_t, leng_t]).cuda()
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            errs_t, leng_t = stats[0].item(), stats[1].item()

        wer = errs_t * 100.0 / leng_t

        if distributed_utils.is_master(cfg.distributed_training):
            with open(wer_file, "w") as f:
                f.write(
                    (
                        f"WER: {wer}\n"
                        f"err / num_ref_words = {errs_t} / {leng_t}\n\n"
                        f"{yaml_str}"
                    )
                )

        return wer


@hydra.main(config_path=config_path, config_name="infer")
def hydra_main(cfg: InferConfig) -> Union[float, Tuple[float, Optional[float]]]:
    container = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    cfg = OmegaConf.create(container)
    OmegaConf.set_struct(cfg, True)

    if cfg.common.reset_logging:
        reset_logging()
    
    # import pdb;pdb.set_trace()

    utils.import_user_module(cfg.common)

    # logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    wer = float("inf")

    try:
        if cfg.common.profile:
            with torch.cuda.profiler.profile():
                with torch.autograd.profiler.emit_nvtx():
                    distributed_utils.call_main(cfg, main)
        else:
            distributed_utils.call_main(cfg, main)

        wer = parse_wer(get_wer_file(cfg))
    except BaseException as e:  # pylint: disable=broad-except
        if not cfg.common.suppress_crashes:
            raise
        else:
            logger.error("Crashed! %s", str(e))

    logger.info("Word error rate: %.4f", wer)
    if cfg.is_ax:
        return wer, None

    return wer


def cli_main() -> None:
    try:
        from hydra._internal.utils import (
            get_args,
        )  # pylint: disable=import-outside-toplevel

        cfg_name = get_args().config_name or "infer"
    except ImportError:
        logger.warning("Failed to get config name from hydra args")
        cfg_name = "infer"

    cs = ConfigStore.instance()
    cs.store(name=cfg_name, node=InferConfig)

    for k in InferConfig.__dataclass_fields__:
        if is_dataclass(InferConfig.__dataclass_fields__[k].type):
            v = InferConfig.__dataclass_fields__[k].default
            cs.store(name=k, node=v)

    hydra_main()  # pylint: disable=no-value-for-parameter


if __name__ == "__main__":
    cli_main()
