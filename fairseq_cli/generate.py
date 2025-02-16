#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate pre-processed data with a trained model.
"""

import ast
import logging
import math
import os
import sys
from argparse import Namespace
from itertools import chain

import numpy as np
import torch
from omegaconf import DictConfig

from fairseq import checkpoint_utils, options, scoring, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter

from pdb import set_trace as Tra


def main(cfg: DictConfig):

    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    assert cfg.common_eval.path is not None, "--path required for generation!"
    assert (
        not cfg.generation.sampling or cfg.generation.nbest == cfg.generation.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
        cfg.generation.replace_unk is None or cfg.dataset.dataset_impl == "raw"
    ), "--replace-unk requires a raw text dataset (--dataset-impl=raw)"

    if cfg.common_eval.results_path is not None:
        os.makedirs(cfg.common_eval.results_path, exist_ok=True)
        output_path = os.path.join(
            cfg.common_eval.results_path,
            "generate-{}.txt".format(cfg.dataset.gen_subset),
        )
        with open(output_path, "w", buffering=1, encoding="utf-8") as h:
            return _main(cfg, h)
    else:
        return _main(cfg, sys.stdout)


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.eos}


def _main(cfg: DictConfig, output_file):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=output_file,
    )
    logger = logging.getLogger("fairseq_cli.generate")

    utils.import_user_module(cfg.common)

    if cfg.dataset.max_tokens is None and cfg.dataset.batch_size is None:
        cfg.dataset.max_tokens = 12000
    logger.info(cfg)

    # Fix seed for stochastic decoding
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    # Load dataset splits
    task = tasks.setup_task(cfg.task)

    # Set dictionaries
    try:
        src_dict = getattr(task, "source_dictionary", None)
    except NotImplementedError:
        src_dict = None
        
    tgt_dict = task.target_dictionary

    overrides = ast.literal_eval(cfg.common_eval.model_overrides)

    # Load ensemble
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    models, saved_cfg = checkpoint_utils.load_model_ensemble(
        utils.split_paths(cfg.common_eval.path),
        arg_overrides=overrides,
        task=task,
        suffix=cfg.checkpoint.checkpoint_suffix,
        strict=(cfg.checkpoint.checkpoint_shard_count == 1),
        num_shards=cfg.checkpoint.checkpoint_shard_count,
    )

    # Tra()

    # loading the dataset should happen after the checkpoint has been loaded so we can give it the saved task config
    task.load_dataset(cfg.dataset.gen_subset, task_cfg=saved_cfg.task)

    '''
    (Pdb) cfg.keys()
    dict_keys(['_name', 'common', 'common_eval', 'distributed_training', 'dataset', 'optimization', 'checkpoint', 'bmuf', 
    'generation', 'eval_lm', 'interactive', 'model', 'task', 'criterion', 'optimizer', 'lr_scheduler', 'scoring', 
    'bpe', 'tokenizer', 'ema', 'simul_type'])

    (Pdb) cfg.generation
    {'_name': None, 'beam': 1, 'nbest': 1, 'max_len_a': 0.0, 'max_len_b': 200, 'min_len': 1, 'match_source_len': False, 
    'unnormalized': False, 'no_early_stop': False, 'no_beamable_mm': False, 'lenpen': 1.0, 'unkpen': 0.0, 'replace_unk': None, 
    'sacrebleu': False, 'score_reference': False, 'prefix_size': 0, 'no_repeat_ngram_size': 0, 'sampling': False, 
    'sampling_topk': -1, 'sampling_topp': -1.0, 'constraints': None, 'temperature': 1.0, 'diverse_beam_groups': -1, 
    'diverse_beam_strength': 0.5, 'diversity_rate': -1.0, 'print_alignment': None, 'print_step': False, 'lm_path': None, 
    'lm_weight': 0.0, 'iter_decode_eos_penalty': 0.0, 'iter_decode_max_iter': 10, 'iter_decode_force_max_iter': False, 
    'iter_decode_with_beam': 1, 'iter_decode_with_external_reranker': False, 'retain_iter_history': False, 'retain_dropout': False, 
    'retain_dropout_modules': None, 'decoding_format': None, 'no_seed_provided': False, 'eos_token': None}
    '''

    if cfg.generation.lm_path is not None:
        overrides["data"] = cfg.task.data

        try:
            lms, _ = checkpoint_utils.load_model_ensemble(
                [cfg.generation.lm_path], arg_overrides=overrides, task=None
            )
        except:
            logger.warning(
                f"Failed to load language model! Please make sure that the language model dict is the same "
                f"as target dict and is located in the data dir ({cfg.task.data})"
            )
            raise

        assert len(lms) == 1
    else:
        lms = [None]

    # Optimize ensemble for generation
    for model in chain(models, lms):
        if model is None:
            continue
        if cfg.common.fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(cfg.generation.replace_unk)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(cfg.dataset.gen_subset),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=utils.resolve_max_positions(
            task.max_positions(), *[m.max_positions() for m in models]
        ),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_shards=cfg.distributed_training.distributed_world_size,
        shard_id=cfg.distributed_training.distributed_rank,
        num_workers=cfg.dataset.num_workers,
        data_buffer_size=cfg.dataset.data_buffer_size,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
    )

    # Initialize generator
    gen_timer = StopwatchMeter()

    extra_gen_cls_kwargs = {"lm_model": lms[0], "lm_weight": cfg.generation.lm_weight}
    generator = task.build_generator(
        models, cfg.generation, extra_gen_cls_kwargs=extra_gen_cls_kwargs
    )

    # Handle tokenization and BPE
    tokenizer = task.build_tokenizer(cfg.tokenizer)
    bpe = task.build_bpe(cfg.bpe)

    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    # Tra()
    '''
    (Pdb) bpe
    (Pdb) tokenizer
    '''

    scorer = scoring.build_scorer(cfg.scoring, tgt_dict)

    # check tgt_dict once more
    if tgt_dict==None:
        try:
            tgt_dict = task.target_dictionary
        except:
            pass

    '''
    (Pdb) cfg.dataset.gen_subset; cfg.dataset.max_tokens; cfg.dataset.batch_size; cfg.dataset.skip_invalid_size_inputs_valid_test;
    'dev_other'
    4000000
    False

    (Pdb) cfg.scoring;
    {'_name': 'wer', 'wer_tokenizer': 'none', 'wer_remove_punct': False, 'wer_char_level': False, 'wer_lowercase': False}

    (Pdb) cfg.tokenizer
    '''
    
    # if cfg.criterion._name == 'ctc':
    #     total_len = list()
    #     total_dist = list()
    #     total_dist_ = list()
    #     logging_outputs = list()
    #     from fairseq.logging.meters import safe_round

    #     # build criterion
    #     criterion = task.build_criterion(cfg.criterion)

    #     for sample in progress:
    #         sample = utils.move_to_cuda(sample) if use_cuda else sample
    #         if "net_input" not in sample:
    #             continue
    #         hypos = task.valid_step(sample, models[0], criterion)
    #         '''
    #         (Pdb) hypos
    #         (tensor(141.6250, device='cuda:0'), 7, {'loss': 141.62496948242188, 'ntokens': 3411, 'nsentences': 7, 'sample_size': 7, 
    #         'wv_errors': 33, 'w_errors': 33, 'w_total': 600, 'c_errors': 38, 'c_total': 3411})
    #         '''
    #         total_len.append(hypos[2]['w_total'])
    #         total_dist.append(hypos[2]['w_errors'])
    #         total_dist_.append(hypos[2]['wv_errors'])
    #         # logging_outputs.append(hypos)
    #         # task.reduce_metrics(logging_outputs, criterion)
    #     # WER = sum(total_dist)/sum(total_len)
    #     wer = safe_round(sum(total_dist) * 100.0 / sum(total_len), 3) if sum(total_len) > 0 else float("nan")
    #     raw_wer = safe_round(sum(total_dist_) * 100.0 / sum(total_len), 3) if sum(total_len) > 0 else float("nan")
    #     print('WER is {} <== ( safe_round(sum(total_dist) * 100.0 / sum(total_len), 3) if sum(total_len) > 0 else float("nan") )'.format(wer))
    #     print('RAW WER is {} <== ( safe_round(sum(total_dist_) * 100.0 / sum(total_len), 3) if sum(total_len) > 0 else float("nan") )'.format(raw_wer))
    #     return wer

    num_sentences = 0
    has_target = True
    wps_meter = TimeMeter()
    for sample in progress:
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if "net_input" not in sample:
            continue

        prefix_tokens = None
        if cfg.generation.prefix_size > 0:
            prefix_tokens = sample["target"][:, : cfg.generation.prefix_size]

        constraints = None
        if "constraints" in sample:
            constraints = sample["constraints"]

        ###################################################################################################
        ############################################ Generation ###########################################
        ###################################################################################################

        gen_timer.start()


        with torch.no_grad():
            hypos = task.inference_step(
                generator,
                models,
                sample,
                prefix_tokens=prefix_tokens,
                constraints=constraints,
            )

            # net_output = models[0](**sample["net_input"])
            # hypos2 = greedy_decoding(task, models[0], sample, net_output)

        # Tra()

        num_generated_tokens = sum(len(h[0]["tokens"]) for h in hypos)
        gen_timer.stop(num_generated_tokens)

        ###################################################################################################

        for i, sample_id in enumerate(sample["id"].tolist()):
            has_target = sample["target"] is not None

            # Remove padding
            if "src_tokens" in sample["net_input"]:
                src_tokens = utils.strip_pad(
                    sample["net_input"]["src_tokens"][i, :], tgt_dict.pad()
                )
            else:
                src_tokens = None

            '''
            <<< ================== w2v2_seq2seq debugging ================== >>>

            (Pdb) tgt_dict
            <fairseq.data.dictionary.Dictionary object at 0x7f492443cf40>

            (Pdb) task; task.target_dictionary; len(task.target_dictionary);
            <fairseq.tasks.audio_finetuning.AudioFinetuningTask object at 0x7f492441e580>
            <fairseq.data.dictionary.Dictionary object at 0x7f492443cf40>
            10001

            (Pdb) sample.keys(); sample["net_input"].keys();
            dict_keys(['id', 'net_input', 'target_lengths', 'ntokens', 'target'])
            dict_keys(['source', 'padding_mask', 'prev_output_tokens'])

            (Pdb) sample['net_input']['source'].size(); sample['target'].size();
            torch.Size([7, 562480])
            torch.Size([7, 90])

            (Pdb) len(hypos); len(hypos[0]); hypos[0][0].keys();
            7
            1
            dict_keys(['tokens', 'score', 'attention', 'alignment', 'positional_scores'])

            ##### it was bug
            (Pdb) src_tokens # None
            (Pdb) tgt_dict.pad()
            *** AttributeError: 'NoneType' object has no attribute 'pad'
            '''

            target_tokens = None
            if has_target:
                target_tokens = (
                    utils.strip_pad(sample["target"][i, :], tgt_dict.pad()).int().cpu()
                )

            # Either retrieve the original sentences or regenerate them from tokens.
            if align_dict is not None:
                src_str = task.dataset(cfg.dataset.gen_subset).src.get_original_text(
                    sample_id
                )
                target_str = task.dataset(cfg.dataset.gen_subset).tgt.get_original_text(
                    sample_id
                )
            else:
                if src_dict is not None:
                    src_str = src_dict.string(src_tokens, cfg.common_eval.post_process)
                else:
                    src_str = ""
                if has_target:
                    target_str = tgt_dict.string(
                        target_tokens,
                        cfg.common_eval.post_process,
                        escape_unk=True,
                        extra_symbols_to_ignore=get_symbols_to_strip_from_output(
                            generator
                        ),
                    )

            src_str = decode_fn(src_str)
            if has_target:
                target_str = decode_fn(target_str)

            if not cfg.common_eval.quiet:
                if src_dict is not None:
                    print("S-{}\t{}".format(sample_id, src_str), file=output_file)
                if has_target:
                    ####
                    print("T-{}\t{}".format(sample_id, target_str), file=output_file)

            # Process top predictions
            for j, hypo in enumerate(hypos[i][: cfg.generation.nbest]):
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo["tokens"].int().cpu(),
                    src_str=src_str,
                    alignment=hypo["alignment"],
                    align_dict=align_dict,
                    tgt_dict=tgt_dict,
                    remove_bpe=cfg.common_eval.post_process,
                    extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
                )

                # Tra()

                '''
                (Pdb) hypo["tokens"]; hypo_tokens
                tensor([  16, 5311,   38,   12,   21, 1237,   30,    8, 6616,   73,   50,  318,
                        1136,   30,    8,  429,  803,   42,    4,  492,  137,   30,   59,   23,
                        74,  260, 4060,  609,  232, 7871,   27,   17, 1857,    9,    6,   16,
                        7970, 6808,   27,   17, 1857,    9,    6,   16, 7970, 6808,   27,   17,
                        1857,    9,    6,   16, 7970,  137,   30,   59,   23,   74,  260, 4060,
                        609,  232, 7871,   27,   17, 1857,    9,   13, 1826,    8,  441,    7,
                        223,   57, 4060,  609,  232, 7871,   27,   17,   15,   30,   59,   23,
                        74,  260, 4060,  609,  232, 7871,   27,   17, 1857,    9,   13, 1826,
                        8,  441,    7,  223,   57, 4060,  609,  232, 7871,   27,   17, 1857,
                        9,   13, 1826,    8,  441,    7,  223,   57, 4060,  609,  232, 7871,
                        27,   17, 1857,    9,   13, 1826,    8,  441,    7,  223,   57, 4060,
                        622,  508,  609,  232, 7871,   27,   17, 1857,    9,   13, 1826,    8,
                        441,    7,  223,   57, 4060,  609,  232, 7871,   27,   17, 1857,    9,
                        13, 1826,    8,  441,    7,  223,   57, 4060,  622,  508,  609,  232,
                        7871,   27,   17, 1857,    9,   13, 1826,    8,  441,    7,  223,   57,
                        4060, 5779,    9,   13,   15,   30,   59,   23,   74,  260, 4060, 5779,
                        9,   13,   23,   74,  260, 4060, 5779,    9,    2], device='cuda:0')

                tensor([10001, 10002, 10003,  1109, 10004, 10005,   785,   202, 10006,   343,
                        10007, 10008,   785,   202, 10009, 10010, 10011,  2758, 10012, 10013,
                        785, 10014,   381, 10015, 10016, 10017, 10018, 10019, 10020, 10021,
                        10022, 10023, 10001, 10024, 10025, 10021, 10022, 10023, 10001, 10024,
                        10025, 10021, 10022, 10023, 10001, 10024, 10013,   785, 10014,   381,
                        10015, 10016, 10017, 10018, 10019, 10020, 10021, 10022, 10026, 10027,
                        202, 10028,   879,  3057, 10029, 10017, 10018, 10019, 10020, 10021,
                        424,   785, 10014,   381, 10015, 10016, 10017, 10018, 10019, 10020,
                        10021, 10022, 10026, 10027,   202, 10028,   879,  3057, 10029, 10017,
                        10018, 10019, 10020, 10021, 10022, 10026, 10027,   202, 10028,   879,
                        3057, 10029, 10017, 10018, 10019, 10020, 10021, 10022, 10026, 10027,
                        202, 10028,   879,  3057, 10029, 10030, 10018, 10019, 10020, 10021,
                        10022, 10026, 10027,   202, 10028,   879,  3057, 10029, 10017, 10018,
                        10019, 10020, 10021, 10022, 10026, 10027,   202, 10028,   879,  3057,
                        10029, 10030, 10018, 10019, 10020, 10021, 10022, 10026, 10027,   202,
                        10028,   879,  3057, 10029, 10017, 10031, 10026,   424,   785, 10014,
                        381, 10015, 10016, 10017, 10031, 10026,   381, 10015, 10016, 10017,
                        10031,     2], dtype=torch.int32)
                '''

                detok_hypo_str = decode_fn(hypo_str)
                if not cfg.common_eval.quiet:
                    score = hypo["score"] / math.log(2)  # convert to base 2
                    # original hypothesis (after tokenization and BPE)
                    ####
                    print(
                        "H-{}\t{}\t{}".format(sample_id, score, hypo_str),
                        file=output_file,
                    )
                    # detokenized hypothesis
                    ####
                    print(
                        "D-{}\t{}\t{}".format(sample_id, score, detok_hypo_str),
                        file=output_file,
                    )
                    ####
                    print(
                        "P-{}\t{}".format(
                            sample_id,
                            " ".join(
                                map(
                                    lambda x: "{:.4f}".format(x),
                                    # convert from base e to base 2
                                    hypo["positional_scores"]
                                    .div_(math.log(2))
                                    .tolist(),
                                )
                            ),
                        ),
                        file=output_file,
                    )

                    if cfg.generation.print_alignment == "hard":
                        print(
                            "A-{}\t{}".format(
                                sample_id,
                                " ".join(
                                    [
                                        "{}-{}".format(src_idx, tgt_idx)
                                        for src_idx, tgt_idx in alignment
                                    ]
                                ),
                            ),
                            file=output_file,
                        )
                    if cfg.generation.print_alignment == "soft":
                        print(
                            "A-{}\t{}".format(
                                sample_id,
                                " ".join(
                                    [",".join(src_probs) for src_probs in alignment]
                                ),
                            ),
                            file=output_file,
                        )

                    if cfg.generation.print_step:
                        print(
                            "I-{}\t{}".format(sample_id, hypo["steps"]),
                            file=output_file,
                        )

                    if cfg.generation.retain_iter_history:
                        for step, h in enumerate(hypo["history"]):
                            _, h_str, _ = utils.post_process_prediction(
                                hypo_tokens=h["tokens"].int().cpu(),
                                src_str=src_str,
                                alignment=None,
                                align_dict=None,
                                tgt_dict=tgt_dict,
                                remove_bpe=None,
                            )
                            print(
                                "E-{}_{}\t{}".format(sample_id, step, h_str),
                                file=output_file,
                            )

                # Score only the top hypothesis
                if has_target and j == 0:
                    if (
                        align_dict is not None
                        or cfg.common_eval.post_process is not None
                    ):
                        # Convert back to tokens for evaluation with unk replacement and/or without BPE
                        target_tokens = tgt_dict.encode_line(
                            target_str, add_if_not_exist=True
                        )
                        hypo_tokens = tgt_dict.encode_line(
                            detok_hypo_str, add_if_not_exist=True
                        )
                    if hasattr(scorer, "add_string"):
                        scorer.add_string(target_str, detok_hypo_str)
                    else:
                        scorer.add(target_tokens, hypo_tokens)

                    '''
                    joint inter vanilla (아 이거 그냥 학습이 안된거네 ㅋㅋ)

                    (Pdb) detok_hypo_str
                    'HIS ABODE WHICH HE HAD FIXED AT A BOWERY OR COUNTRY SEAT AT A SHORT DISTANCE FROM THE CITY JUST 
                    AT WHAT IS NOW CALLED DUTCH STREET SOON ABOUNDED WITH PROOFS OF HIS INGENUITY PATENTED WITH PROOFS 
                    OF HIS INGENUITY PATENTED WITH PROOFS OF HIS INGENUITY JUST AT WHAT IS NOW CALLED DUTCH STREET SOON 
                    ABOUNDED WITH PROOFS THAT REQUIRED A HORSE TO WORK THEM DUTCH STREET SOON ABOUNDED WITH IT AT WHAT 
                    IS NOW CALLED DUTCH STREET SOON ABOUNDED WITH PROOFS THAT REQUIRED A HORSE TO WORK THEM DUTCH STREET 
                    SOON ABOUNDED WITH PROOFS THAT REQUIRED A HORSE TO WORK THEM DUTCH STREET SOON ABOUNDED WITH PROOFS 
                    THAT REQUIRED A HORSE TO WORK THEM DUTCHCOCK STREET SOON ABOUNDED WITH PROOFS THAT REQUIRED A HORSE 
                    TO WORK THEM DUTCH STREET SOON ABOUNDED WITH PROOFS THAT REQUIRED A HORSE TO WORK THEM DUTCHCOCK STREET 
                    SOON ABOUNDED WITH PROOFS THAT REQUIRED A HORSE TO WORK THEM DUTCH OVENS THAT IT AT WHAT IS NOW CALLED 
                    DUTCH OVENS THAT IS NOW CALLED DUTCH OVENS'

                    (Pdb) target_str
                    'HIS ABODE WHICH HE HAD FIXED AT A BOWERY OR COUNTRY SEAT AT A SHORT DISTANCE FROM THE CITY JUST AT 
                    WHAT IS NOW CALLED DUTCH STREET SOON ABOUNDED WITH PROOFS OF HIS INGENUITY PATENT SMOKE JACKS THAT REQUIRED 
                    A HORSE TO WORK THEM DUTCH OVENS THAT ROASTED MEAT WITHOUT FIRE CARTS THAT WENT BEFORE THE HORSES WEATHERCOCKS 
                    THAT TURNED AGAINST THE WIND AND OTHER WRONG HEADED CONTRIVANCES THAT ASTONISHED AND CONFOUNDED ALL BEHOLDERS'

                    (Pdb) hypo["tokens"]
                    tensor([  16, 5311,   38,   12,   21, 1237,   30,    8, 6616,   73,   50,  318,
                            1136,   30,    8,  429,  803,   42,    4,  492,  137,   30,   59,   23,
                            74,  260, 4060,  609,  232, 7871,   27,   17, 1857,    9,    6,   16,
                            7970, 6808,   27,   17, 1857,    9,    6,   16, 7970, 6808,   27,   17,
                            1857,    9,    6,   16, 7970,  137,   30,   59,   23,   74,  260, 4060,
                            609,  232, 7871,   27,   17, 1857,    9,   13, 1826,    8,  441,    7,
                            223,   57, 4060,  609,  232, 7871,   27,   17,   15,   30,   59,   23,
                            74,  260, 4060,  609,  232, 7871,   27,   17, 1857,    9,   13, 1826,
                            8,  441,    7,  223,   57, 4060,  609,  232, 7871,   27,   17, 1857,
                            9,   13, 1826,    8,  441,    7,  223,   57, 4060,  609,  232, 7871,
                            27,   17, 1857,    9,   13, 1826,    8,  441,    7,  223,   57, 4060,
                            622,  508,  609,  232, 7871,   27,   17, 1857,    9,   13, 1826,    8,
                            441,    7,  223,   57, 4060,  609,  232, 7871,   27,   17, 1857,    9,
                            13, 1826,    8,  441,    7,  223,   57, 4060,  622,  508,  609,  232,
                            7871,   27,   17, 1857,    9,   13, 1826,    8,  441,    7,  223,   57,
                            4060, 5779,    9,   13,   15,   30,   59,   23,   74,  260, 4060, 5779,
                            9,   13,   23,   74,  260, 4060, 5779,    9,    2], device='cuda:0')
                            
                    '''

                    '''
                    vanilla (아 이거 그냥 joint 학습이 안된거네 ㅋㅋ)

                    (Pdb) detok_hypo_str
                    'HIS ABODE WHICH HE HAD FIXED AT A BOWERY OR COUNTRY SEAT AT A SHORT DISTANCE FROM THE CITY JUST 
                    AT WHAT IS NOW CALLED DUTCH STREET SOON ABOUNDED WITH PROOFS OF HIS INGENUITY PATENT SMOKE JACKS 
                    THAT WENT BEFORE THE HORSES WEATHERCOCKS THAT WENT BEFORE THE WIND AND OTHER WRONG HEADED CONTRIVANCES THAT ASTONISHED AND CONFOUNDED ALL BEHOLDERS'
                    (Pdb) target_str
                    'HIS ABODE WHICH HE HAD FIXED AT A BOWERY OR COUNTRY SEAT AT A SHORT DISTANCE FROM THE CITY JUST 
                    AT WHAT IS NOW CALLED DUTCH STREET SOON ABOUNDED WITH PROOFS OF HIS INGENUITY PATENT SMOKE JACKS 
                    THAT REQUIRED A HORSE TO WORK THEM DUTCH OVENS THAT ROASTED MEAT WITHOUT FIRE CARTS THAT WENT BEFORE 
                    THE HORSES WEATHERCOCKS THAT TURNED AGAINST THE WIND AND OTHER WRONG HEADED CONTRIVANCES THAT ASTONISHED AND CONFOUNDED ALL BEHOLDERS'

                    '''

                    # Tra()

        wps_meter.update(num_generated_tokens)
        progress.log({"wps": round(wps_meter.avg)})
        num_sentences += (
            sample["nsentences"] if "nsentences" in sample else sample["id"].numel()
        )

    logger.info("NOTE: hypothesis and token scores are output in base 2")
    logger.info(
        "Translated {:,} sentences ({:,} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)".format(
            num_sentences,
            gen_timer.n,
            gen_timer.sum,
            num_sentences / gen_timer.sum,
            1.0 / gen_timer.avg,
        )
    )
    if has_target:
        if cfg.bpe and not cfg.generation.sacrebleu:
            if cfg.common_eval.post_process:
                logger.warning(
                    "BLEU score is being computed by splitting detokenized string on spaces, this is probably not what you want. Use --sacrebleu for standard 13a BLEU tokenization"
                )
            else:
                logger.warning(
                    "If you are using BPE on the target side, the BLEU score is computed on BPE tokens, not on proper words.  Use --sacrebleu for standard 13a BLEU tokenization"
                )
        # use print to be consistent with other main outputs: S-, H-, T-, D- and so on
        print(
            "Generate {} with beam={}: {}".format(
                cfg.dataset.gen_subset, cfg.generation.beam, scorer.result_string()
            ),
            file=output_file,
        )

    return scorer
    

def cli_main():
    parser = options.get_generation_parser()
    # TODO: replace this workaround with refactoring of `AudioPretraining`
    parser.add_argument(
        "--arch",
        "-a",
        metavar="ARCH",
        default="wav2vec2",
        help="Model architecture. For constructing tasks that rely on "
        "model args (e.g. `AudioPretraining`)",
    )
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == "__main__":
    cli_main()
