#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Run inference for pre-processed data with a trained model.
"""

import ast
import logging
import math
import os
import sys
import time

import editdistance
import numpy as np
import torch
from fairseq import checkpoint_utils, options, progress_bar, tasks, utils
from fairseq.data.data_utils import post_process
from fairseq.logging.meters import StopwatchMeter, TimeMeter


logging.basicConfig()
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_asr_eval_argument(parser):
    parser.add_argument("--kspmodel", default=None, help="sentence piece model")
    parser.add_argument(
        "--wfstlm", default=None, help="wfstlm on dictonary output units"
    )
    parser.add_argument(
        "--rnnt_decoding_type",
        default="greedy",
        help="wfstlm on dictonary\
output units",
    )

    try:
        parser.add_argument(
            "--lm-weight",
            "--lm_weight",
            type=float,
            default=0.2,
            help="weight for lm while interpolating with neural score",
        )
    except:
        pass

    parser.add_argument(
        "--rnnt_len_penalty", default=-0.5, help="rnnt length penalty on word level"
    )

    parser.add_argument(
        "--w2l-decoder",
        choices=["viterbi", "kenlm", "fairseqlm"],
        help="use a w2l decoder",
    )

    parser.add_argument("--lexicon", help="lexicon for w2l decoder")
    parser.add_argument("--unit-lm", action="store_true", help="if using a unit lm")
    parser.add_argument("--kenlm-model", "--lm-model", help="lm model for w2l decoder")
    parser.add_argument("--beam-threshold", type=float, default=25.0) 
    parser.add_argument("--beam-size-token", type=float, default=100)
    # parser.add_argument("--beam-size-token", type=float, default=None)
    parser.add_argument("--word-score", type=float, default=1.0)
    parser.add_argument("--unk-weight", type=float, default=-math.inf)
    parser.add_argument("--sil-weight", type=float, default=0.0)
    parser.add_argument("--rescoring", action="store_true")
    parser.add_argument("--rescoring_weight", type=float, default=-1)
    parser.add_argument("--spelling_correction", action="store_true")
    parser.add_argument("--hparam_search", action="store_true")


    parser.add_argument(
        "--dump-emissions",
        type=str,
        default=None,
        help="if present, dumps emissions into this file and exits",
    )
    parser.add_argument(
        "--dump-features",
        type=str,
        default=None,
        help="if present, dumps features into this file and exits",
    )
    parser.add_argument(
        "--load-emissions",
        type=str,
        default=None,
        help="if present, loads emissions from this file",
    )
    return parser


def check_args(args):
    # assert args.path is not None, "--path required for generation!"
    # assert args.results_path is not None, "--results_path required for generation!"
    assert (
        not args.sampling or args.nbest == args.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
        args.replace_unk is None or args.raw_text
    ), "--replace-unk requires a raw text dataset (--raw-text)"


def get_dataset_itr(args, task, models):
    return task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.batch_size,
        max_positions=(sys.maxsize, sys.maxsize),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
        data_buffer_size=args.data_buffer_size,
    ).next_epoch_itr(shuffle=False)


def process_predictions(
    args, hypos, sp, tgt_dict, target_tokens, res_files, speaker, id, spelling_correction_model
):

    stats=list()
    
    # 여기가 뭔가 이상한데
    # for문을 돌다가 갑자기 리턴을해? 걍 1best 결과만 하겠다는거니?
    for i, hypo in enumerate(hypos[: min(len(hypos), args.nbest)]):
        hyp_pieces = tgt_dict.string(hypo["tokens"].int().cpu())

        '''
        (Pdb) hyp_pieces
        "| I | A M | W I L L I N G | T O | E N T E R | I N T O | C O M P E T I T I O N | W I T H | T H E | A N C I E N T S | A N D | F E E L | A B L E | T O | S U R P A S S | 
        T H E M | F O R | S I N C E | T H O S E | E A R L Y | D A Y S | I N | W H I C H | I | M A D E | T H E | M E T A L S | O F | P O P E | C L E M E N T | I | H A V E | 
        L E A R N E D | S O | M U C H | T H A T | I | C A N | N O W | P R O D U C E | F A R | B E T T E R | P I E C E S | O F | T H E | K I N D | I | T H I N K | I | C A N | 
        A L S O | O U T D O | T H E | C O I N S | I | S T R U C K | F O  R | D U K E | A L E S S A N D R O | W H I C H | A R E | S T I L L | H E L D | I N | H I G H | 
        E S T E E M | I N | L I K E | M A N N E R | I | C O U L D | M A K E | F O R | Y O U | L A R G E | P I E C E S | O F | G O L D | A N D | S I L V E R | P L A T E | 
        A S | I | D I D | S O | O F T E N | F O R | T H A T | N O B L E | M O N A R C H | K I N G | F R A N C I S | O F | F R A N C E | T H A N K S | T O | T H E | 
        G R E A T | C O N V E N I E N C E S | H E | A L L O W E D | M E | W I T H O U T | E V E R | L O S I N G | T I M E | F O R | T H E | E X E C U T I O N | O F | 
        C O L O S S A L | S T A T U E S | O R | O T H E R | W O R K S | O F | T H E | S C U L P T O R ' S | C R A F T | |"
        '''

        if "words" in hypo:
            hyp_words = " ".join(hypo["words"])
        else:
            hyp_words = post_process(hyp_pieces, args.post_process)

        if args.spelling_correction:
            iter_decode_max_iter = 5
            corrected_hyp_words = spelling_correction_model.translate(hyp_words, iter_decode_max_iter=iter_decode_max_iter)

        hyp_words = hyp_words.upper()

        if i == 0 :
            if res_files is not None:
                print(
                    "{} ({}-{})".format(hyp_pieces, speaker, id),
                    file=res_files["hypo.units"],
                )
                print(
                    "{} ({}-{})".format(hyp_words, speaker, id),
                    file=res_files["hypo.words"],
                )

        tgt_pieces = tgt_dict.string(target_tokens)
        tgt_words = post_process(tgt_pieces, args.post_process)

        tgt_words = tgt_words.upper()
        
        if i == 0 :
            if res_files is not None:
                print(
                    "{} ({}-{})".format(tgt_pieces, speaker, id),
                    file=res_files["ref.units"],
                )
                print(
                    "{} ({}-{})".format(tgt_words, speaker, id), file=res_files["ref.words"]
                )

            if not args.quiet:
                logger.info("HYPO:" + hyp_words)
                # logger.info("TARGET:" + tgt_words)
                logger.info("TARG:" + tgt_words)
                logger.info("___________________")

        hyp_words = hyp_words.split()
        tgt_words = tgt_words.split()



        '''
        (Pdb) " ".join(hypos[0][0]["words"])
        "I AM WILLING TO ENTER INTO COMPETITION WITH THE ANCIENTS AND FEEL ABLE TO SURPASS THEM FOR 
        SINCE THOSE EARLY DAYS IN WHICH I MADE THE METALS OF POPE CLEMENT I HAVE LEARNED SO MUCH THAT 
        I CAN NOW PRODUCE FAR BETTER PIECES OF THE KIND I THINK I CAN ALSO OUTDO THE COINS I STRUCK 
        FOR DUKE ALESSANDRO WHICH ARE STILL HELD IN HIGH ESTEEM IN LIKE MANNER I COULD MAKE FOR YOU 
        LARGE PIECES OF GOLD AND SILVER PLATE AS I DID SO OFTEN FOR THAT NOBLE MONARCH KING FRANCIS 
        OF FRANCE THANKS TO THE GREAT CONVENIENCES HE ALLOWED ME WITHOUT EVER LOSING TIME FOR THE EXECUTION 
        OF COLOSSAL STATUES OR OTHER WORKS OF THE SCULPTOR'S CRAFT"

        (Pdb) " ".join(hypos[0][0]["words"]).split()
        ['I', 'AM', 'WILLING', 'TO', 'ENTER', 'INTO', 'COMPETITION', 'WITH', 'THE', 'ANCIENTS', 'AND', 'FEEL', 'ABLE', 'TO', 'SURPASS', 'THEM', 'FOR', 
        'SINCE', 'THOSE', 'EARLY', 'DAYS', 'IN', 'WHICH', 'I', 'MADE', 'THE', 'METALS', 'OF', 'POPE', 'CLEMENT', 'I', 'HAVE', 'LEARNED', 'SO', 'MUCH', 'THAT', 
        'I', 'CAN', 'NOW', 'PRODUCE', 'FAR', 'BETTER', 'PIECES', 'OF', 'THE', 'KIND', 'I', 'THINK', 'I', 'CAN', 'ALSO', 'OUTDO', 'THE', 'COINS', 'I', 'STRUCK', 
        'FOR', 'DUKE', 'ALESSANDRO', 'WHICH', 'ARE', 'STILL', 'HELD', 'IN', 'HIGH', 'ESTEEM', 'IN', 'LIKE', 'MANNER', 'I', 'COULD', 'MAKE', 'FOR', 'YOU', 
        'LARGE', 'PIECES', 'OF', 'GOLD', 'AND', 'SILVER', 'PLATE', 'AS', 'I', 'DID', 'SO', 'OFTEN', 'FOR', 'THAT', 'NOBLE', 'MONARCH', 'KING', 'FRANCIS', 
        'OF', 'FRANCE', 'THANKS', 'TO', 'THE', 'GREAT', 'CONVENIENCES', 'HE', 'ALLOWED', 'ME', 'WITHOUT', 'EVER', 'LOSING', 'TIME', 'FOR', 'THE', 'EXECUTION', 
        'OF', 'COLOSSAL', 'STATUES', 'OR', 'OTHER', 'WORKS', 'OF', 'THE', "SCULPTOR'S", 'CRAFT']



        (Pdb) tgt_dict.string(target_tokens)
        'I | A M | W I L L I N G | T O | E N T E R | I N T O | C O M P E T I T I O N | W I T H | T H E | A N C I E N T S | A N D | F E E L | A B L E | T O | S U R P A S S | T H E M | F O R | 
        S I N C E | T H O S E | E A R L Y | D A Y S | I N | W H I C H | I | M A D E | T H E | M E D A L S | O F | P O P E | C L E M E N T | I | H A V E | L E A R N E D | S O | M U C H | T H A T | 
        I | C A N | N O W | P R O D U C E | F A R | B E T T E R | P I E C E S | O F | T H E | K I N D | I | T H I N K | I | C A N | A L S O | O U T D O | T H E | C O I N S | I | S T R U C K | 
        F O R | D U K E | A L E S S A N D R O | W H I C H | A R E | S T I L L | H E L D | I N | H I G H | E S T E E M | I N | L I K E | M A N N E R | I | C O U L D | M A K E | F O R | Y O U | 
        L A R G E | P I E C E S | O F | G O L D | A N D | S I L V E R | P L A T E | A S | I | D I D | S O | O F T E N | F O R | T H A T | N O B L E | M O N A R C H | K I N G | F R A N C I S | 
        O F | F R A N C E | T H A N K S | T O | T H E | G R E A T | C O N V E N I E N C E S | H E | A L L O W E D | M E | W I T H O U T | E V E R | L O S I N G | T I M E | F O R | T H E | E X E C U T I O N | 
        O F | C O L O S S A L | S T A T U E S | O R | O T H E R | W O R K S | O F | T H E | S C U L P T O R S | C R A F T |'

        (Pdb) post_process(tgt_dict.string(target_tokens),args.post_process)
        'I AM WILLING TO ENTER INTO COMPETITION WITH THE ANCIENTS AND FEEL ABLE TO SURPASS THEM FOR 
        SINCE THOSE EARLY DAYS IN WHICH I MADE THE MEDALS OF POPE CLEMENT I HAVE LEARNED SO MUCH THAT 
        I CAN NOW PRODUCE FAR BETTER PIECES OF THE KIND I THINK I CAN ALSO OUTDO THE COINS I STRUCK 
        FOR DUKE ALESSANDRO WHICH ARE STILL HELD IN HIGH ESTEEM IN LIKE MANNER I COULD MAKE FOR YOU 
        LARGE PIECES OF GOLD AND SILVER PLATE AS I DID SO OFTEN FOR THAT NOBLE MONARCH KING FRANCIS 
        OF FRANCE THANKS TO THE GREAT CONVENIENCES HE ALLOWED ME WITHOUT EVER LOSING TIME FOR THE EXECUTION 
        OF COLOSSAL STATUES OR OTHER WORKS OF THE SCULPTORS CRAFT'

        (Pdb) post_process(tgt_dict.string(target_tokens),args.post_process).split()
        ['I', 'AM', 'WILLING', 'TO', 'ENTER', 'INTO', 'COMPETITION', 'WITH', 'THE', 'ANCIENTS', 'AND', 'FEEL', 'ABLE', 'TO', 'SURPASS', 'THEM', 'FOR', 
        'SINCE', 'THOSE', 'EARLY', 'DAYS', 'IN', 'WHICH', 'I', 'MADE', 'THE', 'MEDALS', 'OF', 'POPE', 'CLEMENT', 'I', 'HAVE', 'LEARNED', 'SO', 'MUCH', 'THAT', 
        'I', 'CAN', 'NOW', 'PRODUCE', 'FAR', 'BETTER', 'PIECES', 'OF', 'THE', 'KIND', 'I', 'THINK', 'I', 'CAN', 'ALSO', 'OUTDO', 'THE', 'COINS', 'I', 'STRUCK', 
        'FOR', 'DUKE', 'ALESSANDRO', 'WHICH', 'ARE', 'STILL', 'HELD', 'IN', 'HIGH', 'ESTEEM', 'IN', 'LIKE', 'MANNER', 'I', 'COULD', 'MAKE', 'FOR', 'YOU', 
        'LARGE', 'PIECES', 'OF', 'GOLD', 'AND', 'SILVER', 'PLATE', 'AS', 'I', 'DID', 'SO', 'OFTEN', 'FOR', 'THAT', 'NOBLE', 'MONARCH', 'KING', 'FRANCIS', 
        'OF', 'FRANCE', 'THANKS', 'TO', 'THE', 'GREAT', 'CONVENIENCES', 'HE', 'ALLOWED', 'ME', 'WITHOUT', 'EVER', 'LOSING', 'TIME', 'FOR', 'THE', 'EXECUTION', 
        'OF', 'COLOSSAL', 'STATUES', 'OR', 'OTHER', 'WORKS', 'OF', 'THE', 'SCULPTORS', 'CRAFT']
        '''


        '''
        (Pdb) editdistance.eval(" ".join(hypos[0][0]["words"]).split(),post_process(tgt_dict.string(target_tokens),args.post_process).split())
        2
        (Pdb) len(post_process(tgt_dict.string(target_tokens),args.post_process).split())
        119
        (Pdb) len(" ".join(hypos[0][0]["words"]).split())
        119
        '''

        stats.append(
            {
            'wer' : editdistance.eval(hyp_words, tgt_words),
            'length' : len(tgt_words),
            }
        )

        # return editdistance.eval(hyp_words, tgt_words), len(tgt_words)
    return stats


def prepare_result_files(args):
    def get_res_file(file_prefix):
        if args.num_shards > 1:
            file_prefix = f"{args.shard_id}_{file_prefix}"
        path = os.path.join(
            args.results_path,
            "{}-{}-{}.txt".format(
                file_prefix, os.path.basename(args.path), args.gen_subset
            ),
        )
        return open(path, "w", buffering=1)

    if not args.results_path:
        return None

    return {
        "hypo.words": get_res_file("hypo.word"),
        "hypo.units": get_res_file("hypo.units"),
        "ref.words": get_res_file("ref.word"),
        "ref.units": get_res_file("ref.units"),
    }


def optimize_models(args, use_cuda, models):
    """Optimize ensemble for generation"""
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()


def apply_half(t):
    if t.dtype is torch.float32:
        return t.to(dtype=torch.half)
    return t


class ExistingEmissionsDecoder(object):
    def __init__(self, decoder, emissions):
        self.decoder = decoder
        self.emissions = emissions

    def generate(self, models, sample, **unused):
        ids = sample["id"].cpu().numpy()
        try:
            emissions = np.stack(self.emissions[ids])
        except:
            print([x.shape for x in self.emissions[ids]])
            raise Exception("invalid sizes")
        emissions = torch.from_numpy(emissions)
        return self.decoder.decode(emissions)


def main(args, task=None, model_state=None):
    check_args(args)

    use_fp16 = args.fp16
    if args.max_tokens is None and args.batch_size is None:
        args.max_tokens = 4000000
    print('batch_size : {}'.format(args.batch_size))
    print('max_tokens : {}'.format(args.max_tokens))
    logger.info(args)

    # import pdb; pdb.set_trace()

    use_cuda = torch.cuda.is_available() and not args.cpu

    logger.info("| decoding with criterion {}".format(args.criterion))

    task = tasks.setup_task(args)

    # Load ensemble
    if args.load_emissions:
        models, criterions = [], []
        task.load_dataset(args.gen_subset)
    else:
        logger.info("| loading model(s) from {}".format(args.path))
        models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            utils.split_paths(args.path, separator="\\"),
            arg_overrides=ast.literal_eval(args.model_overrides),
            task=task,
            suffix=args.checkpoint_suffix,
            strict=(args.checkpoint_shard_count == 1),
            num_shards=args.checkpoint_shard_count,
            state=model_state,
        )
        optimize_models(args, use_cuda, models)
        task.load_dataset(args.gen_subset, task_cfg=saved_cfg.task)

    spelling_correction_model = None
    if args.spelling_correction:
        from fairseq.models.transformer import TransformerModel
        model_name_or_path = ""
        checkpoint_file = "checkpoint_best.pt"
        data_name_or_path = "./data"
        bpe = "sentencepiece"
        sentencepiece_model = "./sentence.bpe.model"
        spelling_correction_model = TransformerModel.from_pretrained(
            model_name_or_path, 
            checkpoint_file=checkpoint_file, 
            data_name_or_path=data_name_or_path, 
            bpe=bpe, 
            sentencepiece_model=sentencepiece_model
            ).eval().cuda()

    if args.rescoring : 
        device = 'cuda'
        model_name = "gpt2"
        # model_name = "gpt2-large"
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast, AutoModelForCausalLM, AutoTokenizer

        rescoring_model = (
            AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name, is_decoder=True)
            .to(device)
            .eval()
        )
        rescoring_tokenizer = AutoTokenizer.from_pretrained(model_name)
        rescoring_tokenizer.pad_token = rescoring_tokenizer.eos_token

        # device = 'cuda'
        # model_name = "gpt2"
        # self.rescoring_model = (
        #     GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path=model_name, is_decoder=True)
        #     .to(device)
        #     .eval()
        # )
        # self.rescoring_tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        # self.rescoring_tokenizer.pad_token = self.rescoring_tokenizer.eos_token

        # device = 'cuda'
        # model_name = "roberta-base"
        # from transformers import RobertaTokenizer, RobertaModel, RobertaForMaskedLM
        # rescoring_model = RobertaForMaskedLM.from_pretrained(model_name).to(device).eval()
        # rescoring_tokenizer = RobertaTokenizer.from_pretrained(model_name)

        # class cfg:
        #     path = "former_xl_char_lm_model_bbs_bos]$ pwd/home1/irteam/users/seosh/decoder_pratice/training_shell_scripts/librispeech_12laye"
        #     task = "language_modeling"
        #     model_overrides = '{"mem_len":640,"clamp_len":400,"same_length":True}'

        # # Load ensemble
        # from fairseq import checkpoint_utils
        # task = tasks.setup_task(cfg.task)
        # models, model_args, task = checkpoint_utils.load_model_ensemble_and_task(
        #     [cfg.path],
        #     arg_overrides=eval(cfg.model_overrides),
        #     task=task,
        # )

        # import pdb
        # pdb.set_trace()


    # Set dictionary
    tgt_dict = task.target_dictionary

    logger.info(
        "| {} {} {} examples".format(
            args.data, args.gen_subset, len(task.dataset(args.gen_subset))
        )
    )

    # hack to pass transitions to W2lDecoder
    if args.criterion == "asg_loss":
        raise NotImplementedError("asg_loss is currently not supported")
        # trans = criterions[0].asg.trans.data
        # args.asg_transitions = torch.flatten(trans).tolist()

    # Load dataset (possibly sharded)
    itr = get_dataset_itr(args, task, models)

    # Initialize generator
    gen_timer = StopwatchMeter()

    def build_generator(args):
        w2l_decoder = getattr(args, "w2l_decoder", None)
        if w2l_decoder == "viterbi":
            from examples.speech_recognition.w2l_decoder import W2lViterbiDecoder
            return W2lViterbiDecoder(args, task.target_dictionary)
        elif w2l_decoder == "kenlm":
            from examples.speech_recognition.w2l_decoder import W2lKenLMDecoder
            return W2lKenLMDecoder(args, task.target_dictionary)
        elif w2l_decoder == "fairseqlm": # transformer
            from examples.speech_recognition.w2l_decoder import W2lFairseqLMDecoder
            return W2lFairseqLMDecoder(args, task.target_dictionary)
        else:
            print(
                "only flashlight decoders with (viterbi, kenlm, fairseqlm) options are supported at the moment"
            )

    # please do not touch this unless you test both generate.py and infer.py with audio_pretraining task
    generator = build_generator(args)

    if args.load_emissions:
        generator = ExistingEmissionsDecoder(
            generator, np.load(args.load_emissions, allow_pickle=True)
        )
        logger.info("loaded emissions from " + args.load_emissions)

    num_sentences = 0

    if args.results_path is not None and not os.path.exists(args.results_path):
        os.makedirs(args.results_path)

    max_source_pos = (
        utils.resolve_max_positions(
            task.max_positions(), *[model.max_positions() for model in models]
        ),
    )

    if max_source_pos is not None:
        max_source_pos = max_source_pos[0]
        if max_source_pos is not None:
            max_source_pos = max_source_pos[0] - 1

    if args.dump_emissions:
        emissions = {}
    if args.dump_features:
        features = {}
        models[0].bert.proj = None
    else:
        res_files = prepare_result_files(args)


    errs_t = 0
    lengths_t = 0

    best_possible_errs_t = 0
    best_possible_lengths_t = 0

    with progress_bar.build_progress_bar(args, itr) as t:
        wps_meter = TimeMeter()

        # 두 번째 배치부터 터지는데?
        for sample in t:

            # import pdb
            # pdb.set_trace()

            '''
            (Pdb) sample.keys()
            dict_keys(['id', 'net_input', 'target_lengths', 'ntokens', 'target'])

            sample['net_input']['source'].size() # speech input 인듯                                                                                                          
            torch.Size([7, 552160])

            (Pdb) sample['target'].size() # sentence target 인듯
            torch.Size([7, 619])
            '''

            sample = utils.move_to_cuda(sample) if use_cuda else sample
            if use_fp16:
                sample = utils.apply_to_sample(apply_half, sample)
            if "net_input" not in sample:
                continue

            prefix_tokens = None
            if args.prefix_size > 0:
                prefix_tokens = sample["target"][:, : args.prefix_size]

            gen_timer.start()

            if args.dump_emissions:
                with torch.no_grad():
                    encoder_out = models[0](**sample["net_input"])
                    emm = models[0].get_normalized_probs(encoder_out, log_probs=True)
                    emm = emm.transpose(0, 1).cpu().numpy()
                    for i, id in enumerate(sample["id"]):
                        emissions[id.item()] = emm[i]
                    continue
            elif args.dump_features:
                with torch.no_grad():
                    encoder_out = models[0](**sample["net_input"])
                    feat = encoder_out["encoder_out"].transpose(0, 1).cpu().numpy()
                    for i, id in enumerate(sample["id"]):
                        padding = (
                            encoder_out["encoder_padding_mask"][i].cpu().numpy()
                            if encoder_out["encoder_padding_mask"] is not None
                            else None
                        )
                        features[id.item()] = (feat[i], padding)
                    continue

            hypos = task.inference_step(generator, models, sample, prefix_tokens)

            ## generate -> get emission -> decode

            '''
            (Pdb) hypos[0][0].keys()
            dict_keys(['tokens', 'score', 'timesteps', 'words'])

            (Pdb) hypos[0][0]['words']
            ['I', 'AM', 'WILLING', 'TO', 'ENTER', 'INTO', 'COMPETITION', 'WITH', 'THE', 'ANCIENTS', 
            'AND', 'FEEL', 'ABLE', 'TO', 'SURPASS', 'THEM', 'FOR', 'SINCE', 'THOSE', 'EARLY', 'DAYS', 
            'IN', 'WHICH', 'I', 'MADE', 'THE', 'METALS', 'OF', 'POPE', 'CLEMENT', 'I', 'HAVE', 'LEARNED', 
            'SO', 'MUCH', 'THAT', 'I', 'CAN', 'NOW', 'PRODUCE', 'FAR', 'BETTER', 'PIECES', 'OF', 'THE', 'KIND', 
            'I', 'THINK', 'I', 'CAN', 'ALSO', 'OUTDO', 'THE', 'COINS', 'I', 'STRUCK', 'FOR', 'DUKE', 'ALESSANDRO', 
            'WHICH', 'ARE', 'STILL', 'HELD', 'IN', 'HIGH', 'ESTEEM', 'IN', 'LIKE', 'MANNER', 'I', 'COULD', 'MAKE', 
            'FOR', 'YOU', 'LARGE', 'PIECES', 'OF', 'GOLD', 'AND', 'SILVER', 'PLATE', 'AS', 'I', 'DID', 'SO', 'OFTEN', 
            'FOR', 'THAT', 'NOBLE', 'MONARCH', 'KING', 'FRANCIS', 'OF', 'FRANCE', 'THANKS', 'TO', 'THE', 'GREAT', 
            'CONVENIENCES', 'HE', 'ALLOWED', 'ME', 'WITHOUT', 'EVER', 'LOSING', 'TIME', 'FOR', 'THE', 'EXECUTION', 
            'OF', 'COLOSSAL', 'STATUES', 'OR', 'OTHER', 'WORKS', 'OF', 'THE', "SCULPTOR'S", 'CRAFT']
            '''

            num_generated_tokens = sum(len(h[0]["tokens"]) for h in hypos)

            gen_timer.stop(num_generated_tokens)

            '''
            (Pdb) num_generated_tokens
            622            
            '''

            '''
            hypos 는 nested list인데
            첫 번째는 speaker id를 의미하고
            두 번째는 nbest개의 hypothesis를 의미한다.
            '''

            if args.rescoring:

                reordered_hypos = []

                for nbest_hypos in hypos:
                    inputs = []
                    beams = []
                    for hypo in nbest_hypos:
                        inputs.append(' '.join(hypo['words']))
                    encoded = rescoring_tokenizer(inputs, return_tensors='pt', padding=True, truncation=True)['input_ids'].to(rescoring_model.device)
                    inputs_mask = (encoded != rescoring_tokenizer.pad_token_id)

                    with torch.no_grad():
                        output = rescoring_model(input_ids=encoded, attention_mask=inputs_mask)
                        log_probs = torch.nn.functional.log_softmax(output.logits, dim=-1)
                        target_log_probs = log_probs[:, :-1].gather(2, encoded[:, 1:].unsqueeze(2)).squeeze(2)
                        neural_lm_score = torch.sum(target_log_probs * inputs_mask[:, 1:], dim=-1)
                        rescored_results=(neural_lm_score)

                    for hypo, rescored_result in zip(nbest_hypos, rescored_results):
                        beams.append({
                            "tokens":hypo['tokens'], 
                            "score" : hypo['score'], 
                            "timesteps" : hypo['timesteps'], 
                            "words" : hypo['words'], 
                            "rescoring" : rescored_result,
                            # "total_score" : hypo['score'] + self.rescoring_weight * rescored_result
                            })

                    '''
                    ## flashlight decoder score (am + lm beam search maybe?)
                    (Pdb) torch.tensor([tmp['score'] for tmp in nbest_hypos])
                    tensor([21031.2617, 21030.2500, 21026.6172, 21026.3652, 21025.8457, 21025.7324,
                            21025.6055, 21025.4688])

                    ## log softmax sum
                    (Pdb) rescored_results
                    tensor([-711.9091, -710.1793, -716.3192, -712.5317, -713.6451, -716.3398,
                            -714.7936, -702.9783], device='cuda:0')

                    ## logit sum
                    (Pdb) neural_lm_score
                    tensor([-20230.5352, -20261.6445, -20284.9336, -20233.1250, -20201.2852,
                            -20069.5234, -20313.2480, -20026.4883], device='cuda:0')

                    ## GPT 기준 best path
                    (Pdb) inputs[-1] 
                    "I AM WILLING TO ENTER INTO COMPETITION WITH THE ANCIENTS AND FEEL ABLE TO SURPASS THEM FOR SINCE THOSE EARLY DAYS IN 
                    WHICH I MADE THE MEDALS OF POPE CLEMENT I HAVE LEARNED SO MUCH THAT I CAN NOW PRODUCE FAR BETTER PIECES OF THE KIND I THINK 
                    I CAN ALSO OUTDO THE COINS I STRUCK FOR DUKE ALESSANDRO WHICH ARE STILL HELD IN HIGH ESTEEM IN LIKE MANNER I COULD MAKE FOR YOU 
                    LARGE PIECES OF GOLD AND SILVER PLATE AS I DID SO OFTEN FOR THAT NOBLE MONARCH KING FRANCIS OF FRANCE THANKS TO THE GREAT CONVENIENCE 
                    HE ALLOWED ME WITHOUT EVER LOSING TIME FOR THE EXECUTION OF COLOSSAL STATUES OR OTHER WORKS OF THE SCULPTOR'S CRAFT"

                    # am 기준 best path
                    (Pdb) inputs[0]
                    "I AM WILLING TO ENTER INTO COMPETITION WITH THE ANCIENTS AND FEEL ABLE TO SURPASS THEM FOR SINCE THOSE EARLY DAYS IN 
                    WHICH I MADE THE MEDALS OF POPE CLEMENT I HAVE LEARNED SO MUCH THAT I CAN NOW PRODUCE FAR BETTER PIECES OF THE KIND I THINK 
                    I CAN ALSO OUTDO THE COINS I STRUCK FOR DUKE ALESSANDRO WHICH ARE STILL HELD IN HIGH ESTEEM IN LIKE MANNER I COULD MAKE FOR YOU 
                    LARGE PIECES OF GOLD AND SILVER PLATE AS I DID SO OFTEN FOR THAT NOBLE MONARCH KING FRANCIS OF FRANCE THANKS TO THE GREAT CONVENIENCES 
                    HE ALLOWED ME WITHOUT EVER LOSING TIME FOR THE EXECUTION OF COLOSSAL STATUES OR OTHER WORKS OF THE SCULPTOR'S CRAFT"
                    '''

                    # Original Rescoring log P_{AM} (y|x) + \alpha1 log P_{LM1}(y) + \beta |y| + \alpha2 log P_{LM2}(y)

                    # on the fly
                    sorted_beams = sorted(beams, reverse=True, key=lambda x:x['rescoring'])
                    # sorted_beams = sorted(beams, reverse=False, key=lambda x:x['rescoring'])

                    import pdb
                    pdb.set_trace()

                    reordered_hypos.append(sorted_beams)

                hypos = reordered_hypos

            for i, sample_id in enumerate(sample["id"].tolist()):
                '''
                (Pdb) sample["id"].tolist()
                [1877]                
                '''
                speaker = None
                # id = task.dataset(args.gen_subset).ids[int(sample_id)]
                id = sample_id
                toks = (
                    sample["target"][i, :]
                    if "target_label" not in sample
                    else sample["target_label"][i, :]
                )
                '''
                (Pdb) toks.size()
                torch.Size([619])
                '''

                target_tokens = utils.strip_pad(toks, tgt_dict.pad()).int().cpu()
                '''
                ## 패딩을 없애는 부분
                (Pdb) tgt_dict.pad()
                1
                (Pdb) target_tokens.size()
                torch.Size([619])
                '''

                # # Process top predictions
                # errs, length = process_predictions(
                #     args,
                #     hypos[i],
                #     None,
                #     tgt_dict,
                #     target_tokens,
                #     res_files,
                #     speaker,
                #     id,
                # )

                '''
                (Pdb) errs
                2
                (Pdb) length
                119
                '''

                # Process top predictions
                stats_per_path = process_predictions(
                    args,
                    hypos[i],
                    None,
                    tgt_dict,
                    target_tokens,
                    res_files,
                    speaker,
                    id,
                    spelling_correction_model,
                )

                best_path = stats_per_path[0]
                errs, length = best_path['wer'], best_path['length']

                best_possible_path  = sorted(stats_per_path, reverse=False, key=lambda x:x['wer'])[0]
                best_possible_errs, best_possible_length = best_possible_path['wer'], best_possible_path['length']

                errs_t += errs # +2
                lengths_t += length # +119

                best_possible_errs_t += best_possible_errs 
                best_possible_lengths_t += best_possible_length

            # import pdb
            # pdb.set_trace()

            '''
            nbest = 1
            (Pdb) " ".join(hypos[0][0]["words"]).split()
            ['I', 'AM', 'WILLING', 'TO', 'ENTER', 'INTO', 'COMPETITION', 'WITH', 'THE', 'ANCIENTS', 'AND', 'FEEL', 'ABLE', 'TO', 'SURPASS', 'THEM', 'FOR', 
            'SINCE', 'THOSE', 'EARLY', 'DAYS', 'IN', 'WHICH', 'I', 'MADE', 'THE', 'METALS', 'OF', 'POPE', 'CLEMENT', 'I', 'HAVE', 'LEARNED', 'SO', 'MUCH', 'THAT', 
            'I', 'CAN', 'NOW', 'PRODUCE', 'FAR', 'BETTER', 'PIECES', 'OF', 'THE', 'KIND', 'I', 'THINK', 'I', 'CAN', 'ALSO', 'OUTDO', 'THE', 'COINS', 'I', 'STRUCK', 
            'FOR', 'DUKE', 'ALESSANDRO', 'WHICH', 'ARE', 'STILL', 'HELD', 'IN', 'HIGH', 'ESTEEM', 'IN', 'LIKE', 'MANNER', 'I', 'COULD', 'MAKE', 'FOR', 'YOU', 
            'LARGE', 'PIECES', 'OF', 'GOLD', 'AND', 'SILVER', 'PLATE', 'AS', 'I', 'DID', 'SO', 'OFTEN', 'FOR', 'THAT', 'NOBLE', 'MONARCH', 'KING', 'FRANCIS', 
            'OF', 'FRANCE', 'THANKS', 'TO', 'THE', 'GREAT', 'CONVENIENCES', 'HE', 'ALLOWED', 'ME', 'WITHOUT', 'EVER', 'LOSING', 'TIME', 'FOR', 'THE', 'EXECUTION', 
            'OF', 'COLOSSAL', 'STATUES', 'OR', 'OTHER', 'WORKS', 'OF', 'THE', "SCULPTOR'S", 'CRAFT']

            (Pdb) post_process(tgt_dict.string(target_tokens),args.post_process).split()
            ['I', 'AM', 'WILLING', 'TO', 'ENTER', 'INTO', 'COMPETITION', 'WITH', 'THE', 'ANCIENTS', 'AND', 'FEEL', 'ABLE', 'TO', 'SURPASS', 'THEM', 'FOR', 
            'SINCE', 'THOSE', 'EARLY', 'DAYS', 'IN', 'WHICH', 'I', 'MADE', 'THE', 'MEDALS', 'OF', 'POPE', 'CLEMENT', 'I', 'HAVE', 'LEARNED', 'SO', 'MUCH', 'THAT', 
            'I', 'CAN', 'NOW', 'PRODUCE', 'FAR', 'BETTER', 'PIECES', 'OF', 'THE', 'KIND', 'I', 'THINK', 'I', 'CAN', 'ALSO', 'OUTDO', 'THE', 'COINS', 'I', 'STRUCK', 
            'FOR', 'DUKE', 'ALESSANDRO', 'WHICH', 'ARE', 'STILL', 'HELD', 'IN', 'HIGH', 'ESTEEM', 'IN', 'LIKE', 'MANNER', 'I', 'COULD', 'MAKE', 'FOR', 'YOU', 
            'LARGE', 'PIECES', 'OF', 'GOLD', 'AND', 'SILVER', 'PLATE', 'AS', 'I', 'DID', 'SO', 'OFTEN', 'FOR', 'THAT', 'NOBLE', 'MONARCH', 'KING', 'FRANCIS', 
            'OF', 'FRANCE', 'THANKS', 'TO', 'THE', 'GREAT', 'CONVENIENCES', 'HE', 'ALLOWED', 'ME', 'WITHOUT', 'EVER', 'LOSING', 'TIME', 'FOR', 'THE', 'EXECUTION', 
            'OF', 'COLOSSAL', 'STATUES', 'OR', 'OTHER', 'WORKS', 'OF', 'THE', 'SCULPTORS', 'CRAFT']
            '''

            wps_meter.update(num_generated_tokens)
            t.log({"wps": round(wps_meter.avg)})
            num_sentences += (
                sample["nsentences"] if "nsentences" in sample else sample["id"].numel()
            )
            '''
            (Pdb) num_generated_tokens
            622
            (Pdb) wps_meter.avg
            1.1986295499371433
            '''

    # import pdb
    # pdb.set_trace()

    wer = None
    best_possible_wer = None

    if args.dump_emissions:
        emm_arr = []
        for i in range(len(emissions)):
            emm_arr.append(emissions[i])
        np.save(args.dump_emissions, emm_arr)
        logger.info(f"saved {len(emissions)} emissions to {args.dump_emissions}")
    elif args.dump_features:
        feat_arr = []
        for i in range(len(features)):
            feat_arr.append(features[i])
        np.save(args.dump_features, feat_arr)
        logger.info(f"saved {len(features)} emissions to {args.dump_features}")
    else:
        if lengths_t > 0:
            wer = errs_t * 100.0 / lengths_t
            logger.info(f"WER: {wer}")

            best_possible_wer = best_possible_errs_t * 100.0 / best_possible_lengths_t
            logger.info(f"BEST POSSIBLE WER: {best_possible_wer} when n-best is {args.nbest}")

            if args.rescoring:
                pass

            if args.spelling_correction:
                pass

        logger.info(
            "| Processed {} sentences ({} tokens) in {:.1f}s ({:.2f}"
            "sentences/s, {:.2f} tokens/s)".format(
                num_sentences,
                gen_timer.n,
                gen_timer.sum,
                num_sentences / gen_timer.sum,
                1.0 / gen_timer.avg,
            )
        )
        logger.info("| {} for extracting ctc emission".format(generator.get_emission_time))
        logger.info("| {} for actual decoding time (viterbi, ngram so on)".format(generator.decoding_time))
        logger.info("| Generate {} with beam={}".format(args.gen_subset, args.beam))

    # return task, wer
    return wer, task


def make_parser():
    parser = options.get_generation_parser()
    parser = add_asr_eval_argument(parser)
    return parser


def cli_main():
    parser = make_parser()
    args = options.parse_args_and_arch(parser)
    # if args.hparam_search:
    #     total_trials = 5
    #     print('\n========================================================')
    #     print('We have to find resocring fusion parameter using Ax...')
    #     print('total trials is {}'.format(total_trials))
    #     print('========================================================\n')

    #     print('args',args)
    #     from ax.service.managed_loop import optimize
    #     best_parameters, values, experiment, model = optimize(
    #         parameters=[
    #             {"name": "rescoring_weight", "type": "range", "bounds": [-5.0, 5.0]},
    #         ],
    #         evaluation_function=main,
    #         objective_name='wer',
    #         total_trials = total_trials,
    #         minimize=True,
    #     )
    main(args)


if __name__ == "__main__":
    cli_main()
