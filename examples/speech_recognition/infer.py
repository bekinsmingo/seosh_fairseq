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
        help="wfstlm on dictonaryoutput units",
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
    parser.add_argument("--rescoring-model", help="lm model for rescoring")
    parser.add_argument("--rescoring-weight", type=float, default=1.0)
    parser.add_argument("--rescoring-word-len-weight", type=float, default=1.0)
    parser.add_argument("--general-rescoring", action="store_true")
    parser.add_argument("--general-rescoring-model", default = None, help="lm model for rescoring")
    parser.add_argument("--general-rescoring-weight", type=float, default=0.0)
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
    with torch.no_grad():
        y = model.forward(x)[0]
        if model.adaptive_softmax is not None:
            logprobs = (model.adaptive_softmax.get_log_prob(y, None).detach().cpu().numpy())
        else:
            logprobs = torch.nn.functional.log_softmax(y, 2).detach().cpu().numpy()

    for index, input_i in enumerate(encoded_input):
        loss = numpy.sum(logprobs[index, numpy.arange(len(input_i)), input_i])
        loss += logprobs[index, len(input_i), fairseq_dict.eos()]
        ppls.append(loss)

        total_loss += loss
        nwords += len(input_i) + 1

    '''
    (Pdb) ' '.join(sentences[0]); x[0]; torch.LongTensor(encoded_input[0]); model.adaptive_softmax.get_log_prob(y, None)[0].max(-1)[1]; logprobs[0, numpy.arange(len(encoded_input[0])), encoded_input[0]]                                                                      
                                                                
    "i am willing to enter into competition with the ancients and feel able to surpass them for since those early days 
    in which i made the medals of pope clement i have learned so much that i can now produce far better pieces of the kind 
    i think i can also outdo the coins i struck for duke alessandro which are still held in high esteem in like manner 
    i could make for you large pieces of gold and silver plate as i did so often for that noble monarch king francis of france 
    thanks to the great conveniences he allowed me without ever losing time for the execution of colossal statues or other works of the sculptor's craft"

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

    tensor([   10,   134,  1376,     7,  1106,    64,  7431,    16,     4, 10432,
                5,   356,   433,     7, 15349,    53,    19,   264,   130,   519,
            233,     9,    34,    10,    93,     4, 16706,     6,  3404,  9543,
            10,    29,   765,    41,   105,    12,    10,    91,    65,  2204,
            194,   209,  1380,     6,     4,   292,    10,   124,    10,    91,
            226, 30331,     4,  8978,    10,   642,    19,  1299, 19243,    34,
            47,   140,   361,     9,   300,  4266,     9,    73,   430,    10,
            62,   129,    19,    17,   325,  1380,     6,   590,     5,  1027,
            2565,    18,    10,    79,    41,   336,    19,    12,   930,  5009,
            321,  3275,     6,   903,  1804,     7,     4,    99, 18188,    11,
            933,    40,   141,   161,  3092,    71,    19,     4,  3615,     6,
            8886,  6828,    46,    88,  1199,     6,     4, 32003,  2603])
    
    tensor([    4,    29,    24,     7,  1697,    64,    81,    16,    17,   268,
                9,     7,    12,     7,    63,    53,     9,    10,    10,   233,
            233,    10,    34,    10,    13,    37,  1672,    10,     4,  7846,
            10,  1611,    55,     7,   105,    36,    10,   134,    65,   767,
                4,    61,  1052,    75,   187,   292,     2,    29,    10,   114,
            63,    63,    53, 21600,     6,    29,     9,     4,  1341,     2,
            38,    65,     9,     9,   300,  8527,     2,  2503,   430,    18,
            91,   129,     4,   261,     8,  7201,     6,   590,     5,  1027,
            34,    34,    82,    29,    19,    10,    19,     4,    22,  1299,
                2,  1341,    10,   903,     2,     7,     4,    94,   385,    34,
            20,    40,     9,    81,  1643,    15,     9,    10,   187,     6,
            15,  1199,     2,     4,  1199,     6,   602,   292,   602,     2,
            10], device='cuda:0')
            
    array([ -2.615 ,  -2.7   ,  -5.297 ,  -0.2228,  -6.863 ,  -0.7495,
            -5.875 ,  -0.1395,  -2.504 ,  -8.67  ,  -2.242 ,  -7.816 ,
            -7.16  ,  -0.0781,  -5.723 ,  -0.1248,  -4.305 ,  -6.55  ,
            -5.242 ,  -4.184 ,  -0.5024,  -3.232 ,  -1.035 ,  -0.7236,
            -4.66  ,  -1.757 , -11.84  ,  -1.67  , -10.555 ,  -4.082 ,
            -0.9688,  -2.02  ,  -3.447 ,  -5.598 ,  -0.2983,  -2.12  ,
            -0.8003,  -1.899 ,  -1.126 ,  -5.266 ,  -7.195 ,  -1.287 ,
            -4.254 ,  -2.688 ,  -4.19  ,  -1.108 ,  -5.52  ,  -4.945 ,
            -2.092 ,  -1.81  ,  -3.822 ,  -6.98  ,  -1.836 , -11.664 ,
            -4.05  ,  -6.39  ,  -3.504 ,  -9.18  ,  -5.527 ,  -4.78  ,
            -2.094 ,  -2.988 ,  -6.29  ,  -0.5767,  -1.183 ,  -1.285 ,
            -1.819 , -10.97  ,  -0.0951,  -1.179 ,  -4.45  ,  -2.348 ,
            -5.49  ,  -3.62  ,  -7.07  ,  -4.54  ,  -0.1964,  -1.231 ,
            -1.785 ,  -1.103 ,  -4.457 ,  -3.604 ,  -2.438 ,  -2.36  ,
            -5.875 ,  -6.848 ,  -1.839 ,  -5.17  ,  -3.281 ,  -3.613 ,
            -5.516 ,  -2.512 ,  -2.635 ,  -0.9287, -10.234 ,  -0.4602,
            -0.908 ,  -3.922 , -11.08  ,  -2.99  ,  -4.92  ,  -0.54  ,
            -6.17  ,  -3.47  ,  -5.57  ,  -3.275 ,  -3.29  ,  -2.719 ,
            -5.24  ,  -0.0607, -12.1   ,  -2.686 ,  -3.887 ,  -2.244 ,
            -0.75  ,  -0.3792,  -3.389 ,  -6.965 ,  -4.855 ], dtype=float16)
    '''


    return ppls, total_loss, nwords


def predict_batch_for_rescoring_roberta(sentences, model, fairseq_dict, max_len=None):
    encoded_input = []
    input_length = []
    padded_input = []
    ppls = []

    total_loss = 0.0
    nwords = 0

    for sentence in sentences:
        encoded = model.encode(sentence).tolist()
        encoded_input.append(encoded)
        input_length.append(len(encoded))

    max_len = max(input_length)


    for inp in encoded_input: 
        if len(inp) < max_len:
            padded_input.append(
                inp + [fairseq_dict.pad()] * (max_len - len(inp))
            )
        else:
            padded_input.append(inp)
    x = torch.LongTensor(padded_input).cuda()

    with torch.no_grad():
        y = model.model(x)[0]
        # logprobs = model.model.get_normalized_probs(y, True)
        logprobs = torch.nn.functional.log_softmax(y, 2).detach().cpu().numpy()

    for index, input_i in enumerate(encoded_input):
        loss = numpy.sum(logprobs[index, numpy.arange(len(input_i)), input_i])
        ppls.append(loss)

        total_loss += loss
        nwords += len(input_i)

    '''
    Example 1
    (Pdb) sentences[0]; model.decode(x[0].cpu()); model.decode(torch.nn.functional.log_softmax(y, 2)[0].max(-1)[1].cpu())

    'This he set in a saucer wetted with a little water and after waiting a short time smelt and tasted it and then he took 
    out of the chest a booklet wherein he read a while and said weeping know o ye passengers that in this book is a marvellous 
    matter denoting that whoso come hither shall surely die without hope of escape for that this ocean is called the sea of 
    the clime of the king wherein is a sepulchre of our lord solomon son of david on both be peace.'

    'This he set in a saucer wetted with a little water and after waiting a short time smelt and tasted it and then he took 
    out of the chest a booklet wherein he read a while and said weeping know o ye passengers that in this book is a marvellous 
    matter denoting that whoso come hither shall surely die without hope of escape for that this ocean is called the sea of 
    the clime of the king wherein is a sepulchre of our lord solomon son of david on both be peace.'

    'This he set in a saucer wetted with a little water and after waiting a short time smelt and tasted it and then he took 
    out of the chest a booklet wherein he read a while and said, know o ye, that in this book is a marvellous 
    matter denoting that whoso come hither shall surely die without hope of escape for that this ocean is called the sea of 
    the clime of the king wherein is a sepulchre of our lord solomon son of david on both be peace the'

    missing words => weeping, passengers
    added words => the (last)

    (Pdb) logprobs[0, numpy.arange(len(encoded_input[0])), encoded_input[0]]
    array([-1.6475e-04, -7.5817e-05, -2.3007e-05, -1.2624e-04, -1.0848e-05,
        -5.2452e-06, -1.1921e-07, -2.3842e-07, -2.9778e-04, -1.1702e-03,
        -7.6175e-05, -1.1921e-06, -1.1921e-05, -7.0333e-06, -3.3932e-03,
        -1.4424e-05, -1.0300e-04, -3.5763e-07, -2.6655e-04, -2.9087e-05,
        -6.5565e-06, -2.0027e-05, -1.0097e-04, -1.8477e-05, -8.2254e-06,
        -1.2083e-03, -4.7207e-05, -1.6212e-05, -1.7643e-05, -1.7881e-06,
        -2.9731e-04, -4.2319e-05, -2.2717e-03, -7.1526e-06, -2.2449e-03,
        -3.1662e-03, -6.4373e-06, -5.4002e-05, -3.5763e-07, -1.6470e-03,
        -1.6689e-06, -5.7578e-05, -1.3711e+01, -2.4609e-01, -9.9304e-02,
        -2.0866e-03, -1.0258e+01, -1.4067e-05, -9.8944e-06, -1.5020e-05,
        -1.0920e-04, -9.7752e-06, -2.8610e-06, -1.1826e-04, -1.3590e-05,
        -1.4782e-05, -2.9526e-02, -2.3770e-04, -4.7946e-04, -7.8678e-06,
        -8.3447e-07, -9.7752e-06, -1.4186e-04, -1.6606e-04, -1.8954e-05,
        -3.2496e-04, -1.2398e-04, -1.2994e-05, -1.3924e-04, -1.5020e-05,
        -1.1530e-03, -6.2256e-03, -2.1935e-05, -1.5378e-05, -1.5320e-02,
        -4.7684e-07, -1.9908e-05, -3.7551e-05, -1.9228e-04, -1.5974e-05,
        -7.3633e-01, -1.4257e-03, -2.8885e-02, -6.3181e-06, -7.6514e-01,
        -3.1982e-02, -1.9817e-03, -1.3113e-05, -1.2040e-05, -5.2452e-06,
        -8.3447e-07, -1.5488e-03, -4.8923e-04, -3.2425e-05, -6.7472e-05,
        -5.2404e-04, -1.9760e-03, -1.0884e-04, -5.1975e-05, -2.4676e-04,
        -8.4961e-02, -1.0669e-01, -1.4541e+00, -1.6556e-02, -2.6840e-02,
        -9.5654e-01, -2.2650e-05], dtype=float16)

    (Pdb) encoded_input[0]
    [0, 713, 37, 278, 11, 10, 2241, 43886, 7727, 5357, 19, 10, 410, 514, 8, 71, 2445, 10, 765, 86, 5278, 6607, 
    8, 29143, 24, 8, 172, 37, 362, 66, 9, 5, 7050, 10, 39521, 26134, 37, 1166, 10, 150, 8, 26, 39423, 216, 1021, 
    32440, 3670, 14, 11, 42, 1040, 16, 10, 4401, 22752, 1827, 948, 3069, 12653, 14, 8401, 18865, 283, 48586, 
    5658, 8349, 1597, 396, 1034, 9, 5111, 13, 14, 42, 6444, 16, 373, 5, 3342, 9, 5, 3741, 4235, 9, 5, 8453, 
    26134, 16, 10, 45821, 922, 611, 241, 9, 84, 30722, 9281, 28344, 979, 9, 44009, 15, 258, 28, 1987, 4, 2]
    '''

    return ppls, total_loss, nwords


########################################################################
###################               main               ###################
########################################################################


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

    rescoring_model = None
    if args.rescoring : 
        device = 'cuda'

        print(args.rescoring_model)
        path, checkpoint = os.path.split(args.rescoring_model)

        overrides = {
            "task": 'language_modeling',
            "data": path,
        }
        logger.info("| loading rescoring lm model from {}".format(args.rescoring_model))
        rescoring_models, rescoring_saved_cfg, rescoring_task = checkpoint_utils.load_model_ensemble_and_task(
            utils.split_paths(args.rescoring_model, separator="\\"),
            arg_overrides=overrides,
            strict=True,
        )

        rescoring_model = rescoring_models[0]
        rescoring_model.eval().cuda()
        rescoring_model.make_generation_fast_()
        if rescoring_saved_cfg.common.fp16:
            rescoring_model.half()
        rescoring_model = rescoring_model.decoder

        dict_path = os.path.join(path,'dict.txt')
        rescoring_dict = Dictionary.load(dict_path)

        general_rescoring_model = None
        general_rescoring_dict = None
        if args.general_rescoring:
            ## roberta
            path, checkpoint = os.path.split(args.general_rescoring_model)
            from fairseq.models.roberta import RobertaModel
            general_rescoring_model = RobertaModel.from_pretrained(path, checkpoint_file=checkpoint)
            general_rescoring_model.eval().cuda()
            general_rescoring_model.half()

            dict_path = os.path.join(path,'dict.txt')
            general_rescoring_dict = Dictionary.load(dict_path)


        # path, checkpoint = os.path.split(args.rescoring_model)
        # dict_path = os.path.join(path,'dict.txt')
        # transformer_rescoring_dict = Dictionary.load(dict_path)
        # model = load_rescoring_model(args.rescoring_model, 'transformer', dict_path)
        # model.eval().cuda()

        # model_name = "gpt2"
        # # model_name = "gpt2-large"
        # from transformers import GPT2LMHeadModel, GPT2TokenizerFast, AutoModelForCausalLM, AutoTokenizer

        # gpt_rescoring_model = (
        #     AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name, is_decoder=True)
        #     .to(device)
        #     .eval()
        # )
        # gpt_rescoring_tokenizer = AutoTokenizer.from_pretrained(model_name)
        # gpt_rescoring_tokenizer.pad_token = gpt_rescoring_tokenizer.eos_token

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

    gen_timer_for_rescoring = StopwatchMeter()

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

            '''
            (Pdb) sample.keys()
            dict_keys(['id', 'net_input', 'target_lengths', 'ntokens', 'target'])
            sample['net_input']['source'].size() # speech input                                                                                                         
            torch.Size([7, 552160])
            (Pdb) sample['target'].size() # sentence target
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
            num_generated_tokens = sum(len(h[0]["tokens"]) for h in hypos)

            gen_timer.stop(num_generated_tokens)



            gen_timer_for_rescoring.start()

            # total_loss = 0.0
            # nwords = 0.0
            # batch = []
            # original_lines = []
            # max_len = 0

            if args.rescoring:
                for i, nbest_hypos in enumerate(hypos):
                    batch = []
                    batch_for_roberta = []
                    max_len = 0
                    for j, n_th_hypo in enumerate(nbest_hypos):

                        # for tfm
                        sent = n_th_hypo['words']
                        score = n_th_hypo['score'] # am + lm + word_penalty
                        hypos[i][j]['wl_len'] = len(sent) + len("".join(sent)) # word length + char length 
                        sent = " ".join(sent).lower().split()

                        batch.append(sent)

                        # for roberta
                        tmp = " ".join(n_th_hypo['words'])
                        if len(tmp)>1:
                            batch_for_roberta.append(tmp[0].upper()+tmp[1:].lower()+'.')
                        else:
                            batch_for_roberta.append(' ')

                    max_len = len(sorted(batch, key=lambda x: len(x))[-1])

                    ppls, loss_batch, nwords_batch = predict_batch_for_rescoring(batch, rescoring_model, rescoring_dict, max_len) # -39, -40

                    if args.general_rescoring:
                        general_ppls, general_loss_batch, general_nwords_batch = predict_batch_for_rescoring_roberta(batch_for_roberta, general_rescoring_model, general_rescoring_dict)
                        # import pdb; pdb.set_trace()

                    '''
                    ## 20L TFM # in-domain
                    (Pdb) ppls
                    [-450.0, -458.0, -454.0, -451.5, -455.8, -452.2, -462.0, -450.8, 
                    -459.8, -455.5, -464.8, -449.2, -460.2, -458.5, -470.8, -456.8, 
                    -463.5, -456.5, -458.5, -478.5, -464.8, -462.2, -464.8, -454.2, 
                    -464.8, -455.0, -454.5, -453.2, -451.0, -459.2, -470.8, -459.8, 
                    -455.8, -453.2, -460.8, -465.0, -456.0, -462.0, -462.2, -457.0, 
                    -460.2, -454.0, -465.2, -473.0, -454.5, -463.8, -450.2, -460.5, 
                    -463.2, -456.2]

                    # 중간에 -458.0 과 - 459.8 의 loss가 역전됨.

                    ## 24L Roberta (Large) # out-domain
                    (Pdb) general_ppls
                    [-3.613, -10.66, -3.9, -3.613, -4.22, -3.396, -10.58, -3.703, 
                    -10.2, -6.188, -11.27, -3.49, -10.48, -10.96, -3.418, -3.822, 
                    -13.99, -3.984, -11.484, -10.664, -11.64, -3.531, -10.05, -4.03, 
                    -3.582, -3.494, -3.51, -3.523, -3.543, -3.86, -10.734, -3.525, 
                    -3.56, -4.07, -5.113, -4.035, -3.904, -3.662, -10.65, -3.547, 
                    -3.275, -6.043, -6.836, -11.41, -3.637, -15.15, -3.527, -4.27, 
                    -10.77, -2.256]

                    ## 12L Roberta (Base)
                    (Pdb) general_ppls
                    [-1.846, -6.0, -1.913, -1.834, -2.822, -1.774, -5.78, -4.473, 
                    -5.82, -3.25, -6.86, -1.707, -6.703, -8.71, -1.296, -2.508, 
                    -6.977, -2.342, -5.598, -5.113, -5.977, -2.229, -6.05, -10.11, 
                    -1.957, -1.839, -6.445, -1.879, -1.84, -4.195, -5.836, -1.863, 
                    -8.45, -2.316, -2.562, -1.961, -1.904, -3.28, -13.26, -2.389, 
                    -1.985, -4.484, -3.49, -5.89, -1.901, -13.13, -2.184, -2.826, 
                    -5.86, -1.747]
                    '''

                    for j, n_th_hypo in enumerate(nbest_hypos):
                        ppl = ppls[j]
                        general_ppl = general_ppls[j] if args.general_rescoring else 0
                        hypos[i][j]['rescoring_lm_ppl'] = ppl
                        hypos[i][j]['general_rescoring_lm_ppl'] = general_ppl
                        hypos[i][j]['total_score'] = (
                            n_th_hypo['am_score'] 
                            + args.rescoring_weight * ppl 
                            + args.general_rescoring_weight * general_ppl
                            + args.rescoring_word_len_weight * n_th_hypo['wl_len'] 
                            )

                    # hypos[i] = sorted(nbest_hypos, key=lambda x: -x["rescoring_lm_ppl"])
                    hypos[i] = sorted(nbest_hypos, key=lambda x: -x["total_score"])

            num_generated_tokens_for_rescoring = sum(len(h[0]["tokens"]) for h in hypos)
            gen_timer_for_rescoring.stop(num_generated_tokens_for_rescoring)

                #         import pdb; pdb.set_trace()
                #         if (len(batch) + 1) * numpy.maximum(max_len, len(sent)) > args.max_tokens:
                #             if len(batch) == 0 :
                #                 batch.append(sent)
                #                 max_len = len(sent)
                #                 continue

                #             ppls, loss_batch, nwords_batch = predict_batch_for_rescoring(batch, rescoring_model.decoder, rescoring_dict, max_len)
                #             # ppls, loss_batch, nwords_batch = predict_batch_for_rescoring(batch, model, rescoring_dict, max_len)

                #             total_loss += loss_batch
                #             nwords += nwords_batch

                #             batch = [sent]
                #             max_len = len(sent)
                #         else:
                #             batch.append(sent)
                #             max_len = numpy.maximum(max_len, len(sent))

                #     import pdb; pdb.set_trace()

                # if len(batch) > 0:
                #     ppls, loss_batch, nwords_batch = predict_batch_for_rescoring(batch, rescoring_model, rescoring_dict, max_len)
                #     total_loss += loss_batch
                #     nwords += nwords_batch


            # if args.rescoring:

            #     reordered_hypos = []

            #     for nbest_hypos in hypos:
            #         inputs = []
            #         beams = []
            #         for hypo in nbest_hypos:
            #             inputs.append(' '.join(hypo['words']))
            #         encoded = rescoring_tokenizer(inputs, return_tensors='pt', padding=True, truncation=True)['input_ids'].to(rescoring_model.device)
            #         inputs_mask = (encoded != rescoring_tokenizer.pad_token_id)

            #         with torch.no_grad():
            #             output = rescoring_model(input_ids=encoded, attention_mask=inputs_mask)
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
            #                 # "total_score" : hypo['score'] + self.rescoring_weight * rescored_result
            #                 })

            #         '''
            #         ## flashlight decoder score (am + lm beam search maybe?)
            #         (Pdb) torch.tensor([tmp['score'] for tmp in nbest_hypos])
            #         tensor([21031.2617, 21030.2500, 21026.6172, 21026.3652, 21025.8457, 21025.7324,
            #                 21025.6055, 21025.4688])

            #         ## log softmax sum
            #         (Pdb) rescored_results
            #         tensor([-711.9091, -710.1793, -716.3192, -712.5317, -713.6451, -716.3398,
            #                 -714.7936, -702.9783], device='cuda:0') # 

            #         ## logit sum
            #         (Pdb) neural_lm_score
            #         tensor([-20230.5352, -20261.6445, -20284.9336, -20233.1250, -20201.2852,
            #                 -20069.5234, -20313.2480, -20026.4883], device='cuda:0')

            #         ## GPT 기준 best path
            #         (Pdb) inputs[-1] 
            #         "I AM WILLING TO ENTER INTO COMPETITION WITH THE ANCIENTS AND FEEL ABLE TO SURPASS THEM FOR SINCE THOSE EARLY DAYS IN 
            #         WHICH I MADE THE MEDALS OF POPE CLEMENT I HAVE LEARNED SO MUCH THAT I CAN NOW PRODUCE FAR BETTER PIECES OF THE KIND I THINK 
            #         I CAN ALSO OUTDO THE COINS I STRUCK FOR DUKE ALESSANDRO WHICH ARE STILL HELD IN HIGH ESTEEM IN LIKE MANNER I COULD MAKE FOR YOU 
            #         LARGE PIECES OF GOLD AND SILVER PLATE AS I DID SO OFTEN FOR THAT NOBLE MONARCH KING FRANCIS OF FRANCE THANKS TO THE GREAT CONVENIENCE 
            #         HE ALLOWED ME WITHOUT EVER LOSING TIME FOR THE EXECUTION OF COLOSSAL STATUES OR OTHER WORKS OF THE SCULPTOR'S CRAFT"

            #         # am 기준 best path
            #         (Pdb) inputs[0]
            #         "I AM WILLING TO ENTER INTO COMPETITION WITH THE ANCIENTS AND FEEL ABLE TO SURPASS THEM FOR SINCE THOSE EARLY DAYS IN 
            #         WHICH I MADE THE MEDALS OF POPE CLEMENT I HAVE LEARNED SO MUCH THAT I CAN NOW PRODUCE FAR BETTER PIECES OF THE KIND I THINK 
            #         I CAN ALSO OUTDO THE COINS I STRUCK FOR DUKE ALESSANDRO WHICH ARE STILL HELD IN HIGH ESTEEM IN LIKE MANNER I COULD MAKE FOR YOU 
            #         LARGE PIECES OF GOLD AND SILVER PLATE AS I DID SO OFTEN FOR THAT NOBLE MONARCH KING FRANCIS OF FRANCE THANKS TO THE GREAT CONVENIENCES 
            #         HE ALLOWED ME WITHOUT EVER LOSING TIME FOR THE EXECUTION OF COLOSSAL STATUES OR OTHER WORKS OF THE SCULPTOR'S CRAFT"
            #         '''

            #         # Original Rescoring log P_{AM} (y|x) + \alpha1 log P_{LM1}(y) + \beta |y| + \alpha2 log P_{LM2}(y)

            #         # on the fly
            #         sorted_beams = sorted(beams, reverse=True, key=lambda x:x['rescoring'])
            #         # sorted_beams = sorted(beams, reverse=False, key=lambda x:x['rescoring'])

            #         import pdb
            #         pdb.set_trace()

            #         reordered_hypos.append(sorted_beams)

            #     hypos = reordered_hypos

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


            wps_meter.update(num_generated_tokens)
            t.log({"wps": round(wps_meter.avg)})
            num_sentences += (
                sample["nsentences"] if "nsentences" in sample else sample["id"].numel()
            )


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
        if args.rescoring:
            logger.info("| {} for rescoring with {}-best lists".format(gen_timer_for_rescoring.sum ,args.nbest))
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
