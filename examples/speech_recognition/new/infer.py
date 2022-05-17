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


def predict_batch_for_rescoring_roberta_pll(sentences, model, fairseq_dict, max_len=None):
    encoded_input = []
    input_length = []
    padded_input = []
    ppls = []

    total_loss = 0.0
    nwords = 0

    mask_token = model.task.source_dictionary.indices['<mask>']
    sos_token = model.task.source_dictionary.indices['<s>']
    eos_token = model.task.source_dictionary.indices['</s>']

    # import pdb; pdb.set_trace()

    for sentence in sentences:
        encoded = model.encode(sentence).tolist()
        encoded_tensor = torch.LongTensor(encoded)

        repeated_encoded_tensor = encoded_tensor.repeat(len(encoded)-2,1)

        diagonal_mask = torch.cat((torch.diag((torch.zeros(len(encoded)-2)+1).long(),1)[:-1],torch.zeros(len(encoded)-2,1).long()),1)
        x = repeated_encoded_tensor.masked_fill_(diagonal_mask.bool(),mask_token).cuda()

        log_prob_index = encoded_tensor[1:-1]

        '''
        (Pdb) encoded_tensor
        tensor([    0,   100,   524,  2882,     7,  2914,    88,  1465,    19,     5,
                41, 48319,     8,   619,   441,     7, 14874,   106,    13,   187,
                167,   419,   360,    11,    61,   939,   156,     5, 10214,     9,
                16627, 12479,  1757,   939,    33,  2435,    98,   203,    14,   939,
                64,   122,  2592,   444,   357,  3745,     9,     5,   761,   939,
                206,   939,    64,    67,    66,  5016,     5, 15750,   939,  2322,
                13,   385,  7480,  1076,  3361, 22967,    61,    32,   202,   547,
                11,   239, 38638,    11,   101,  4737,   939,   115,   146,    13,
                47,   739,  3745,     9,  1637,     8,  4334,  5299,    25,   939,
                222,    98,   747,    13,    14, 25097, 20303,  8453, 13638,   354,
                    9,  6664,  2389,  2446,     7,     5,   372, 16602, 27797,    37,
                1220,   162,   396,   655,  2086,    86,    13,     5,  7356,     9,
                33568, 19638,    50,    97,  1364,     9,     5, 21546,   368,    18,
                6306,     4,     2])

        (Pdb) repeated_encoded_tensor
        tensor([[   0,  100,  524,  ..., 6306,    4,    2],
                [   0,  100,  524,  ..., 6306,    4,    2],
                [   0,  100,  524,  ..., 6306,    4,    2],
                ...,
                [   0,  100,  524,  ..., 6306,    4,    2],
                [   0,  100,  524,  ..., 6306,    4,    2],
                [   0,  100,  524,  ..., 6306,    4,    2]])
        (Pdb) repeated_encoded_tensor.size()
        torch.Size([131, 133])

        (Pdb) diagonal_mask
        tensor([[0, 1, 0,  ..., 0, 0, 0],
                [0, 0, 1,  ..., 0, 0, 0],
                [0, 0, 0,  ..., 0, 0, 0],
                ...,
                [0, 0, 0,  ..., 0, 0, 0],
                [0, 0, 0,  ..., 1, 0, 0],
                [0, 0, 0,  ..., 0, 1, 0]])

        (Pdb) x
        tensor([[    0, 50264,   524,  ...,  6306,     4,     2],
                [    0,   100, 50264,  ...,  6306,     4,     2],
                [    0,   100,   524,  ...,  6306,     4,     2],
                ...,
                [    0,   100,   524,  ...,  6306,     4,     2],
                [    0,   100,   524,  ..., 50264,     4,     2],
                [    0,   100,   524,  ...,  6306, 50264,     2]], device='cuda:0')
        '''

        # tmp = torch.range(1,len(encoded)-2).long()
        # tmp = torch.cat((torch.zeros(len(encoded)-2,1).long(),torch.diag(tmp),torch.zeros(len(encoded)-2,1).long()),1).unsqueeze(-1).cuda()

        with torch.no_grad():
            y = model.model(x)[0]
            logprobs = torch.nn.functional.log_softmax(y, 2).detach().cpu().numpy()
        
        # import pdb; pdb.set_trace()

        '''
        torch.return_types.max(
        values=tensor([[-2.4796e-05, -2.3877e-01, -1.1320e-03,  ..., -1.0620e-02,
                -1.1921e-05, -1.1730e-04],
                [-1.2159e-05, -2.3346e-03, -1.2128e-01,  ..., -1.2001e-02,
                -1.1563e-05, -1.1480e-04],
                [-1.2398e-05, -5.2071e-03, -8.3351e-04,  ..., -1.1269e-02,
                -1.1802e-05, -1.1468e-04],
                ...,
                [-5.2452e-06, -1.3901e-02, -6.8378e-04,  ..., -9.3231e-03,
                -1.7047e-05, -1.2195e-04],
                [-5.0068e-06, -1.4557e-02, -6.5470e-04,  ..., -3.4023e+00,
                -1.4186e-05, -1.2791e-04],
                [-7.5102e-06, -8.3237e-03, -6.6423e-04,  ..., -1.0826e-02,
                -6.7949e-06, -4.1723e-04]], device='cuda:0', dtype=torch.float16),
        indices=tensor([[   0,  939,  524,  ..., 6306,    4,    2],
                [   0,  100,  524,  ..., 6306,    4,    2],
                [   0,  100,  524,  ..., 6306,    4,    2],
                ...,
                [   0,  100,  524,  ..., 6306,    4,    2],
                [   0,  100,  524,  ..., 1808,    4,    2],
                [   0,  100,  524,  ..., 6306,    4,    2]], device='cuda:0'))

        (Pdb) encoded_tensor
        tensor([    0,   100,   524,  2882,     7,  2914,    88,  1465,    19,     5,
                41, 48319,     8,   619,   441,     7, 14874,   106,    13,   187,
                167,   419,   360,    11,    61,   939,   156,     5, 10214,     9,
                16627, 12479,  1757,   939,    33,  2435,    98,   203,    14,   939,
                64,   122,  2592,   444,   357,  3745,     9,     5,   761,   939,
                206,   939,    64,    67,    66,  5016,     5, 15750,   939,  2322,
                13,   385,  7480,  1076,  3361, 22967,    61,    32,   202,   547,
                11,   239, 38638,    11,   101,  4737,   939,   115,   146,    13,
                47,   739,  3745,     9,  1637,     8,  4334,  5299,    25,   939,
                222,    98,   747,    13,    14, 25097, 20303,  8453, 13638,   354,
                    9,  6664,  2389,  2446,     7,     5,   372, 16602, 27797,    37,
                1220,   162,   396,   655,  2086,    86,    13,     5,  7356,     9,
                33568, 19638,    50,    97,  1364,     9,     5, 21546,   368,    18,
                6306,     4,     2])

        (Pdb) sentence
        "I am willing to enter into competition with the ancients and feel able to surpass them for since those early days 
        in which i made the medals of pope clement i have learned so much that i can now produce far better pieces of the kind 
        i think i can also outdo the coins i struck for duke alessandro which are still held in high esteem in like manner 
        i could make for you large pieces of gold and silver plate as i did so often for that noble monarch king francis of france 
        thanks to the great conveniences he allowed me without ever losing time for the execution of colossal statues or other works of the sculptor's craft."

        # 나는 스테이크를 먹었다 교도소에서.
        '''


        loss = 0.0
        for i, index in enumerate(log_prob_index):
            loss += logprobs[i,i+1,index]
        ppls.append(loss)

        total_loss += loss
        nwords += len(log_prob_index)

    return ppls, total_loss, nwords



class InferenceProcessor:
    cfg: InferConfig

    def __init__(self, cfg: InferConfig) -> None:
        self.cfg = cfg
        self.task = tasks.setup_task(cfg.task)

        # import pdb; pdb.set_trace()

        self.use_fp16 = False
        self.use_cuda = (not cfg.common.cpu and not torch.cuda.is_available())

        models, saved_cfg = self.load_model_ensemble()
        self.models = models
        self.saved_cfg = saved_cfg
        self.tgt_dict = self.task.target_dictionary

        # import pdb; pdb.set_trace()

        self.task.load_dataset(
            self.cfg.dataset.gen_subset,
            task_cfg=saved_cfg.task,
        )
        # cfg.decoding.type = 'pyctcdecoder'
        self.generator = Decoder(cfg.decoding, self.tgt_dict)
        # import pdb; pdb.set_trace()
        self.gen_timer = StopwatchMeter()
        self.gen_timer_for_rescoring = StopwatchMeter()
        self.wps_meter = TimeMeter()
        self.num_sentences = 0

        self.total_errors = 0
        self.total_length = 0
        self.best_total_errors = 0
        self.best_total_length = 0 

        self.hypo_words_file = None
        self.hypo_units_file = None
        self.ref_words_file = None
        self.ref_units_file = None

        self.progress_bar = self.build_progress_bar()

        self.rescoring = cfg.decoding.rescoring
        self.rescoring_weight = cfg.decoding.rescoringweight
        self.rescoring_word_len_weight = cfg.decoding.rescoringwordlenweight

        # self.general_rescoring = cfg.decoding.generalrescoring
        # self.general_rescoring_weight = cfg.decoding.generalrescoringweight

        self.save_result = cfg.decoding.saveresult
        self.save_result_path = cfg.decoding.saveresultpath

        # # original code
        # logger.info("| loading rescoring lm model from {}".format(cfg.decoding.rescoringlmpath))
        # path, checkpoint = os.path.split(self.cfg.decoding.rescoringlmpath)
        # dict_path = os.path.join(path,'dict.txt')
        # self.rescoring_dict = Dictionary.load(dict_path)
        # self.rescoring_model = load_rescoring_model(self.cfg.decoding.rescoringlmpath, 'transformer', dict_path)

        if self.rescoring : 
                
            # my model
            print('cfg.decoding.rescoringlmpath',cfg.decoding.rescoringlmpath)
            path, checkpoint = os.path.split(cfg.decoding.rescoringlmpath)
            dict_path = os.path.join(path,'dict.txt')
            self.rescoring_dict = Dictionary.load(dict_path)

            if 'bert' in cfg.decoding.rescoringlmpath:
                ## roberta
                from fairseq.models.roberta import RobertaModel
                self.rescoring_model = RobertaModel.from_pretrained(path, checkpoint_file=checkpoint)
                # self.rescoring_model.model.make_generation_fast_()
                self.rescoring_model.eval().cuda()
                self.rescoring_model.half()
                self.rescoring_model_type = 'fairseq_bert'

                # ## for debugging tfm vs roberta
                # tmp_path = '/workspace/librispeech_model/decoder/lm_librispeech_word_transformer/lm_librispeech_word_transformer.pt'
                # path, checkpoint = os.path.split(tmp_path)
                # dict_path = os.path.join(path,'dict.txt')
                # self.tfm_rescoring_dict = Dictionary.load(dict_path)

                # overrides = {
                #     "task": 'language_modeling',
                #     "data": path,
                # }
                # logger.info("| loading rescoring lm model from {}".format(tmp_path))
                # rescoring_models, rescoring_saved_cfg, rescoring_task = checkpoint_utils.load_model_ensemble_and_task(
                #     utils.split_paths(tmp_path, separator="\\"),
                #     arg_overrides=overrides,
                #     strict=True,
                # )

                # self.tfm_rescoring_model = rescoring_models[0]
                # self.tfm_rescoring_model.eval().cuda()
                # self.tfm_rescoring_model.make_generation_fast_()
                # if rescoring_saved_cfg.common.fp16:
                #     self.tfm_rescoring_model.half()
                # self.tfm_rescoring_model = self.tfm_rescoring_model.decoder

            else:
                overrides = {
                    "task": 'language_modeling',
                    "data": path,
                }
                logger.info("| loading rescoring lm model from {}".format(cfg.decoding.rescoringlmpath))
                rescoring_models, rescoring_saved_cfg, rescoring_task = checkpoint_utils.load_model_ensemble_and_task(
                    utils.split_paths(cfg.decoding.rescoringlmpath, separator="\\"),
                    arg_overrides=overrides,
                    strict=True,
                )

                self.rescoring_model = rescoring_models[0]
                self.rescoring_model.eval().cuda()
                self.rescoring_model.make_generation_fast_()
                if rescoring_saved_cfg.common.fp16:
                    self.rescoring_model.half()
                self.rescoring_model = self.rescoring_model.decoder
                self.rescoring_model_type = 'fairseq_tfm_decoder'

            # else:
            #     device = 'cuda'
            #     model_name = "gpt2"
            #     # model_name = "gpt2-large"
            #     from transformers import GPT2LMHeadModel, GPT2TokenizerFast, AutoModelForCausalLM, AutoTokenizer
            #     self.rescoring_model = (
            #         AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name, is_decoder=True)
            #         .to(device)
            #         .eval()
            #     )
            #     self.rescoring_dict = AutoTokenizer.from_pretrained(model_name)
            #     self.rescoring_dict.pad_token = self.rescoring_dict.eos_token

            # import pdb; pdb.set_trace()

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


    def optimize_model(self, model: FairseqModel, model_cfg) -> None:
        model.make_generation_fast_()
        # model.half()
        # self.use_fp16 = True
        if (model_cfg.common.fp16) and (torch.cuda.get_device_capability(0)[0] > 6):
            model.half()
            self.use_fp16 = True
        if not self.cfg.common.cpu:
            model.cuda()
        model.eval()

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
        # import pdb; pdb.set_trace()
        
        '''
        (Pdb) self.cfg.common.fp16
        False
        (Pdb) saved_cfg.common.fp16
        True

        (Pdb) models[0].w2v_encoder.w2v_model.encoder.layers[0].fc1.weight.dtype                                                                            
        torch.float32
        (Pdb) models[0].w2v_encoder.w2v_model.encoder.layers[0].fc1.weight
        Parameter containing:
        tensor([[ 0.0888,  0.2118, -0.1012,  ...,  0.0049,  0.1906,  0.1567],
                [-0.0044,  0.0175, -0.0861,  ...,  0.0859, -0.0656,  0.0097],
                [-0.0216, -0.0445, -0.0444,  ..., -0.3267, -0.0875, -0.0676],
                ...,
                [-0.0927, -0.1041, -0.1175,  ...,  0.0565, -0.0975, -0.0226],
                [ 0.1456,  0.1434, -0.0168,  ..., -0.1377,  0.0690, -0.1309],
                [-0.0152,  0.0212, -0.0604,  ...,  0.1694, -0.1129, -0.1088]],
            requires_grad=True)
        (Pdb) models[0].half()

        (Pdb) models[0].w2v_encoder.w2v_model.encoder.layers[0].fc1.weight
        Parameter containing:
        tensor([[ 0.0888,  0.2118, -0.1012,  ...,  0.0049,  0.1906,  0.1567],
                [-0.0044,  0.0175, -0.0861,  ...,  0.0859, -0.0656,  0.0097],
                [-0.0216, -0.0445, -0.0444,  ..., -0.3267, -0.0875, -0.0676],
                ...,
                [-0.0927, -0.1041, -0.1175,  ...,  0.0565, -0.0975, -0.0226],
                [ 0.1456,  0.1434, -0.0168,  ..., -0.1377,  0.0690, -0.1309],
                [-0.0152,  0.0212, -0.0604,  ...,  0.1694, -0.1129, -0.1088]],
            dtype=torch.float16, requires_grad=True)
        (Pdb) models[0].w2v_encoder.w2v_model.encoder.layers[0].fc1.weight.dtype
        torch.float16
        '''

        for model in models:
            self.optimize_model(model, saved_cfg)
        return models, saved_cfg

    def get_dataset_itr(self, disable_iterator_cache: bool = False) -> None:
        # import pdb; pdb.set_trace()
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
        # import pdb; pdb.set_trace()
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
        hypos,
        sid: int,
        batch_id: int,
    ) -> Tuple[int, int]:
        speaker = None  # Speaker can't be parsed from dataset.

        if "target_label" in sample:
            toks = sample["target_label"]
        else:
            toks = sample["target"]
        toks = toks[batch_id, :]

        all_wers = []

        # 1. Processes target.
        target_tokens = utils.strip_pad(toks, self.tgt_dict.pad())
        tgt_pieces = self.tgt_dict.string(target_tokens.int().cpu())
        tgt_words = post_process(tgt_pieces, self.cfg.common_eval.post_process)
        tgt_words = tgt_words.upper() # upper; necessary for LM traind with lower word 

        if self.cfg.decoding.type == 'pyctcdecoder':
            hyp_words = hypos
            if self.cfg.decoding.results_path is not None:
                print(f"{hyp_words} ({speaker}-{sid})", file=self.hypo_words_file)
                print(f"{tgt_words} ({speaker}-{sid})", file=self.ref_words_file)
            if not self.cfg.common_eval.quiet:
                logger.info(f"HYPO : {hyp_words}")
                logger.info(f"TARG : {tgt_words}")
                logger.info("---------------------")
            wer = editdistance.eval(hyp_words.split(), tgt_words.split())
            return wer, len(tgt_words.split()), wer, len(tgt_words.split())
        else:
            # 2. Processes hypothesis.
            hyp_pieces_list = []
            hyp_words_list = []
            for i, hypo in enumerate(hypos):
                hyp_pieces = self.tgt_dict.string(hypo["tokens"].int().cpu())
                if "words" in hypo:
                    hyp_words = " ".join(hypo["words"])
                else:
                    hyp_words = post_process(hyp_pieces, self.cfg.common_eval.post_process)
                hyp_words = hyp_words.upper() # upper; necessary for LM traind with lower word

                hyp_pieces_list.append(hyp_pieces)
                hyp_words_list.append(hyp_words)
                all_wers.append(editdistance.eval(hyp_words.split(), tgt_words.split()))

            if self.cfg.decoding.results_path is not None:
                print(f"{hyp_pieces_list[0]} ({speaker}-{sid})", file=self.hypo_units_file)
                print(f"{hyp_words_list[0]} ({speaker}-{sid})", file=self.hypo_words_file)
                print(f"{tgt_pieces} ({speaker}-{sid})", file=self.ref_units_file)
                print(f"{tgt_words} ({speaker}-{sid})", file=self.ref_words_file)

            if not self.cfg.common_eval.quiet:
                logger.info(f"HYPO : {hyp_words_list[0]}")
                logger.info(f"TARG : {tgt_words}")
                logger.info("---------------------")

            # import pdb; pdb.set_trace()

            if self.save_result:
                file_name = os.path.join(self.save_result_path, 'w2v2_' + self.cfg.decoding.type + '_nbest_' + str(self.cfg.decoding.nbest))
                if self.cfg.decoding.rescoringlmpath:
                    file_name = file_name + '_rescoring'
                with open(file_name, "a") as fout:
                    for i, (hyp, hyp_word, wer) in enumerate(zip(hypos, hyp_words_list, all_wers)):
                        # sid, hyp['am_score'], hyp['lm_score'], hyp['rescoring_lm_ppl'], hyp['total_score'], wer, hyp_word, tgt_words
                        # import pdb; pdb.set_trace()
                        fout.write(
                            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                            sid,
                            wer,
                            hyp['am_score'],
                            hyp['lm_score'], 
                            hyp['score'], 
                            hyp['rescoring_lm_ppl'] if 'rescoring_lm_ppl' in hyp.keys() else 0, 
                            hyp['total_score'] if 'total_score' in hyp.keys() else 0, 
                            hyp_word, 
                            tgt_words)
                            )

            return all_wers[0], len(tgt_words.split()), min(all_wers), len(tgt_words.split())

    def apply_half(self, t):
        if t.dtype is torch.float32:
            return t.to(dtype=torch.half)
        return t


    def process_sample(self, sample: Dict[str, Any]) -> None:

        if self.use_fp16:
            sample = utils.apply_to_sample(self.apply_half, sample)

        self.gen_timer.start()

        '''
        (Pdb) sample.keys()
        dict_keys(['id', 'net_input', 'target_lengths', 'ntokens', 'target'])

        (Pdb) sample['net_input'].keys()
        dict_keys(['source', 'padding_mask'])

        (Pdb) self.models[0].w2v_encoder.w2v_model.encoder.layers[0].fc1.weight.dtype
        torch.float32

        (Pdb) self.models[0].half()
        (Pdb) self.models[0].w2v_encoder.w2v_model.encoder.layers[0].fc1.weight.dtype
        torch.float16

        (Pdb) sample['net_input']['source'].size()
        torch.Size([7, 552160])
        '''

        hypos = self.task.inference_step(
            generator=self.generator,
            models=self.models,
            sample=sample,
        )

        if self.cfg.decoding.type == 'pyctcdecoder':
            # num_generated_tokens = sum(len(h.split()) for h in hypos)
            num_generated_tokens = sum(len(h) for h in hypos)
        else:
            num_generated_tokens = sum(len(h[0]["tokens"]) for h in hypos)
        self.gen_timer.stop(num_generated_tokens)
        self.wps_meter.update(num_generated_tokens)

        # if self.rescoring:
        #     reordered_hypos = []
        #     for nbest_hypos in hypos:
        #         inputs = []
        #         beams = []
        #         for hypo in nbest_hypos:
        #             inputs.append(' '.join(hypo['words']))
        #         encoded = self.rescoring_dict(inputs, return_tensors='pt', padding=True, truncation=True)['input_ids'].to(self.rescoring_model.device)
        #         inputs_mask = (encoded != self.rescoring_dict.pad_token_id)

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

            self.gen_timer_for_rescoring.start()

            for i, nbest_hypos in enumerate(hypos): # B -> nbest 
                batch = []
                # tmp_batch = []
                max_len = 0
                for j, n_th_hypo in enumerate(nbest_hypos):

                    sent = n_th_hypo['words']
                    score = n_th_hypo['score'] # am + lm + word_penalty
                    hypos[i][j]['wl_len'] = len(sent) + len("".join(sent)) # word length + char length 

                    if self.rescoring_model_type == 'fairseq_tfm_decoder':
                        tmp = " ".join(sent).lower().split()
                        batch.append(tmp) # word 
                    elif self.rescoring_model_type == 'fairseq_bert':
                        tmp = " ".join(sent)
                        if len(tmp)>1:
                            batch.append(tmp[0].upper()+tmp[1:].lower()+'.')
                        else:
                            batch.append(' ')

                        # tmp = " ".join(sent).lower().split()
                        # tmp_batch.append(tmp) # word 

                if self.rescoring_model_type == 'fairseq_tfm_decoder':
                    max_len = len(sorted(batch, key=lambda x: len(x))[-1])
                    ppls, loss_batch, nwords_batch = predict_batch_for_rescoring(batch, self.rescoring_model, self.rescoring_dict, max_len)
                elif self.rescoring_model_type == 'fairseq_bert':
                    ppls, loss_batch, nwords_batch = predict_batch_for_rescoring_roberta_pll(batch, self.rescoring_model, self.rescoring_dict)

                    # max_len = len(sorted(batch, key=lambda x: len(x))[-1])
                    # tmp_ppls, loss_batch, nwords_batch = predict_batch_for_rescoring(tmp_batch, self.tfm_rescoring_model, self.tfm_rescoring_dict, max_len)

                    '''
                    (Pdb) ppls
                    [-323.59491527080536, -323.9512754678726, -340.0725562572479, -324.9497228860855, -335.42124104499817, 
                    -325.53756296634674, -340.4702961444855, -324.4972655773163, -325.3306745290756, -328.8111617565155]

                    (Pdb) tmp_ppls
                    [-450.0, -458.0, -454.2, -451.5, -455.8, 
                    -452.2, -462.2, -450.5, -459.8, -455.5]
                    '''

                # import pdb; pdb.set_trace()

                for j, n_th_hypo in enumerate(nbest_hypos):
                    ppl = ppls[j]
                    hypos[i][j]['rescoring_lm_ppl'] = ppl
                    hypos[i][j]['total_score'] = (
                        n_th_hypo['am_score']
                        + self.rescoring_weight * ppl 
                        + self.rescoring_word_len_weight * n_th_hypo['wl_len'] 
                        )

                # import pdb; pdb.set_trace()

                # Original Rescoring log P_{AM} (y|x) + \alpha1 log P_{LM1}(y) + \beta |y| + \alpha2 log P_{LM2}(y)
                # hypos[i] = sorted(nbest_hypos, key=lambda x: -x["rescoring_lm_ppl"])
                hypos[i] = sorted(nbest_hypos, key=lambda x: -x["total_score"])

            num_generated_tokens = sum(len(h[0]["tokens"]) for h in hypos)
            self.gen_timer_for_rescoring.stop(num_generated_tokens)

            '''
            (Pdb) hypos[0][0].keys()
            dict_keys(['tokens', 'score', 'am_score', 'lm_score', 'timesteps', 'words', 'wl_len', 'rescoring_lm_ppl', 'total_score'])
            '''

        for batch_id, sample_id in enumerate(sample["id"].tolist()):
            errs, length, best_errs, best_length = self.process_sentence(
                sample=sample,
                sid=sample_id,
                batch_id=batch_id,
                # hypo=hypos[batch_id][0],
                hypos=hypos[batch_id],
            )
            self.total_errors += errs
            self.total_length += length

            self.best_total_errors += best_errs
            self.best_total_length += best_length
            
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
            (self.gen_timer.sum + self.gen_timer_for_rescoring.sum) if self.rescoring else self.gen_timer.sum,
            self.num_sentences / ((self.gen_timer.sum + self.gen_timer_for_rescoring.sum) + 1e-6) if self.rescoring else self.num_sentences / (self.gen_timer.sum + 1e-6),
            1.0 / ((self.gen_timer.avg + self.gen_timer_for_rescoring.avg)/2 + 1e-6) if self.rescoring else 1.0 / (self.gen_timer.avg + 1e-6),
        )
        logger.info(
            "%.1fs for Beam Search",
            self.gen_timer.sum
        )
        if self.rescoring:
            logger.info(
                "%.1fs for Rescoring",
                self.gen_timer_for_rescoring.sum
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
        best_errs_t, best_leng_t = processor.best_total_errors, processor.best_total_length

        if cfg.common.cpu:
            logger.warning("Merging WER requires CUDA.")
        elif processor.data_parallel_world_size > 1:
            stats = torch.LongTensor([errs_t, leng_t]).cuda()
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            errs_t, leng_t = stats[0].item(), stats[1].item()

            stats = torch.LongTensor([best_errs_t, best_leng_t]).cuda()
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            best_errs_t, best_leng_t = stats[0].item(), stats[1].item()

        wer = errs_t * 100.0 / leng_t
        best_wer = best_errs_t * 100.0 / best_leng_t

        logger.info("Word error rate: %.4f", wer)
        logger.info("Best Possible (Oracle) Word error rate: %.4f", best_wer)

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
    # import pdb; pdb.set_trace()
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

    # import pdb; pdb.set_trace()

    hydra_main()  # pylint: disable=no-value-for-parameter


if __name__ == "__main__":
    cli_main()
