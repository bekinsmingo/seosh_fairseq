"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the MIT-style license found in the
LICENSE file in the root directory of this source tree.
----------
Script to prepare recipe to train/eval model on Librispeech in wav2letter++ pipelines
Command : python3 prepare.py --data_dst [...] --model_dst [...] --wp 10000 --nbest 10
Replace [...] with appropriate paths
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
import re
from collections import defaultdict
import contextlib
import sys

import sentencepiece as spm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Librispeech Dataset creation.")
    # parser.add_argument(
    #     "--data_dst", help="data destination directory", default="./librispeech"
    # )
    parser.add_argument(
        "--dataset", help="", default=""
    )
    parser.add_argument(
        "--model_dst",
        help="model auxilary files destination directory",
        default="./model",
    )
    parser.add_argument(
        "-p", "--process", help="# of process for Multiprocessing", default=8, type=int
    )
    parser.add_argument("--wp", help="number of word pieces", default=10000, type=int)
    parser.add_argument(
        "--nbest",
        help="number of best segmentations for each word (or numbers comma separated)",
        default="10",
    )
    parser.add_argument(
        "--letter", help="", default="upper_case", type=str, choices=["upper_case","lower_case"]
    )

    args = parser.parse_args()

    print("There is no sentence piece vocab (model), lets started from the bottom ... \n", flush=True)


    decoder_path = os.path.join(args.model_dst, "decoder")
    os.makedirs(decoder_path, exist_ok=True)

    num_wordpieces = args.wp
    train_all_text = args.dataset
    dataset_name = train_all_text.split("/")[-1]
    prefix = "{}_wordpiece_{}.{}".format(args.letter.split("_")[0], num_wordpieces, dataset_name)
    prefix = os.path.join(decoder_path, prefix)
    vocab_name = prefix + ".vocab"
    model_name = prefix + ".model"

    # train
    print("Computing word pieces...\n", flush=True)
    train_cmd = (
        "--input={input} --model_prefix={prefix} --vocab_size={sz}"
        " --character_coverage=1.0 --model_type=unigram --train_extremely_large_corpus=True"
        " --split_by_unicode_script=false".format(
            input=train_all_text, prefix=prefix, sz=num_wordpieces
        )
    )
    spm.SentencePieceTrainer.Train(train_cmd)

    # word piece dictionary
    print("Creating word piece list...\n", flush=True)
    exclude_list = {"<unk>", "<s>", "</s>"}
    with open(vocab_name.replace(".vocab", ".tokens"), "w") as fvocab_filt:
        with open(vocab_name, "r", encoding="utf-8") as fvocab:
            for line in fvocab:
                val, _ = line.strip().split("\t", 1)
                if val not in exclude_list:
                    fvocab_filt.write(val.replace("\u2581", "_") + "\n")

    print("Done!", flush=True)




