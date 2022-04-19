
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
import sys
from collections import defaultdict

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(FILE_DIR, "../utilities"))


import re
import numpy


EOS = "</s>"


def convert_words_to_letters_ctc(fin_name, fout_name, letter):
    with open(fin_name, "r") as fin, open(fout_name, "w") as fout:
        for line in fin:
            # print('line',line)
            words = line.strip().split(" ")
            # print('words',words)
            for i, word in enumerate(words):
                if letter == "upper_case":
                    word = re.sub("[^A-Z'.]+", "", word)
                else:
                    word = re.sub("[^a-z'.]+", "", word)
                # print('word',word)
                if len(word) == 0:
                    continue
                if i != len(words)-1:
                    new_word = word + "|"
                else:
                    new_word = word
                # print('new_word',new_word)
                fout.write(" ".join(list(new_word)) + " ")
            # import pdb
            # pdb.set_trace()
            fout.write("\n")



def convert_words_to_letters_asg_rep2(fin_name, fout_name, letter):
    with open(fin_name, "r") as fin, open(fout_name, "w") as fout:
        for line in fin:
            words = line.strip().split(" ")
            for word in words:
                if letter == "upper_case":
                    word = re.sub("[^A-Z'.]+", "", word)
                else:
                    word = re.sub("[^a-z'.]+", "", word)
                if len(word) == 0:
                    continue
                new_word = transform_asg(word) + "|"
                fout.write(" ".join(list(new_word)) + " ")
            fout.write("\n")


def transform_asg(word):
    if word == "":
        return ""
    new_word = word[0]
    prev = word[0]
    repetition = 0
    for letter in word[1:]:
        if letter == prev:
            repetition += 1
        else:
            if repetition != 0:
                new_word += "1" if repetition == 1 else "2"
                repetition = 0
            new_word += letter
        prev = letter
    if repetition != 0:
        new_word += "1" if repetition == 1 else "2"
    return new_word




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Librispeech Dataset creation.")
    parser.add_argument(
        "--data_dst", help="data destination directory", default="./librispeech"
    )

    parser.add_argument(
        "--model_dst",
        help="model auxilary files destination directory",
        default="./conv_glu_librispeech_char",
    )

    parser.add_argument(
        "-p", "--process", help="# of process for Multiprocessing", default=8, type=int
    )
    parser.add_argument(
        "--letter", help="", default="upper_case", type=str, choices=["upper_case","lower_case"]
    )

    args = parser.parse_args()

    decoder_path = os.path.join(args.model_dst, "decoder")
    os.makedirs(decoder_path, exist_ok=True)


    # Prepare data for char lm training/evaluation
    if os.path.exists(os.path.join(decoder_path, "upper_char_lm_data2.train")):
        print(
            "Skip generation of {}. Please remove the file to regenerate it".format(
                os.path.join(decoder_path, "upper_char_lm_data2.train")
            )
        )
    else:
        convert_words_to_letters_ctc(
            os.path.join(args.data_dst, "text/librispeech-lm-norm.txt.upper.shuffle"),
            os.path.join(decoder_path, "upper_char_lm_data2.train"),
            args.letter
        )

    convert_words_to_letters_ctc(
        os.path.join(args.data_dst, "text/upper-dev-clean.txt"),
        os.path.join(decoder_path, "upper_char_lm_data2.dev-clean"),
        args.letter
    )
    convert_words_to_letters_ctc(
        os.path.join(args.data_dst, "text/upper-dev-other.txt"),
        os.path.join(decoder_path, "upper_char_lm_data2.dev-other"),
        args.letter
    )

    print("Done!", flush=True)