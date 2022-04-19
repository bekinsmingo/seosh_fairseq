
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


def convert_words_to_letters_ctc(input_dict, reference_dict, output_dict):
    tmp=dict()
    with open(input_dict, "r") as inp:
        for line in inp:
            tmp_=line.strip().split(" ")
            tmp[tmp_[0]]=tmp_[1]

    # import pdb
    # pdb.set_trace()

    with open(input_dict, "r") as inp, open(reference_dict, "r") as ref, open(output_dict, "w") as out:
        for line in ref:
            token = line.strip().split("\t")[0]
            if token in tmp.keys():
                out.write("{t}\t{n}\n".format(t=token, n=tmp[token]))
            else:
                out.write("{t}\t{n}\n".format(t=token, n=0))
            # import pdb
            # pdb.set_trace()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Librispeech Dataset creation.")
    parser.add_argument(
        "--dst", help="", default="./librispeech"
    )
    parser.add_argument(
        "--input_dict", help="", default="./librispeech"
    )
    parser.add_argument(
        "--reference_dict", help="", default="./librispeech"
    )
    parser.add_argument(
        "--output_dict", help="", default="./librispeech"
    )

    args = parser.parse_args()

    # import pdb
    # pdb.set_trace()

    convert_words_to_letters_ctc(
        os.path.join(args.dst, args.input_dict),
        os.path.join(args.dst, args.reference_dict),
        os.path.join(args.dst, args.output_dict),
    )

    print("Done!", flush=True)