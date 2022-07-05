
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
import sys
from collections import defaultdict

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(FILE_DIR, "../utilities"))

import re
import numpy

from pdb import set_trace as Tra

def replace_space_symbol(fin_name, fout_name):
    with open(fin_name, "r") as fin, open(fout_name, "w") as fout:
        for line in fin:
            fout.write(line.replace('|','▁'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="replace space symbol")
    parser.add_argument(
        "--input", 
        help="input path", 
        default="/workspace/librispeech_data/text/upper_char_lm_data.train"
    )

    parser.add_argument(
        "--target",
        help="target path",
        default="/workspace/librispeech_data/text/upper_char_lm_data_for_nest.train",
    )

    args = parser.parse_args()

    dir, file = os.path.split(args.target)
    os.makedirs(dir, exist_ok=True)

    # Prepare data for char lm training/evaluation
    if os.path.exists(args.target):
        print(
            "Skip generation of {}. Please remove the file to regenerate it".format(
                args.target
            )
        )
    else:
        replace_space_symbol(
            args.input,
            args.target,
        )

    print("Done!", flush=True)