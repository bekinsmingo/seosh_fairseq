
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
import sys
from collections import defaultdict
import re
import numpy

def split_train_and_valid(p, file_name, train_file_name, valid_file_name):
    file = open(file_name, "r")
    lines = file.readlines()

    total_len = len(lines)
    num_train_lines = int(total_len*(1-p))
    cnt=0

    print("The number of total lines is {}. so train : {}, valid : {}".format(total_len,num_train_lines,total_len-num_train_lines))

    with open(train_file_name, "w") as ftrain, open(valid_file_name, "w") as fvalid:
        for line in lines:
            cnt+=1
            if cnt < num_train_lines:
                ftrain.write(line)
                # ftrain.write("\n")
            else:
                fvalid.write(line)
                # fvalid.write("\n")

    file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Librispeech Dataset creation.")
    parser.add_argument(
        "--file_path", help="data destination directory", default="tmp.txt"
    )

    parser.add_argument(
        "--model_dst", help="data destination directory", default="tmp_out.txt"
    )

    parser.add_argument(
        "-p", "--percentage", help="", default=0.05, type=float
    )

    args = parser.parse_args()
    file_name = args.file_path.split("/")[-1]
    train_file_name = os.path.join(args.model_dst, str(file_name) + ".train")
    valid_file_name = os.path.join(args.model_dst, str(file_name) + ".valid")

    print('file_path',args.file_path)
    print('train_file_name',train_file_name)
    print('valid_file_name',valid_file_name)

    # Prepare data for char lm training/evaluation
    if os.path.exists(train_file_name):
        print(
            "Skip generation of {}. Please remove the file to regenerate it".format(
                train_file_name
            )
        )
    else:
        split_train_and_valid(
            args.percentage,
            args.file_path,
            train_file_name,
            valid_file_name,
        )

    print("Done!", flush=True)
