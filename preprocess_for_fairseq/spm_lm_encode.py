from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import contextlib
import sys
import os

import sentencepiece as spm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, help="sentencepiece model to use for encoding"
    )
    parser.add_argument(
        "--model_dst",
        help="model auxilary files destination directory",
        default="./model",
    )
    parser.add_argument(
        "--inputs", nargs="+", default=["-"], help="input files to filter/encode"
    )
    # parser.add_argument(
    #     "--outputs", nargs="+", default=["-"], help="path to save encoded outputs"
    # )
    parser.add_argument("--output_format", choices=["piece", "id"], default="piece")
    parser.add_argument(
        "--min-len",
        type=int,
        metavar="N",
        help="filter sentence pairs with fewer than N tokens",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        metavar="N",
        help="filter sentence pairs with more than N tokens",
    )
    args = parser.parse_args()

    # assert len(args.inputs) == len(
    #     args.outputs
    # ), "number of input and output paths should match"

    sp = spm.SentencePieceProcessor()
    sp.Load(args.model)

    if args.output_format == "piece":

        def encode(input):
            return sp.EncodeAsPieces(input)

    elif args.output_format == "id":

        def encode(input):
            return list(map(str, sp.EncodeAsIds(input)))

    else:
        raise NotImplementedError

    if args.min_len is not None or args.max_len is not None:

        def valid(line):
            return (args.min_len is None or len(line) >= args.min_len) and (
                args.max_len is None or len(line) <= args.max_len
            )

    else:

        def valid(lines):
            return True

    with contextlib.ExitStack() as stack:
        inputs = [
            stack.enter_context(open(input, "r", encoding="utf-8"))
            if input != "-"
            else sys.stdin
            for input in args.inputs
        ]
        # outputs = [
        #     stack.enter_context(open(output, "w", encoding="utf-8"))
        #     if output != "-"
        #     else sys.stdout
        #     for output in args.outputs
        # ]

        outputs = [
            #stack.enter_context(open(os.path.join(args.model_dst,"decoder","wordpiece_"+input.split("/")[-1]), "w", encoding="utf-8"))
            stack.enter_context(open(os.path.join(args.model_dst,"wordpiece_"+input.split("/")[-1]), "w", encoding="utf-8"))

            if input != "-"
            else sys.stdin
            for input in args.inputs
        ]

        # import pdb
        # pdb.set_trace()

        stats = {
            "num_empty": 0,
            "num_filtered": 0,
        }

        def encode_line(line):
            line = line.strip()
            if len(line) > 0:
                line = encode(line)
                if valid(line):
                    return line
                else:
                    stats["num_filtered"] += 1
            else:
                stats["num_empty"] += 1
            return None

        for i, lines in enumerate(zip(*inputs), start=1):
            enc_lines = list(map(encode_line, lines))
            if not any(enc_line is None for enc_line in enc_lines):
                for enc_line, output_h in zip(enc_lines, outputs):
                    print(" ".join(enc_line), file=output_h)
            if i % 10000 == 0:
                print("processed {} lines".format(i), file=sys.stderr)

        print("skipped {} empty lines".format(stats["num_empty"]), file=sys.stderr)
        print("filtered {} lines".format(stats["num_filtered"]), file=sys.stderr)


if __name__ == "__main__":
    main()
