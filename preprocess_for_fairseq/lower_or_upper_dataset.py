
def lower(fname, output_fname, letter):
    with open(fname, 'rt') as f:
        lines = f.readlines()

    for i in range(len(lines)):
        if letter == "upper_case":
            lines[i] = lines[i].upper()
        else:
            lines[i] = lines[i].lower()
        
    with open(output_fname, 'wt') as f:
        f.writelines(lines)

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Librispeech Dataset creation.")
    parser.add_argument(
        "--input_data", help="", default="/workspace/librispeech_data/text/librispeech-lm-norm.txt.lower.shuffle"
    )
    parser.add_argument(
        "--output_data", help="", default="/workspace/librispeech_data/text/librispeech-lm-norm.txt.shuffle"
    )
    parser.add_argument(
        "--letter", help="", default="upper_case", type=str, choices=["upper_case","lower_case"]
    )
    args = parser.parse_args()

    lower(args.input_data, args.output_data, args.letter)
