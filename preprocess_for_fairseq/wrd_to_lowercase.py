
def process(fname, output_fname):
    with open(fname, 'rt') as f:
        lines = f.readlines()

    for i in range(len(lines)):
        lines[i] = (lines[i][0].upper() + lines[i][1:].lower() + '.').replace('\n','') + ('\n')

    # import pdb; pdb.set_trace()
        
    with open(output_fname, 'wt') as f:
        f.writelines(lines)

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Librispeech Dataset creation.")
    parser.add_argument(
        "--input_data", help="", default="/workspace/data2vec/librispeech_audio_for_data2vec/train.wrd"
    )
    parser.add_argument(
        "--output_data", help="", default="/workspace/data2vec/librispeech_audio_for_data2vec/train.lower_wrd"
    )
    args = parser.parse_args()
    process(args.input_data, args.output_data)
