
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
import re

# # arpa_file='/workspace/librispeech_model/librispeech-lm-train-norm-word-4gram.arpa'
# # lex_file='/workspace/librispeech_model/tmp.lexicon'

# arpa_file='/workspace/librispeech_model/4-gram.arpa'
# lex_file='/workspace/librispeech_model/tmp.lexicon'

# print("Writing Lexicon file - {}...".format(lex_file))
# with open(lex_file, "w") as f:
#     with open(arpa_file, "r") as arpa:
#         for i,line in enumerate(arpa):
#             # verify if the line corresponds to unigram
#             if not re.match(r"[-]*[0-9\.]+\t\S+\t*[-]*[0-9\.]*$", line):
#                 continue
#             # print(line)
#             word = line.split("\t")[1]
#             word = word.strip()
#             # print(word)
#             # if word == "<unk>" or word == "<s>" or word == "</s>":
#             #     continue
#             if word == "<UNK>" or word == "<s>" or word == "</s>":
#                 continue
#             assert re.match("^[A-Z']+$", word), "invalid word - {w}".format(w=word)
#             f.write("{w}\t{s} |\n".format(w=word, s=" ".join(word)))
#             print("{w}\t{s} |\n".format(w=word, s=" ".join(word)))
# print("Done!", flush=True)


dict_file='/workspace/librispeech_model/decoder/lm_librispeech_word_transformer/lm_librispeech_word_transformer.dict'
lex_file='/workspace/librispeech_model/tmp.lexicon'

with open(lex_file, "w") as f:
    with open(dict_file, "r") as arpa:
        for i,line in enumerate(arpa):
            # print('line',line)
            word = line.split(" ")[0]
            word = word.strip()
            # print('word',word)
            # import pdb; pdb.set_trace()
            if word == "<unk>" or word == "<s>" or word == "</s>":
                continue
            if word.startswith('madeupword'):
                continue
            assert re.match("^[a-z']+$", word), "invalid word - {w}".format(w=word)
            f.write("{w}\t{s} |\n".format(w=word, s=" ".join(word)))
            # print("{w}\t{s} |\n".format(w=word, s=" ".join(word.upper())))


