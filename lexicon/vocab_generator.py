
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
import re


arpa_file='/home1/irteam/users/seosh/decoder_pratice/librispeech_model/decoder/lm_librispeech_word_transformer/lm_librispeech_word_transformer.dict'
vocab_file='/home1/irteam/users/seosh/decoder_pratice/librispeech_model/decoder/lm_librispeech_word_transformer/lm_librispeech_upper_word_transformer.dict'
with open(vocab_file, "w") as f:
    with open(arpa_file, "r") as arpa:
        for i,line in enumerate(arpa):
            # print('line',line)
            tmp = line.split(" ")
            freq = tmp[1]
            word = tmp[0].strip()
            # print('word',word)
            # print('freq',freq)

            if word == "<unk>" or word == "<s>" or word == "</s>":
                continue
            if 'madeupword' in word :
                f.write("{w} {s}".format(w=word,s=freq))
            else:
                assert re.match("^[a-z']+$", word), "invalid word - {w}".format(w=word)
                f.write("{w} {s}".format(w=word.upper(),s=freq))