
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
import re

# ################################### ngram ####################################

# # arpa_file='/workspace/librispeech_model/librispeech-lm-train-norm-word-4gram.arpa'
# # lex_file='/workspace/librispeech_model/tmp.lexicon'

# arpa_file='/workspace/librispeech_model/4-gram.arpa'
# lex_file='/workspace/fairseq/lexicon/tmp.lexicon'

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

#################################### TFM ####################################

# dict_file='/workspace/librispeech_model/decoder/lm_librispeech_word_transformer/lm_librispeech_word_transformer.dict'
# lex_file='/workspace/fairseq/lexicon/tmp.lexicon'

# with open(lex_file, "w") as f:
#     with open(dict_file, "r") as arpa:
#         for i,line in enumerate(arpa):
#             # print('line',line)
#             word = line.split(" ")[0]
#             word = word.strip()
#             # print('word',word)
#             # import pdb; pdb.set_trace()
#             if word == "<unk>" or word == "<s>" or word == "</s>":
#                 continue
#             if word.startswith('madeupword'):
#                 continue
#             assert re.match("^[a-z']+$", word), "invalid word - {w}".format(w=word)
#             f.write("{w}\t{s} |\n".format(w=word, s=" ".join(word)))
#             # print("{w}\t{s} |\n".format(w=word, s=" ".join(word.upper())))


# ################################### 700k word GBW dict ####################################

# import torch
# from fairseq.data.dictionary import Dictionary
# from fairseq.dataclass.utils import convert_namespace_to_omegaconf
# from fairseq import utils
# from fairseq.data import encoders

# # lm_dict = Dictionary.load('/mnt/clova_speech/users/seosh/adaptive_lm_gbw_huge/dict.txt')
# # import pdb; pdb.set_trace()

# dict_file='/workspace/fairseq/lexicon/gbw_800k_word_dict.txt'
# lex_file='/workspace/fairseq/lexicon/tmp.lexicon'

# with open(lex_file, "w") as f:
#     with open(dict_file, "r") as arpa:
#         for i,line in enumerate(arpa):
#             # print('line',line)
#             word = line.split(" ")[0]
#             word = word.strip()
#             # print('word',word)
#             # import pdb; pdb.set_trace()
#             if word == "<unk>" or word == "<s>" or word == "</s>":
#                 continue
#             if word.startswith('madeupword'):
#                 continue
#             # print(line)
#             # assert re.match("^[a-z']+$", word), "invalid word - {w}".format(w=word)
#             if re.match("^[a-zA-Z']+$", word):
#                 f.write("{w}\t{s} |\n".format(w=word, s=" ".join(word.lower())))
#             else:
#                 print(line)
#             # print("{w}\t{s} |\n".format(w=word, s=" ".join(word.upper())))




# ################################### GPT tokenizer ####################################

# import torch
# from fairseq.data.dictionary import Dictionary
# from fairseq.dataclass.utils import convert_namespace_to_omegaconf
# from fairseq import utils
# from fairseq.data import encoders

# from fairseq.models.transformer_lm import TransformerLanguageModel
# model_dir = '/mnt/clova_speech/users/seosh/en_dense_lm_355m'
# lm = TransformerLanguageModel.from_pretrained(model_dir, bpe='gpt2')


# lm_dict = Dictionary.load('/mnt/clova_speech/users/seosh/en_dense_lm_355m/dict.txt')
# bpe_tokenizer = encoders.build_bpe('gpt2')

# sentence = "This is the first paragraph of the first document."
# tmp = bpe_tokenizer.encode(sentence)
# encoded_line = lm_dict.encode_line(tmp)

# decoded_line = lm_dict.string(encoded_line)
# recon_sentence = bpe_tokenizer.decode(decoded_line)

# dict_file='/workspace/librispeech_model/decoder/lm_librispeech_word_transformer/lm_librispeech_word_transformer.dict'
# lex_file='/workspace/fairseq/lexicon/tmp.lexicon'

# with open(lex_file, "w") as f:
#     with open(dict_file, "r") as arpa:
#         for i,line in enumerate(arpa):
#             # print('line',line)
#             word = line.split(" ")[0]
#             word = word.strip()
#             # print('word',word)
#             # import pdb; pdb.set_trace()
#             if word == "<unk>" or word == "<s>" or word == "</s>":
#                 continue
#             if word.startswith('madeupword'):
#                 continue
#             assert re.match("^[a-z']+$", word), "invalid word - {w}".format(w=word)
#             f.write("{w}\t{s} |\n".format(w=word, s=" ".join(word)))
#             # print("{w}\t{s} |\n".format(w=word, s=" ".join(word.upper())))





# ###### 

# file='/workspace/fairseq/lexicon/w2v2_4gram_lexicon.lexicon'
# lex_file='/workspace/fairseq/lexicon/tmp.lexicon'

# print("Writing Lexicon file - {}...".format(lex_file))
# with open(lex_file, "w") as fout:
#     with open(file, "r") as fin:
#         for i,line in enumerate(fin):
#             word = line.split("\t")[0]
#             word = word.strip()
#             # import pdb; pdb.set_trace()
#             fout.write("{w}\t{s} |\n".format(w=word, s=" ".join(word.lower())))
#             # print("{w}\t{s} |\n".format(w=word, s=" ".join(word)))
# print("Done!", flush=True)


#######

import kenlm
kenlm_model = kenlm.Model('/mnt/clova_speech/e2e_dataset/en/model/KenLM/en_word.binary')
kenlm_model2 = kenlm.LanguageModel('/mnt/clova_speech/e2e_dataset/en/model/KenLM/en_word.binary')

import pdb; pdb.set_trace()