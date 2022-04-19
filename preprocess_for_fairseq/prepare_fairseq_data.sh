#!/bin/bash
DATA_DST="$1"
MODEL_DST="$2"
FAIRSEQ="$3"

mkdir -p "$MODEL_DST/decoder/fairseq_word_data/"
mkdir -p "$MODEL_DST/decoder/fairseq_char_data/"
mkdir -p "$MODEL_DST/decoder/fairseq_char_data2/"
mkdir -p "$MODEL_DST/decoder/fairseq_char_data3/"
mkdir -p "$MODEL_DST/decoder/fairseq_wp_data/"
mkdir -p "$MODEL_DST/decoder/fairseq_wp_data2/"

# python3 "$FAIRSEQ/preprocess.py" --only-source \
# --trainpref "$DATA_DST/text/librispeech-lm-norm.txt.upper.shuffle" \
# --validpref "$DATA_DST/text/upper-dev-clean.txt" \
# --testpref "$DATA_DST/text/upper-dev-other.txt" \
# --destdir "$MODEL_DST/decoder/fairseq_word_data/" \
# --thresholdsrc 10 \
# --padding-factor 1 \
# --workers 16

# cut -f1 -d " " "$MODEL_DST/decoder/fairseq_word_data/dict.txt" | tr "\n" " " > "$MODEL_DST/decoder/kenlm_limit_vocab_file.txt"

# python3 "$FAIRSEQ/preprocess.py" --only-source \
# --trainpref "$MODEL_DST/decoder/upper_char_lm_data.train" \
# --validpref "$MODEL_DST/decoder/upper_char_lm_data.dev-clean" \
# --testpref "$MODEL_DST/decoder/upper_char_lm_data.dev-other" \
# --destdir "$MODEL_DST/decoder/fairseq_char_data/" \
# --thresholdsrc 10 \
# --workers 16

python3 "$FAIRSEQ/preprocess.py" --only-source \
--trainpref "$MODEL_DST/decoder/upper_char_lm_data2.train" \
--validpref "$MODEL_DST/decoder/upper_char_lm_data2.dev-clean" \
--testpref "$MODEL_DST/decoder/upper_char_lm_data2.dev-other" \
--destdir "$MODEL_DST/decoder/fairseq_char_data2/" \
--thresholdsrc 10 \
--workers 16


# python3 "$FAIRSEQ/preprocess.py" --only-source \
# --task language_modeling \
# --srcdict "/workspace/libri_upper_char.vocab" \
# --trainpref "$MODEL_DST/decoder/upper_char_lm_data.train" \
# --validpref "$MODEL_DST/decoder/upper_char_lm_data.dev-clean" \
# --testpref "$MODEL_DST/decoder/upper_char_lm_data.dev-other" \
# --destdir "$MODEL_DST/decoder/fairseq_char_data3/" \
# --thresholdsrc 10 \
# --workers 16

# fairseq-preprocess \
#     --task language_modeling \
#     --only-source \
#     --srcdict "/workspace/libri_upper_char.vocab" \
#     --trainpref "$MODEL_DST/decoder/upper_char_lm_data.train" \
#     --validpref "$MODEL_DST/decoder/upper_char_lm_data.dev-clean" \
#     --testpref "$MODEL_DST/decoder/upper_char_lm_data.dev-other" \
#     --destdir "$MODEL_DST/decoder/fairseq_char_data3/" \
#     --workers 16








# python3 "$FAIRSEQ/preprocess.py" --only-source \
# --trainpref "$MODEL_DST/decoder/wordpiece_librispeech-lm-norm.txt.upper.shuffle" \
# --validpref "$MODEL_DST/decoder/wordpiece_upper-dev-clean.txt" \
# --testpref "$MODEL_DST/decoder/wordpiece_upper-dev-other.txt" \
# --destdir "$MODEL_DST/decoder/fairseq_wp_data/" \
# --workers 16

# python3 "$FAIRSEQ/preprocess.py" --only-source \
# --trainpref "$MODEL_DST/decoder/tmp_wordpiece_librispeech-lm-norm.txt.upper.shuffle" \
# --validpref "$MODEL_DST/decoder/tmp_wordpiece_upper-dev-clean.txt" \
# --testpref "$MODEL_DST/decoder/tmp_wordpiece_upper-dev-other.txt" \
# --destdir "$MODEL_DST/decoder/fairseq_wp_data2/" \
# --workers 16



# fairvocab_output="/workspace/lexicon_free_librispeech/decoder/upper_wordpiece_10000.librispeech-lm-norm.txt.upper.shuffle.vocab"
# python reorganize_dict.py $dict_file $fairvocab_output

# fairseq-preprocess \
#     --task language_modeling \
#     --only-source \
#     --srcdict $fairvocab_output \
#     --trainpref "$MODEL_DST/decoder/wordpiece_librispeech-lm-norm.txt.upper.shuffle" \
#     --validpref "$MODEL_DST/decoder/wordpiece_upper-dev-clean.txt" \
#     --testpref "$MODEL_DST/decoder/wordpiece_upper-dev-other.txt" \
#     --destdir "$MODEL_DST/decoder/fairseq_wp_data2/" \
#     --workers 16