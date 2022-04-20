
FAIRSEQ_ROOT=/workspace/seosh_fairseq &&\
DST_DIR=/workspace/librispeech_model/am/fairseq_audio_data2 &&\
INPUT_DIR=/workspace/librispeech_model/am/fairseq_audio_data2  &&\


python3 $FAIRSEQ_ROOT/preprocess_for_fairseq/spm_lm.py \
--dataset $INPUT_DIR/train.wrd \
--model_dst $DST_DIR \
--letter upper_case


VOCAB=/workspace/librispeech_model/am/fairseq_audio_data2/train_10000_wp.model &&\

for SPLIT in train dev_other dev_clean test_other test_clean; do \
    python3 $FAIRSEQ_ROOT/preprocess_for_fairseq/spm_lm_encode.py \
        --model $VOCAB \
        --model_dst $DST_DIR \
        --inputs $INPUT_DIR/${SPLIT}.wrd
done

mv wordpiece_train.wrd train.wp 
mv wordpiece_dev_clean.wrd dev_clean.wp
mv wordpiece_dev_other.wrd dev_other.wp
mv wordpiece_test_clean.wrd test_clean.wp
mv wordpiece_test_other.wrd test_other.wp


DST_DIR=/workspace/librispeech_model/am/fairseq_audio_data2 &&\
FAIRSEQ=/workspace/seosh_fairseq/fairseq_cli/ &&\

mkdir -p "$DST_DIR/tmp" &&\

fairseq-preprocess \
    --task language_modeling \
    --only-source \
    --srcdict "/workspace/librispeech_model/am/fairseq_audio_data2/train_10000_wp.vocab" \
    --trainpref "$DST_DIR/train.wp" \
    --validpref "$DST_DIR/dev_other.wp" \
    --testpref "$DST_DIR/test_other.wp" \
    --destdir "$DST_DIR/tmp" \
    --workers 16

mv $DST_DIR/tmp/dict.txt $DST_DIR/
mv $DST_DIR/dict.txt $DST_DIR/dict.wp.txt
rm -r $DST_DIR/tmp
