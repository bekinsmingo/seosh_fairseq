

FAIRROOT="/workspace/fairseq"
encoder=$FAIRROOT/scripts/spm_encode.py

DATAROOT="."
DATASET_NAME="raw/kr_corpus"
JOBNAME="kr_corpus_bpe32k"

OUTPUT_PATH=$DATAROOT/output/$JOBNAME
mkdir -p $OUTPUT_PATH

spm_model="/mnt/clova_speech/users/mspark/models/vocabs/spm/kr_bpe_32k.model"
dict_file="/mnt/clova_speech/users/mspark/models/vocabs/spm/kr_bpe_32k.vocab"

fairvocab_output="vocabs/libri.16k.fairvocab"
python reorganize_dict.py $dict_file $fairvocab_output

encode() {
    target=$1
    text=$( find $DATAROOT/$DATASET_NAME/$target -name "*.txt" )
    output_file=$OUTPUT_PATH/$target.tokens
    echo $text | tr " " "\n"

    awk 1 $text \
        | python $encoder \
        --model=$spm_model \
        --output_format=piece \
        > $output_file
}

encode train
encode valid

fairseq-preprocess \
    --task language_modeling \
    --only-source \
    --srcdict $fairvocab_output \
    --trainpref $OUTPUT_PATH/train.tokens \
    --validpref $OUTPUT_PATH/valid.tokens \
    --destdir $OUTPUT_PATH \
    --workers 16
