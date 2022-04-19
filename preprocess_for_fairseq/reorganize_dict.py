

def reorg(fname, output_fname, start_idx=0):
    with open(fname, 'rt') as f:
        lines = f.readlines()
    
    #output_lines = ["<s> -1", "<pad> -1", "</s> -1", "<unk> -1"]
#    output_lines = ["<unk> -1 #fairseq:overwrite", 
#            "<pad> -1 #fairseq:overwrite", 
#            "<s> -1 #fairseq:overwrite", 
#            "</s> -1 #fairseq:overwrite"]
    output_lines = []
    for i, line in enumerate(lines[start_idx:]):
        token, nll = line.split()
        output_lines.append('{} {}'.format(token, i))

    with open(output_fname, 'wt') as f:
        f.writelines('\n'.join(output_lines))


if __name__ == '__main__':
#    fname = "/data/clova_speech/users/mspark/models/vocabs/spm/libri.16k.vocab"
#    output_fname = "libri.16k.fairvocab"
#    reorg(fname, output_fname, 4)
    import sys
    fname = sys.argv[1]
    output_fname = sys.argv[2]
    reorg(fname, output_fname, 4)

