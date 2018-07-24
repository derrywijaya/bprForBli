""" 
Convert all matrices to Magnitude format
"""

import numpy as np
import pandas as pd
import os
import pathlib
from argparse import ArgumentParser 

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('lang_dir', type=str, help='path to a language\'s directory')
    parser.add_argument('vec_dir', type=str, help='path to the directory of Magnitude files')
    args = parser.parse_args()
    return args 

def process_bias_vec(args):
    """
    Convert bias vector to a numpy array 
    """
    bias = pd.read_csv(args.lang_dir+'BVector.txt', sep=',', header=None).values
    # remove '[' and ']'
    bias[0][0] = float(bias[0][0][1:])
    bias[0][99] = float(bias[0][99].split(']')[0])
    return bias 

def process_matrix(args):
    """
    Convert current format to a format that Pandas can process 
    @return matrices: names of files in the matrices folder
    """
    matrix_path = args.lang_dir + '/matrices/'
    # dump all processed matrices into one directory
    processed_path = args.lang_dir +'/processed/'
    pathlib.Path(processed_path).mkdir(exist_ok=True) 

    matrices = os.listdir(matrix_path)
    for m in matrices:
        # check if matrix has already been processed 
        processed_file = processed_path+m
        if not os.path.exists(processed_file):
            matrix = pd.read_csv(matrix_path+m, sep='\t', header=None)
            # remove indices
            matrix = matrix[1]
            for i in range(matrix.shape[0]):
                line = matrix[i][1:len(matrix[i])-1]
                with open(processed_path+m, 'a') as f:
                    f.write(line+'\n')
    return matrices

def convert_matrix(matrix, words, filename, args):
    """
    Convert matrices to a format that can be converted to Magnitude
    @param matrix: matrix to be converted
    @param words: col 0: words, col 1: indices that indicate words' positions
    If matrix is thirdTQ, col 2: count 
    """
    path = args.lang_dir + '/processed/'
    for i in range(words.shape[0]):
        idx = words[i][1]
        if idx >= 0:                  
            # check count if matrix is thirdTQ
            if words.shape[1] == 2 or (words.shape[1] == 3 and words[i][2] > 1):
                row = matrix[idx].astype('<U100')
                # if the word exists in the matrix, insert the word in idx 0 of its embedding row
                row = np.insert(row, 0, words[i][0])
                with open(path+filename, 'a') as f:
                    np.savetxt(f, row.reshape(1, row.shape[0]), fmt='%s')

def main():
    args = parse_args()
    print("### CONVERTING MATRICES ###")
    files = process_matrix(args)
    print('All matrices are processed and ready for conversion\n')

    eng_words = pd.read_csv(args.lang_dir+'englishIds.txt', sep='\t', header=None).values
    foreign_words = pd.read_csv(args.lang_dir+'foreignIds.txt', sep='\t', header=None).values
    
    # indices to access embeddings: wiki: col 2, third: col 3, extended: col 1
    matrices = []
    processed_path = args.lang_dir + '/processed/'
    if 'wikiPMatrix.txt' in files:
        # wiki
        wikiP = pd.read_csv(processed_path+'wikiPMatrix.txt', sep=',', header=None).values
        wikiQ = pd.read_csv(processed_path+'wikiQMatrix.txt', sep=',', header=None).values
        matrices.append((wikiP, wikiQ, 2, 'wiki'))

    if 'thirdTPMatrix.txt' in files:
        # third
        thirdP = pd.read_csv(processed_path+'thirdTPMatrix.txt', sep=',', header=None).values
        thirdQ = pd.read_csv(processed_path+'thirdTQMatrix.txt', sep=',', header=None).values
        matrices.append((thirdP, thirdQ, 3, 'thirdT'))
    
    if 'extendedMatrix.txt' in files:
        # extended
        exP = pd.read_csv(processed_path+'EMatrix.txt', sep=',', header=None).values
        exQ = pd.read_csv(processed_path+'extendedMatrix.txt', sep=',', header=None).values
        matrices.append((exP, exQ, 1, 'E'))
   
    # clean up files
    for f in files:
        os.remove(processed_path + f)

    for eng, foreign, idx, fname in matrices:
        if idx == 3:
            count_idx = 4
            convert_matrix(foreign, foreign_words[:,[0,idx,count_idx]], fname+'QMatrix.txt', args)
        else:
            convert_matrix(foreign, foreign_words[:,[0,idx]], fname+'QMatrix.txt', args)
        convert_matrix(eng, eng_words[:,[0,idx]], fname+'PMatrix.txt', args) 
    print('### MATRIX CONVERSION IS COMPLETE ###')
    print('All converted matrices are stored in {}processed/'.format(args.lang_dir))
    print('Please run the following command to convert a matrix to Magnitude:')
    print('python -m pymagnitude.converter -i <input path> -o <output path> -a')

if __name__=='__main__':
    main()
    
