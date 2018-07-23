"""
Find top 10 English translations of each foreign word
Save translation outputs to txt file 
"""

import numpy as np
import pandas as pd
from pymagnitude import *
from argparse import ArgumentParser
from convertMatrix import process_bias_vec
import os

def find_BPR_translations(args):
    path = args.lang_dir
    if os.path.exists(path+'BPR_trans.txt'):
        print('Warning: Translation outputs have already been produced.')
        exit()
    
    lang_dir = args.lang_dir.split('/')
    lang = lang_dir[len(lang_dir)-2]
    foreign_words = pd.read_csv(path+lang+'.totranslate.norm', header=None).values
    bias = process_bias_vec(args)
    bias /= np.linalg.norm(bias)
    bias = bias.reshape(bias.shape[1],)
    
    magnitude_path = args.vec_dir
    files = os.listdir(magnitude_path)
    
    # if a magnitude object doesn't exist, these variables remain empty dicts
    # all foreign words are skipped
    foreign_wiki, foreign_third, foreign_ex = {}, {}, {}
    # build 6 magnitude objects, remember flag -a 
    if 'wikiPMatrix.magnitude' in files:
        eng_wiki = Magnitude(magnitude_path+'wikiPMatrix.magnitude')
        foreign_wiki = Magnitude(magnitude_path+'wikiQMatrix.magnitude')
    if 'thirdTPMatrix.magnitude' in files:
        eng_third = Magnitude(magnitude_path+'thirdTPMatrix.magnitude')
        foreign_third = Magnitude(magnitude_path+'thirdTQMatrix.magnitude')
    if 'EPMatrix.magnitude' in files:
        eng_ex = Magnitude(magnitude_path+'EPMatrix.magnitude')
        foreign_ex = Magnitude(magnitude_path+'EQMatrix.magnitude')
    if 'user.'+lang+'.magnitude' in files: 
        foreign_mono = Magnitude(magnitude_path+'user.'+lang+'.magnitude')

    for i in range(foreign_words.shape[0]):
        word = foreign_words[i][0]
        wiki_trans, third_trans, ex_trans = [], [], []
        wiki_dict, third_dict = {}, {}

        # check if word exists in wiki
        if word in foreign_wiki:
            foreign_vec = foreign_wiki.query(word)
            wiki_trans = eng_wiki.most_similar_approx(foreign_vec, topn=10)
            wiki_dict = dict(wiki_trans)

        # check if word exists in third (crowdsource of third languages)    
        if word in foreign_third:
            foreign_vec = foreign_third.query(word)
            third_trans = eng_third.most_similar_approx(foreign_vec, topn=10)
            third_dict = dict(third_trans)

        # check if word exists in extended matrix
        if word in foreign_ex:
            foreign_vec = foreign_ex.query(word)
            ex_trans = eng_ex.most_similar_approx(foreign_vec, topn=10)
            bias_score = np.dot(bias, foreign_vec)
            ex_trans = [(w,v+bias_score) for w,v in ex_trans]
            
        # remove duplicates that already appear in more reliable matrices
        if len(wiki_trans) > 0: 
            third_trans = [t for t in third_trans if t[0] not in wiki_dict]
        ex_trans = [t for t in ex_trans if t[0] not in wiki_dict and t[0] not in third_dict]
        
        trans_total = wiki_trans + third_trans + ex_trans 
        if len(trans_total) > 0:
            trans = sorted(trans_total, key=lambda x:x[1], reverse=True)
        # back off to monolingual vectors if word doesn't exist in any of previous 3 matrices 
        else:
            foreign_vec = foreign_mono.query(word)
            trans = eng_ex.most_similar_approx(foreign_vec, topn=10)

        line = ''
        line = str(i) + ': ' + foreign_words[i][0] + ': '
        for j in range(len(trans)):
            line += str(trans[j])
            if j < len(trans)-1:
                line += ', '

        with open(path+'BPR_trans.txt', 'a') as f:
            f.write(line+'\n')

    print('### TRANSLATION IS COMPLETE ###')
    print('Translation outputs are stored at {}BPR_trans.txt'.format(args.lang_dir))

def find_monovec_translations(args): 
    path = args.lang_dir
    if os.path.exists(path+'monovec_trans.txt'):
        print('Warning: Translation outputs have already been produced.')
        exit()
    
    lang_dir = args.lang_dir.split('/')
    lang = lang_dir[len(lang_dir)-2]
    foreign_words = pd.read_csv(path+lang+'.totranslate.norm', header=None).values

    magnitude_path = args.vec_dir
    files = os.listdir(magnitude_path)
    
    if 'user.'+lang+'.magnitude' in files: 
        eng_ex = Magnitude(magnitude_path+'EPMatrix.magnitude')
        foreign_ex = Magnitude(magnitude_path+'user.'+lang+'.magnitude')
    
    for i in range(foreign_words.shape[0]):
        word = foreign_words[i][0]
        line = ''
        if word in foreign_ex:
            foreign_vec = foreign_ex.query(word)
            trans = eng_ex.most_similar_approx(foreign_vec, topn=10)

            line = str(i) + ': ' + foreign_words[i][0] + ': '
            for j in range(len(trans)):
                line += str(trans[j])
                if j < len(trans)-1:
                    line += ', '

        with open(path+'monovec_trans.txt', 'a') as f:
            f.write(line+'\n')

    print('### TRANSLATION IS COMPLETE ###')
    print('Translation outputs are stored at {}monovec_trans.txt'.format(args.lang_dir))

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('lang_dir', type=str, help='path to a language\'s directory')
    parser.add_argument('vec_dir', type=str, help='path to the directory of Magnitude')
    parser.add_argument('--vec', type=str, default='BPR', help='Type of vectors to use for translation: [BPR, monolingual')
    args = parser.parse_args()
    return args 

if __name__=='__main__':
    args = parse_args()
    if args.vec == 'BPR':
        find_BPR_translations(args)
    elif args.vec == 'monolingual':
        find_monovec_translations(args)
