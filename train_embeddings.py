# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 13:33:15 2021

@author: 16028
"""

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import brown
import gensim

from nltk import word_tokenize, sent_tokenize
import pdb
from scipy.sparse import csr_matrix




df = pd.read_csv('data/brown.txt', sep="\n", header=None)
full_str = ' '.join(df[0])

tokens = word_tokenize(full_str)
# PMI is?
# Mij = max{0, PMI(wi, wj )},
# =============================================================================
# 
# PMI(a, b) = log p(a, b)/p(a)p(b)
# 
# conda install smart-open==1.9.0
# conda install gensim==3.8.3
# =============================================================================

V = set(tokens)

sent_list = sent_tokenize(full_str) 
# df['co-occurrences'] = df[0].apply()

    
# [word_tokenize(t) for t in sent_tokenize(s)]
# =============================================================================
# 
# vocab_dict = {}
# for word in V:
#     hit_dict = {}
#     for sent in sent_list:
#         if word in sent:
#             for context_word in sent:
#                 hit_dict[context_word] = 1
#     vocab_dict[word] = hit_dict
#     
#     
# =============================================================================


context_window = 3
# full_str = 'The dog went to the store. The cat ate in the hat. Cats went cats to the store.'# with other cats. Yo Cat went flag to store building.'
full_str = 'a b c d e f g h. i j k. l m n o p they said. Hey! Stop. Quit.'
full_str = full_str.lower()
sent_list_raw = sent_tokenize(full_str)
tokens = word_tokenize(full_str)
V = set(tokens)
N = len(V)

word_index_dict = {}
list_of_sentences = []
for sent in sent_list_raw:
    list_of_sentences.append(word_tokenize(sent))
    
#it's including itself as an index (remove later)
#it's also weirdly like one less than the context windwo as well? as in iit's 
#check ranges and indices etc.
M_numpy = np.zeros((N,N), dtype=int)
M = csr_matrix((N, N), dtype=np.float32)
order = range(0,N)
for i in order:
    print(i)

convert_word_to_index_dict = dict(list(zip(V, order)))

for word in V: 
    word_index_dict[word] = {}
    sentence_num = 0
    for sentence_tokenized in list_of_sentences:

        if word in sentence_tokenized:
            word_locs = [i for i, x in enumerate(sentence_tokenized) if x == word] # citing: https://stackoverflow.com/questions/6294179/how-to-find-all-occurrences-of-an-element-in-a-list
            good_indices = []
            for word_location in word_locs:
                max_word_index = len(sentence_tokenized) - 1 #CHECK
                for i in range(1,context_window + 1): #CHECK
                    if ((word_location-i) >= 0):
                        good_indices.append(word_location-i)
                    # good_indices.append(max(word_location - i, 0))
                    # j = that
                    # i = the word index
                    # M[i][j] += 1
                for i in range(1,context_window + 1): #CHECK
                    if ((word_location+i) <= max_word_index):
                        good_indices.append(word_location+i)
                    # good_indices.append(min(word_location + i, max_word_index))
            good_indices = set(good_indices)

            word_index_dict[word][sentence_num] = good_indices
        sentence_num += 1
    word_index = convert_word_to_index_dict[word]
    
    current_dict = word_index_dict[word]
    
    for key,value_set in current_dict.items(): #loop through dictionary
    #for each sentence index and each corresponding set
        for context_word_position in value_set: #for each
            # pdb.set_trace()
            context_index = convert_word_to_index_dict[list_of_sentences[key][context_word_position]]
            # print('base word: ' + word)
            # print(word_index)
            # print('context word: ' + list_of_sentences[key][context_word_position])
            # print(context_index)
            M_numpy[word_index][context_index] += 1
            M[word_index,context_index] += 1
            
# M1 = M.copy()
output_check2 = pd.DataFrame(data=M_numpy, index=V, columns=V)

import scipy as sp
k = 50
U,S,VT = sp.sparse.linalg.svds(M, k = 5)

U_csr = csr_matrix(U)
VT_csr = csr_matrix(VT)
S_csr = csr_matrix(S)
V_csr = VT_csr.transpose()

S_rooted_csr = S_csr.power(.5)
W = U_csr.multiply(S_rooted_csr)
C = V_csr.multiply(S_rooted_csr)

word_sums = M.sum(axis=1) #these will become denominators?
context_sums = M.sum(axis=0)

# joint_sums = M/
# oh wait are those identical?
#also don't i have to normaliZE AND DIVIDE BY TOTAL COUNT AT SOME POINT?
PMI = csr_matrix((N, N), dtype=np.float32)
total_count = M.sum()



pij = M/total_count
pij.data
M.data

context_sums = context_sums.T



for i in range(N):
    for j in range(N):
        print(pij[i,j])
        print(word_sums[i])
        print(context_sums[i])
        PMI[i,j] = pij[i,j] /(word_sums[i] * context_sums[j])
        
PMIt = np.log(PMI)
now do max with o

# PMI = Pi

lg = M_sparse.sum()
lg

M_fixed

yg = M - M_fixed
yg
# =============================================================================
# # works
# word_index_dict = {}
# list_of_sentences = []
# for sent in sent_list_raw:
#     list_of_sentences.append(word_tokenize(sent))
#     
# #it's including itself as an index (remove later)
# #it's also weirdly like one less than the context windwo as well? as in iit's 
# #check ranges and indices etc.
# M = np.zeros((N,N), dtype=int)
# M
# for word in V: 
#     word_index_dict[word] = {}
#     sentence_num = 0
#     for sentence_tokenized in list_of_sentences:
# 
#         if word in sentence_tokenized:
#             word_locs = [i for i, x in enumerate(sentence_tokenized) if x == word] # citing: https://stackoverflow.com/questions/6294179/how-to-find-all-occurrences-of-an-element-in-a-list
#             good_indices = []
#             for word_location in word_locs:
#                 max_word_index = len(sentence_tokenized) - 1 #CHECK
#                 for i in range(1,context_window + 1): #CHECK
#                     good_indices.append(max(word_location - i, 0))
#                 for i in range(1,context_window + 1): #CHECK
#                     good_indices.append(min(word_location + i, max_word_index))
#                     
#             good_indices = set(good_indices)
# 
#             word_index_dict[word][sentence_num] = good_indices
#         sentence_num += 1
# =============================================================================
            
    
    # word_loc = listy[word]
    
    
    
    
    
    

# good_indices =
# property_asel = [property_a[i] for i in good_indices]

for i in range(5):
    print(i)
    
    
vocab_dict = {}
for word in V:
    hit_dict = dict.fromkeys(V,0)
    for sent in sentence_list:
        if word in sent:
            for context_word in sent:
                if context_word 
                hit_dict[context_word] += 1
    vocab_dict[word] = hit_dict
        

myg = word_tokenize("I do say I do not")


# for word2vec use the straight save function that comes in the documentation
#play around with the number of epochs if score is terrible
# SVD side only ever operate on sparse matrices
# should be able to get naive, try simple models, should get .6

#import spearmanr
#import cosine similarity
#import norm?
N = len(V)
from scipy.sparse import csr_matrix
csr_matrix((N, N), dtype=int)

c1 = 1938
c2 = 1311
c3 = 1159
np.log(c2/(c2 + c3))


#careful with matrix divisions or multiplicaton making stuff unsparse
# 

# PMI = log(a,b) - log(a) - log(b)

#accuracy scores of Bert should be pretty easy to bet
#static embeddings means giant dicitionaries, look it up, that's that
row = np.zeros(len(V))
col = np.array([0, 2, 2, 0, 1, 2])




from gensim.models import Word2Vec, KeyedVectors
model = Word2Vec(sentences=sent_list, window=3, sg=1, negative=5)
# https://radimrehurek.com/gensim/models/word2vec.html citing for help
keyed_word_vectors = model.wv
# wv = KeyedVectors.load("word2vec.wordvectors", mmap='r')
model.wv.save('keyed_word_vectors') #.bin?
