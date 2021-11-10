# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 13:33:15 2021

@author: 16028
"""

# import pandas as pd
import numpy as np
import nltk
from nltk.corpus import brown
import gensim

from nltk import word_tokenize, sent_tokenize
import pdb
import scipy as sp
from scipy.sparse import csr_matrix, lil_matrix
import math
import time


# df = pd.read_csv('data/brown.txt', sep="\n", header=None)

with open('data/brown.txt', 'r') as file:
    full_str = file.read()
# full_str = ' '.join(df[0])

full_str = full_str[:10000]
full_str = full_str.lower()
tokens = word_tokenize(full_str)

context_window = 3
full_str = 'The dog went to the store. The cat ate in the hat. Cats went cats to the store.'# with other cats. Cat went flag to store building.'
# full_str = 'a b a c d d a a c. b e f f g g b k. l m a d n o p they said. Hey! Stop. Quit.'

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
import pandas as pd
output_check2 = pd.DataFrame(data=M_numpy, index=V, columns=V)
ab = M.data
atest = np.array([[0,1], [4, 8]])
atest.sum(axis=1)

word_sums = M.sum(axis=1) #these will become denominators?
context_sums = M.sum(axis=0)
context_sums = context_sums.T
total_count = M.sum()


word_sums = word_sums/total_count
context_sums = context_sums/total_count
probs = M/total_count

word_sums_log = np.log(word_sums)
context_sums_log = np.log(context_sums)
probs.data = np.log(probs.data)

print('is this line ruining it?')

# yc = np.nan_to_num(probs.data)

aa = probs.data

word_sums.std()
context_sums.std()

pmi = csr_matrix((N, N), dtype=np.float32)
lil_pmi = lil_matrix((N,N), dtype=np.float32)
start = time.time()


for i in range(N):
    if i%20 == 0: print(i)
    for j in range(N):
        # PMI = log(a,b) - log(a) - log(b)
# =============================================================================
# =============================================================================
        # new_value = probs[i,j] - word_sums_log_filled[i] - context_sums_log_filled[j]
        # new_value = probs[i,j] - word_sums_log[i] - context_sums_log[j]
        new_value = np.log(probs[i,j]/(word_sums[i]*context_sums[j]))
        if new_value > 0:
            # pmi[i,j] = new_value
            lil_pmi[i,j] = new_value
# =============================================================================
#         # word_sums_log_filled, context_sums_log_filled 
#         new_value = max(new_value, 0)
#         if probs[i,j] == float("inf"):
#             new_value = 0
#         #we don't want to be setting it if it's 0 cause that'll mess up the proce
#         if new_value != 0:
#             pmi[i,j] = new_value
# =============================================================================
        # lil_pmi[i,j] = new_value
end = time.time()
print(round(end-start,2))
# pmi

aa = lil_pmi.data #this step ruins sparsity, will have to fix #(may take the max on the numpy array before subtracting
#and passing it into the new sparse matrix)
pmi_conv = lil_pmi.tocsr()
pmi_conv.data


ab = pmi_conv.data
ac = probs.data

k_value = 50
U,S,VT = sp.sparse.linalg.svds(pmi, k = 5)

U_csr = csr_matrix(U)
VT_csr = csr_matrix(VT)
S_csr = csr_matrix(S)
V_csr = VT_csr.transpose()

S_rooted_csr = S_csr.power(.5)
W = U_csr.multiply(S_rooted_csr)
C = V_csr.multiply(S_rooted_csr)


myg = word_tokenize("I do say I do not")

from gensim.models import Word2Vec, KeyedVectors
model = Word2Vec(sent_list_raw,window=3, sg=1, negative=5, epochs=50)
# model.train(sent_list, total_examples=model.corpus_count, epochs=model.epochs)

# model.train(sentences=sent_list)
# https://radimrehurek.com/gensim/models/word2vec.html citing for help
keyed_word_vectors = model.wv
keyed_word_vectors
model.wv.save('keyed_word_vectors.bin') #.bin?
model.wv.save('vectors.kv')


vector = model.wv['resolution']  # get numpy vector of a word
sims = model.wv.most_similar('computer', topn=10)  # get other similar words
# wv = KeyedVectors.load("word2vec.wordvectors", mmap='r')
# =============================================================================
# 
# for each model, train 5 different variants, changing hyperparameters
# like their context window size (2, 5 or 10), their dimensionality (50, 100, or 300) or the
# number of negative samples they use (1, 5 or 15, applicable to SGNS only).
# =============================================================================
context_window_size = [2,5]
dimensionality = [50, 100, 300]
sgns = [1,5] 

import string
sent_list_stripped = [s.translate(str.maketrans('', '', string.punctuation)) for s in sent_list_raw]

#fix context window size at 2 for word2vec so we only get 6 models
for dim in dimensionality:
    for sgn in sgns:
        model = Word2Vec(sentences=sent_list_raw,window=2, sg=1,  negative=sgn) # vector_size = dim, 
        keyed_word_vectors = model.wv
        model.wv.save('keyed_word_vectors' + '_' + str(dim) + '_' + str(sgn) + '.bin') #.bin?
        
def runSVD(context_window, dimensionality):
    print(context_window)
    print(dimensionality)        
for con in context_window_size:
    for dim in dimensionality:
        runSVD(con, dim)
        
        

        