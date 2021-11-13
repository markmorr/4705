# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 11:17:04 2021

@author: 16028
"""

import os
import argparse

import torch
from transformers import BertModel, BertConfig, BertTokenizer

import time

from nltk import word_tokenize
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity



LABELS = ['F', 'T']

def get_wic_subset(data_dir):
	wic = []
	split = data_dir.strip().split('/')[-1]
	with open(os.path.join(data_dir, '%s.data.txt' % split), 'r', encoding='utf-8') as datafile, \
		open(os.path.join(data_dir, '%s.gold.txt' % split), 'r', encoding='utf-8') as labelfile:
		for (data_line, label_line) in zip(datafile.readlines(), labelfile.readlines()):
			word, _, word_indices, sentence1, sentence2 = data_line.strip().split('\t')
			sentence1_word_index, sentence2_word_index = word_indices.split('-')
			label = LABELS.index(label_line.strip())
			wic.append({
				'word': word,
				'sentence1_word_index': int(sentence1_word_index),
				'sentence2_word_index': int(sentence2_word_index),
				'sentence1_words': sentence1.split(' '),
				'sentence2_words': sentence2.split(' '),
				'label': label
			})
	return wic


word_dict_train = get_wic_subset('wic/train')
word_dict_test = get_wic_subset('wic/dev')

# =============================================================================
# if __name__ == '__main__':
# 	parser = argparse.ArgumentParser(
# 		description='Train a classifier to recognize words in context (WiC).'
# 	)
# 	parser.add_argument(
# 		'--train-dir',
# 		dest='train_dir',
# 		required=True,
# 		help='The absolute path to the directory containing the WiC train files.'
# 	)
# 	parser.add_argument(
# 		'--eval-dir',
# 		dest='eval_dir',
# 		required=True,
# 		help='The absolute path to the directory containing the WiC eval files.'
# 	)
# 	# Write your predictions (F or T, separated by newlines) for each evaluation
# 	# example to out_file in the same order as you find them in eval_dir.  For example:
# 	# F
# 	# F
# 	# T
# 	# where each row is the prediction for the corresponding line in eval_dir.
# 	parser.add_argument(
# 		'--out-file',
# 		dest='out_file',
# 		required=True,
# 		help='The absolute path to the file where evaluation predictions will be written.'
# 	)
# 	args = parser.parse_args()
# =============================================================================





N = len(word_dict_train)
sent_list_form_train = []
sent_list_form_test = []
sent_concat_train = []
sent_concat_test = []
y_train = []
y_test = []

for i in range(N):
    sent_list_form_train.append(word_dict_train[i]['sentence1_words'])
    sent_list_form_train.append(word_dict_train[i]['sentence2_words'])
    sent_concat_train.append(" ".join(word_dict_train[i]['sentence1_words']))
    sent_concat_train.append(" ".join(word_dict_train[i]['sentence2_words']))

    y_train.append(word_dict_train[i]['label'])
    
  
y_train = np.array(y_train)
for i in range(len(word_dict_test)):
    sent_list_form_test.append(word_dict_test[i]['sentence1_words'])
    sent_list_form_test.append(word_dict_test[i]['sentence2_words'])
    sent_concat_test.append(" ".join(word_dict_test[i]['sentence1_words']))
    sent_concat_test.append(" ".join(word_dict_test[i]['sentence2_words']))
    y_test.append(word_dict_test[i]['label'])
    
  
y_test = np.array(y_test)
    
sent_list_form_train


text = " ".join(sent_concat_train)
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertModel.from_pretrained("bert-base-cased", output_hidden_states = True)




# =============================================================================
# encoded_input = tokenizer(text, padding="max_length", truncation=True, return_tensors='pt')
# output = model(**encoded_input)
# =============================================================================
#pass tokenized into model
#output =  model(tokenizer(sent))

# text = ["Replace me by any text you'd like.", "Hey I said.", "No way."]


# tokenizer([tokenizer1, tokenizer2], padding=True, return_tensors='pt')
#use all three input_id, other two things,

# may be useful but more for extra credit: # BertTokenizerFast, char_to_token(index in input sentence, maps to index of token in output)
#*** was using text before
sent_concat_train


word_index_dict = {}
list_of_sentences = []
for sent in sent_concat_train:
    list_of_sentences.append(word_tokenize(sent))
#convert to array 
#for sentence reps take cosine similarity
#each example has one sentence, so
# citing tutorial online for help: https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/#1-loading-pre-trained-bert
# encoded_input = tokenizer(list_of_sentences, padding="max_length", truncation=True, return_tensors='pt')

enc_input = []
for i in range(len(list_of_sentences)):
    enc_input.append(tokenizer(list_of_sentences[i], padding="max_length", truncation=True, return_tensors='pt'))


# data_dict = encoded_input.data


#citing edstem post 5.1.1. for style of reducing the runtime load for faster experiments
# s1_embeddings = self.bert_model(input_ids=s1_input_ids, attention_mask=torch.tensor(s1_attention_mask)).last_hidden_state
# s2_embeddings = self.bert_model(input_ids=s2_input_ids, attention_mask=torch.tensor(s2_attention_mask)).last_hidden_state
ab = data_dict['attention_mask']

N = data_dict['input_ids'].shape[0]

i = 5
a = data_dict['input_ids'][i:i+11]


encoded_input['input_ids'][i]
# sentence_reps_array = np.zeros(N)

start = time.time()

#################################################################################
#################################################################################
#################################################################################
word_dict_train = get_wic_subset('wic/train')

sent1_concat_train = []
sent2_concat_train = []
y_train = []

for i in range(len(word_dict_train)):
    sent_concat_train.append(" ".join(word_dict_train[i]['sentence1_words']))
    y_train.append(word_dict_train[i]['label'])
    
    
for i in range(len(word_dict_train)):
    sent2_concat_train.append(" ".join(word_dict_train[i]['sentence2_words']))


tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertModel.from_pretrained("bert-base-cased", output_hidden_states = True)
encoded_input = tokenizer(sent1_concat_train, padding=True, truncation=True,return_tensors='pt') # padding="max_length", truncation=True,

start = time.time()
N = len(sent_concat_train)
i = 0
inc_num = 5
ii = []
tti = []
am = []
hidden_states = []

with torch.no_grad():
    while i < len(sent1_concat_train):
        print(i)
        ii = encoded_input['input_ids'][i:i+inc_num]
        tti = encoded_input['token_type_ids'][i:i+inc_num]
        am = encoded_input['attention_mask'][i:i+inc_num]

        output=model(input_ids=ii, attention_mask=torch.tensor(am), token_type_ids=tti)
        hidden_states.append(output.last_hidden_state)
        i = i + inc_num 
        

end = time.time()
print(round(end-start,1))
###################################################################################

for tens in hidden_states:
    for i in tens:
        srmw = torch.mean(i, dim=0) 
        hidden_states[i] = srmw
        


#################################################################################

import pandas as pd

sent1_embeds = []
sent2_embeds = []

for i in range(len(hidden_states)):
    if i%2 == 0: 
        sent1_embeds.append(hidden_states[i])
    else:
        sent2_embeds.append(hidden_states[i])
        
both_list = []
for i in range(len(sent1_embeds)):
    both_list.append([sent1_embeds[i], sent2_embeds[i]])
    

diffs = []
cosine_sims = []
for i in range(len(sent1_embeds)):
    cosine_sims.append(cosine_similarity(both_list[i][0].reshape(1,-1), both_list[i][1].reshape(1,-1))[0][0])
    dif = both_list[i][0].reshape(1,-1) - both_list[i][1].reshape(1,-1)
    diffs.append(torch.squeeze(dif, dim=0))
    
    

    # cosine_sims.append(cosine_similarity(both_list[i][0], both_list[i][1]))

# ab = cosine_sims.append(cosine_similarity(sent1_embeds.reshape(1,-1), sent2_embeds.reshape(1,-1)))


a = both_list[10][0].reshape(-1,1)[767]
b = both_list[10][1].reshape(-1,1)[767]
a - b
dif[0][767]    
df = pd.DataFrame(
    # {'sent1_embeds': sent1_embeds,
     # 'sent2_embeds': sent2_embeds,
     # {'diffs': diffs,
     {'cosine_sim': cosine_sims
    })


X_train = np.array(df)
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=0).fit(X_train, y_train)
y_train_preds = clf.predict(X_train)
from sklearn.metrics import accuracy_score
accuracy_score(y_train, y_train_preds)

#################################################################################
#################################################################################
#################################################################################




###################################################################################
###################################################################################
###################################################################################
word_dict_test = get_wic_subset('wic/test')

sent_concat_test = []

y_test = []

for i in range(len(word_dict_test)):
    sent_concat_test.append(" ".join(word_dict_test[i]['sentence1_words']))
    sent_concat_test.append(" ".join(word_dict_test[i]['sentence2_words']))

    y_test.append(word_dict_test[i]['label'])
    
    


tokenizer = BertTokenizer.from_pretested('bert-base-cased')
model = BertModel.from_pretested("bert-base-cased", output_hidden_states = True)
encoded_input = tokenizer(sent_concat_test, padding=True, truncation=True,return_tensors='pt') # padding="max_length", truncation=True,

start = time.time()
N = len(sent_concat_test)
i = 0
inc_num = 5
ii = []
tti = []
am = []
hidden_states_array = []
with torch.no_grad():
    while i < len(sent_concat_test):
        print(i)
        ii = encoded_input['input_ids'][i:i+inc_num]
        tti = encoded_input['token_type_ids'][i:i+inc_num]
        am = encoded_input['attention_mask'][i:i+inc_num]

        output=model(input_ids=ii, attention_mask=torch.tensor(am), token_type_ids=tti)
        hidden_states_array.append(output.last_hidden_state)
        i = i + inc_num 
end = time.time()
print(round(end-start,1))

total_list = []
for tens in hidden_states_array:
    for i in tens:
        srmw = torch.mean(i, dim=0) 
        total_list.append(srmw)

####################################################################################
########################################################################################
###################################################################################




# =============================================================================
#         
# import pickle
# f = open("sample.pkl", "w")
# pickle.dump(hidden_states_array, f)
# f.close()
# =============================================================================
# print pickle.load(open("sample.pkl"))

# ad = hidden_states_array[0]


        
    
       

################################################################################





sreps = torch.mean(output.last_hidden_state, dim=1)
torch.squeeze(sreps, dim=0)
sentence_reps_array = []




###################################################################################
i = 0
inc_num = 5
with torch.no_grad():
    while i < N:
        # loop_start = time.time()
        print(i)
        # ii = data_dict['input_ids'][i:i+inc_num]
        # tti = data_dict['token_type_ids'][i:i+inc_num]
        # am = data_dict['attention_mask'][i:i+inc_num]
        ii = encoded_input['input_ids'][i:i+inc_num]
        tti = encoded_input['token_type_ids'][i:i+inc_num]
        am = encoded_input['attention_mask'][i:i+inc_num]
        # amf = torch.unsqueeze(am, 0)
        # iis = torch.unsqueeze(ii,0)
        s1 = time.time()    
        output=model(input_ids=ii, attention_mask=torch.tensor(am), token_type_ids=tti)
        s2 = time.time()
        print(round(s2-s1,1))
        hidden_states_array.append(output.last_hidden_state)
        # print(output.shape)

        
        # end_loop = time.time()
        # print(round(end_loop - loop_start,1))
        i = i + inc_num #gonna have to adjust last add so we don't overshoot the last index i.e. N%mod10 something
end = time.time()
print(round(end-start,1))


sreps = torch.mean(output.last_hidden_state, dim=1)
torch.squeeze(sreps, dim=0)
sentence_reps_array = []
###################################################################################
# X_train
# 10856 x 768 --> 1536 features per sample?



af = encoded_input_sent1['token_type_ids'].data[1:10]


# use cosine similarity as a feature or what?






start = time.time()
with torch.no_grad(): # is your friend #this'll keep track of the full computation graph, somehow import for RAM
    output = model(**encoded_input)

    # output = model(**tokenized) #what do the stars mean?
    #(composing way of writing it output = model(**tokenizer([sentence], return_tensors='pt'))) #is your friend
    print(output.__class__) #just checking     
    print(output.last_hidden_state.shape) #these are our contextualized representations 
    hidden_states = output[2]
    # (i.e. if it was 2 sentences, with max length of 13, it would be shape 2,13,768. (don't know why 768))
# sentence_reps = torch.mean(output.last_hidden_state.shape, dim=1) #would be 
#might be better for performance to only select nonzero indices etc. becuase we used padding
sentence_reps = torch.mean(output.last_hidden_state, dim=1) #would be 
hidden_states
end = time.time()
print(round(end-start,2))


a0 = output[0]
a1 = output[1]
a2 = output[2]


print ("Number of layers:", len(hidden_states), "  (initial embeddings + 12 BERT layers)")
layer_i = 0

print ("Number of batches:", len(hidden_states[layer_i]))
batch_i = 0

print ("Number of tokens:", len(hidden_states[layer_i][batch_i]))
token_i = 0

print ("Number of hidden units:", len(hidden_states[layer_i][batch_i][token_i]))

model2 = model.eval() #is your friend
model2

#tokens padde to 512
#BERT just uses a 768 dimensional word embedding? 
# `hidden_states` is a Python list.
print('      Type of hidden_states: ', type(hidden_states))

# Each layer in the list is a torch tensor.
print('Tensor shape for each layer: ', hidden_states[12].size())


# token_embeddings = torch.stack(hidden_states, dim=0)
token_embeddings = hidden_states[12]
token_embeddings.size()


# token_embeddings = torch.squeeze(token_embeddings, dim=1)
token_embeddings = torch.squeeze(token_embeddings, dim=0)

token_embeddings.size()
token_embeddings
# Swap dimensions 0 and 1.
# token_embeddings = token_embeddings.permute(1,0,2)
# token_embeddings.size()

sent_rep_my_way = torch.mean(token_embeddings, dim=0)
sent_rep_my_way
sentence_reps = torch.mean(output.last_hidden_state, dim=1) #would be 
sentence_reps = torch.squeeze(sentence_reps, dim=0)

sentence_reps


word_dict_test

# Convert inputs to PyTorch tensors
# tokens_tensor = torch.tensor([indexed_tokens])
# segments_tensors = torch.tensor([segments_ids])



# =============================================================================
# from sklearn.linear_model import LogisticRegression
# clf = LogisticRegression(random_state=0).fit(X_train, y_train)
# y_preds = clf.predict(X_test)
# =============================================================================

#pt 3
#torch.load to load them
#opening the files, using your knowledge of Bert to explore these and plot F1 in matplotlib
# =============================================================================
# conll = torch.load(path)
# conll.keys()
# len(conll(train))
# =============================================================================

#lower case might help
#stripping punctutation migth also help
#should i filter out stopwords?
#cache the embedddings and anything else useful to the directory from part 1
#should i worry about lemmas and stems in pre-processing? probs wouldn't help?
#clean my code?
# we are doing logistic regression for each layer
# don't stress about that run time it took him 10-15 minutes
#they are just looping through the layers
#relation extraction: get the represenation for those 2 words, concat them, and then train your representation on that
# =============================================================================
# conll[train][0]['pos_labels']    
# conll[train][0]['sentence']
# 
# =============================================================================
