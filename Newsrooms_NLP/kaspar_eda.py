#!/usr/bin/env python
# coding: utf-8

# Heavily adapted from: 
# https://github.com/tobyatgithub/bert_tutorial/blob/master/Bert_tutorial1_embeddings.ipynb
# 
# We will be using pretrained BERT model to go from raw words into latent embeddings
# #### word -> tokens -> ids -> hidden states -> embeddings

# In[ ]:


# get_ipython().system('conda install pytorch torchvision -c pytorch')
# get_ipython().system('pip install pytorch_pretrained_bert')


# In[1]:


import pandas as pd
import nlp
#import nltk
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel


# In[2]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[3]:


# labels = nlp.load_labels('articles_db.db')
corpus = nlp.load_articles('articles_db.db')
df = pd.DataFrame(corpus, columns=['title', 'body'])
df.head(2)


# In[4]:


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# In[29]:


df['prepped_body'] = df['body'] + " [SEP]"
df.loc[0, 'prepped_body'] = "[CLS] " + df.loc[0, 'prepped_body']
df['tokens'] = None
len(df)


# In[30]:


for i, text in enumerate(df['prepped_body']):
    df.loc[i, 'tokens'] = tokenizer.tokenize(text)
    df.loc[i, 'tokens'] = df.loc[i, 'tokens'][:512]


# In[36]:


# word -> tokens -> ids -> hidden states -> embeddings

all_tokens = []
input_type_ids = []
# masks for segment, 0 for the first sentence, 1 for the second sentence.
# use 1 if there's only one sentence.

for i, tokens in enumerate(df['tokens']):
    for token in tokens:
        all_tokens.append(token)
        input_type_ids.append(i)
print(len(input_type_ids))
input_type_ids = input_type_ids[:511]
print(len(input_type_ids))
# print("tokens:", tokens)   
# print("type_ids:", input_type_ids)


# In[37]:


len(all_tokens)


# In[38]:


# We can only use 512 tokens with BERT
input_ids = tokenizer.convert_tokens_to_ids(all_tokens[:511])
for pair in zip(tokens[:25], input_ids[:25]):
    print(pair)
# notice the case ---> uncased


# In[43]:


# padding
seq_length = 512 # max allowed length & padding length for each pair of sentences. 512
input_mask = [1] * len(input_ids)
print(input_ids[:20])
print(input_mask[:20])
print(input_type_ids[:20])
while len(input_ids) < seq_length:
    input_ids.append(0)
    input_mask.append(0)
    input_type_ids.append(0)
    
print()
print(input_ids[:20])
print(input_mask[:20])
print(input_type_ids[:20])


# In[40]:


print(len(input_ids))
print(len(input_mask))
print(len(input_type_ids))


# In[41]:


# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased')
# model = model.cuda()
# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval();


# In[42]:


# Predict hidden states features for each layer
with torch.no_grad():
    # ids -> hidden state vectors
    input_tensor = torch.LongTensor(input_ids).view(-1,1)
    input_mask = torch.LongTensor(input_mask).view(-1,1)
    input_type_ids = torch.LongTensor(input_type_ids).view(-1,1)
    
    print(input_tensor.shape)
    print(input_mask.shape)  
    print(input_type_ids.shape)
    encoded_layers, _ = model(input_tensor, token_type_ids=input_type_ids, attention_mask=input_mask)


# In[51]:


# to get the token embedding vector, we can sum the last four
#print(text_a, text_b)
print("sum the last four")
sum_last_four = torch.sum(torch.stack(encoded_layers[-4:]), dim=0)
print('\n\n', sum_last_four.shape)


# In[52]:

print(torch.cat(encoded_layers[-4:]).shape)



