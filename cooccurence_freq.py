#!/usr/bin/env python3
# coding: utf-8
#Li Tang

# In[107]:


def guess_codec(s):
    """
    guess the encoding of a given string
    Params:
        path: string, path+name of file
    Return:
        codec: string, guessed codec
    """
    codecs = ['ascii', 'utf32', 'utf16', 'utf8', 'cp1252']
    for codec in codecs:
        if _test_codec(s, codec):
            return codec

    raise ValueError('ERROR! Unknown encoding.')
    
def _test_codec(byte_stream, codec):
    """
    Params:
        byte_stream: file-like object, bytes file open open for reading
        codec: string, encoding to be tested on byte_streams
    Return:
        bool, True if decode throws no Error, False otherwise
    """
    try:
        byte_stream.decode(codec)
        return True
    except UnicodeDecodeError:
        return False


# In[108]:


import gzip
import sys

texts = []
filename = sys.argv[1]
print(filename)
minfreq = int(sys.argv[2])
print("Opening file:", filename)
with gzip.open(filename, 'rb') as f:
    for line in f.readlines():
        text = line.decode("utf-8")
        x = text.encode('utf-8', 'ignore').decode()
        texts.append(x)
f.close()
print("sentences:", len(texts))


# In[102]:


import os, re
from gensim.parsing.preprocessing import strip_punctuation

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stopword_list = stopwords.words('german')
def remove_stopwords(tokens):
    all_tokens = []
    for t in tokens:
        filtered_tokens = [token for token in t if token not in stopword_list]
        all_tokens.append(filtered_tokens)
    return all_tokens

def preprocess(texts):
    #tokenization
    texts = [re.findall(r'\w+', line.lower()) for line in texts]
    #remove stopwords
    texts = remove_stopwords(texts)
        
    #remove words that are only 1-2 character
    newtexts = []
    for s in texts:
        cleans = [token for token in s if len(token) > 2]
        newtexts.append(cleans)
    
    
    #remove numbers
    newtexts2 = []
    for s in newtexts:
        cleans = [token for token in s if not token.isnumeric()]
        newtexts2.append(cleans)
    
    
    #lemmatization
    lemmatizer = WordNetLemmatizer()
    newtexts3 = []
    for s in newtexts2:
        for w in s:
            cleanw = lemmatizer.lemmatize(w, pos='v')
            newtexts3.append(cleanw)
    
    
    return newtexts3

processed_texts = preprocess(texts)


# In[103]:


import concurrent.futures
from collections import Counter

def cooccurrances(idx, tokens, window_size):

    # beware this will backfire if you feed it large files (token lists)
    window = tokens[idx:idx+window_size]    
    first_token = window.pop(0)

    for second_token in window:
        yield first_token, second_token

def harvest_cooccurrances(tokens, window_size=5, n_workers=5):
    l = len(tokens)
    harvest = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        future_cooccurrances = {
            executor.submit(cooccurrances, idx, tokens, window_size): idx
            for idx
            in range(l)
        }
        for future in concurrent.futures.as_completed(future_cooccurrances):
            try:
                harvest.extend(future.result())
            except Exception as exc:
                # you may want to add some logging here
                continue


    return harvest


# In[104]:


def count(harvest):
    return [
        (first_word, second_word, count) 
        for (first_word, second_word), count 
        in Counter(harvest).items()
    ]

#harvest = harvest_cooccurrances(processed_texts, 5, 5)
#counts = count(harvest)

#print(counts)


# In[106]:


allwords = []
print("words:", len(processed_texts))
for w in processed_texts:
    allwords.append(w)
harvest = harvest_cooccurrances(allwords, 5, 5)
counts = count(harvest)
for c in counts:
    if c[2] > minfreq:
        print(c)

