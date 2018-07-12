#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 15:17:30 2018

tf-idf

@author: xinji
"""
import pandas as pd
df = pd.read_csv('channel_info.csv')
list(df)
df.description[1000:10005]
df.name[1]
df.pid[50]
df = df.fillna('-1')
import re
rgx_list = ['<p>', '\n', '\t','</p>','</strong>','<strong>','\r',
          '</span>','\xa0ni' ,'\xa0' ,'<meta charset="utf-8">','<br>',
          '\xa030%','\xa0','<p class="p1">','<span class="s1">','<span>']

def clean_text( text,rgx_list = rgx_list):
    new_text = text
    for rgx_match in rgx_list:
        new_text = re.sub(rgx_match, '', new_text)
    new_text = new_text.split('Dimension')[0]
    new_text = new_text.split('longueur')[0]
    new_text = re.sub(r'\{.*\}', '', new_text)
    new_text = re.sub(r'\(.*\)', '', new_text)
    new_text = re.sub(r'\<.*\>', '', new_text)
    return new_text
j = clean_text(df.description[1])
product_description = []
for d in df.description:
    temp = clean_text(d)
    product_description.append(temp)

df.clean_description = product_description

df.to_csv('channel_info.csv',index = False)

full_text = []
for content in product_description:
    x  = re.sub('[^a-zA-Z0-9\n\.]', ' ', content)
    full_text = full_text + [x]
# =============================================================================
# TF-IDF
# =============================================================================

from string import punctuation
from nltk.corpus import stopwords
from nltk import word_tokenize
 
stop_words = stopwords.words('english') + list(punctuation)
# build the vocabulary in one pass

def tokenize(text):
    words = word_tokenize(text)
    words = [w.lower() for w in words]
    return [w for w in words if w not in stop_words and not w.isdigit()]

vocabulary = set()
for d in product_description:
    words = tokenize(d)
    vocabulary.update(words)
 
vocabulary = list(vocabulary)
word_index = {w: idx for idx, w in enumerate(vocabulary)}
 
vocabulary_size = len(vocabulary)
doc_count = len(product_description)
 
print (vocabulary_size, doc_count)







