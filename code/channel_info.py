#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 11:11:22 2018

@author: xinji
"""

import pandas as pd
from pymongo import MongoClient
client = MongoClient( host = 'tombraider.shopcade.com',port = 27031  )
db2 = client['live_channel_products']
coll2 = db2.get_collection('shopping_elle_fr_channel_products')
    
agg = coll2.aggregate([
        {"$group":{"_id":{"merhchant":"$m.n",
                  'price':'$pr',
                  'cate':'$cat',
                  "pid":"$_id",
                  'name':'$n',
                  'cc':'$cc',
                  'description':'$d',
                  'brand':'$b.sid',
                  'cur':'$cur'},
        "sum":{"$sum":1}}}],
         allowDiskUse = True)
product_info  = [doc for doc in agg]

channel_name = [pid['_id']['name'] for pid in product_info]
channel_pid = [str(pid['_id']['pid']) for pid in product_info]
merhchant = [pid['_id']['merhchant'] for pid in product_info]
cat = [pid['_id']['cate'] for pid in product_info]
cc = [pid['_id']['cc'] for pid in product_info]
cur = [pid['_id']['cur'] for pid in product_info]
b = [pid['_id'] for pid in product_info]
price = [pid['_id']['price'] for pid in product_info]
brand = []
description = [pid['_id']['description'] for pid in product_info]
for x in b:
    try:
        brand.append(x['brand'])
    except KeyError:
        brand.append('Unkown')
c = []
for x in cat:
    if x is not None:
        c.append(x)
    if x is None:
        x = [1,0,0,0]
        c.append(x)
cat1 = []
for x in c:
    try:
        cat1.append(x[1])
    except IndexError:
        cat1.append(0)
channel_info = pd.DataFrame({'name':channel_name,'pid':channel_pid,'merhchant':merhchant,'cat1':cat1,'price':price,
                            'cc':cc,'cur':cur,'brand':brand,'description':description })
channel_info.to_csv('channel_info.csv',index = False)


# =============================================================================
# one hot info
# =============================================================================
cat1 = pd.get_dummies(channel_info['cat1'],prefix='cat1')
cc = pd.get_dummies(channel_info['cc'],prefix = 'cc')
cur = pd.get_dummies(channel_info['cur'],prefix = 'cur')
merchant = pd.get_dummies(channel_info['merhchant'],prefix = 'merchant')
brand = pd.get_dummies(channel_info['brand'],prefix = 'brand')
onehot_channel = pd.DataFrame({'name':channel_info.name,'pid':channel_info.pid,'description':channel_info.description})
onehot_channel = pd.concat([onehot_channel,cat1,cc,cur,merchant,brand],axis = 1)

onehot_channel.to_csv('onehot_product.csv',index = False)





