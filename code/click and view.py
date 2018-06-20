#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
author:wang xinji

date:dd-mm-yyyy

"""
import operator
import pymongo
import numpy as np
from pymongo import MongoClient
import pandas as pd
from collections import Counter
import seaborn as sns
import os 
import datetime
import time
start = time.time()

os.chdir('/Users/xinjiwang/Desktop/data')

tombraider = MongoClient(host=['tombraider.shopcade.com:27031'])
trkelle = tombraider.lancaster.trkelle
temp = tombraider.lancaster.temp

def pageview(collection_name,start_date, end_date, date_format="%Y-%m-%d"):
   """export pageviews from trkelle or public"""
   data = collection_name
   ts1 = time.mktime(datetime.datetime.strptime(start_date, date_format).utctimetuple())
   ts2 = time.mktime((datetime.datetime.strptime(end_date, date_format)+datetime.timedelta(days=1)).utctimetuple())
   data.aggregate([
           {"$match":{"tag":"page-load",
                       "ts":{"$gt":ts1,
                             "$lt":ts2}}},
           {"$project":{"data.rfr":1,
                       "scid":1,
                       "ts":1,
                       "ref":1,
                       "host":1,
                       "loc-0":1,
                       "loc-1":1,
                       "loc-2":1,
                       "loc-3":1,
                       "ip":1,
                       "ip_city":1,
                       "ip_cc":1,
                       "ua":1,
                       "dim_w":1,
                       "dim_h":1,
                       "dim_sl":1,
                       "dim_st":1}},
            {"$out":"temp"}],
            allowDiskUse=True)
   df = [doc for doc in temp.find()]
   df = pd.DataFrame(df)
   return df

def check(key,data):
    l = []
    for i in range(len(data)):
        if key in data[i].keys() :
#            print(data[i][key])
            l.append(data[i][key])
    l1 = list(set(l))
    print('count of key',key,' in dict is ' ,len(l))
    print('after delete duplicae, got ',len(l1))
    return l,l1
#create subset size = 5000 acorrding to tag name 
def tag_subset(tag_name,collection):    
    data = []
    n = 0
    for doc in collection.find({'tag':tag_name}):
        n +=1
    #    print(doc)
        data.append(doc)

    return data

def sort_array(array):
    score = dict(Counter(array))
    sorted_score = sorted(score.items(), key=operator.itemgetter(1))
    sort = []
    for i in reversed(sorted_score):
        sort.append(i)
    return sort

client = MongoClient( host = 'tombraider.shopcade.com',port = 27031  )
db = client['lancaster']

print(db.collection_names())
names = db.collection_names(include_system_collections=False)

collection = db.collections
coll1 = db.get_collection(names[0])
coll1.find({'tag':'dynamic-product-view'}).count()
    
dynamic_click = tag_subset('dynamic-product-view',coll1)

check('tag',dynamic_click)

id_ = []
wid = []
iid = []
for i in range(len(dynamic_click)):
#    if 'wid' in dynamic_click[i]['data2'].keys() == True:
#        if 'iid' in dynamic_click[i].keys() == True:                
    id_ .append(dynamic_click[i]['_id'])
    wid.append(dynamic_click[i]['data2']['wid'])
    iid.append(dynamic_click[i]['iid'])

wid_score = pd.DataFrame(list(sort_array(wid)))
iid_score = pd.DataFrame(list(sort_array(iid)))
wid_score.columns = ['model','click count']

#rank widget according to product clicked in the widget
i = 0
total_product = []        
rank_of_products =[]
for model_id in wid_score['model']:
    docs = coll1.find({'data.wid':model_id}) 
    pids = []
    rank_of_product = []
    for doc in docs:
        pids.append(doc['data']['pid'])
        rank_of_product.append(wid_score['click count'][i])
    i +=1
    total_product.append(pids)
    rank_of_products.append(rank_of_product)
df = pd.DataFrame({'pid':total_product,'rank':rank_of_products})
product_list = [y for x in total_product for y in x]
rank_list = [y for x in rank_of_products for y in x]
len(product_list)
len(rank_list)
x = [product_list,rank_list]
y  = Counter(x[0])
data = [y.keys(),y.values()]
df = pd.DataFrame({'item':list(y.keys()),'widget_click':list(y.values())})

#df.to_csv('widget_and_clicks')
df = pd.io.parsers.read_csv('avg_click.csv')
#number a product loaded
load = pd.io.parsers.read_csv('tagpid.csv')
load1 = load.dropna()
product_load_count = load1[load1.tag == 'product-load']
count = Counter(list(load1['data.pid']))
show_time = []
for pid in df['item']:
    if pid in count.keys():
        show_time.append(count.get(pid))
    else:
        show_time.append(1)
df['load_time'] = pd.Series(show_time,index = df.index)

def pip(pid):
    pipeline = [
            {'$match':{
                    '$and': [  
                            { "tag": {'$in':['dynamic-product-view','list-product-view','product-product-view','button-product-view']}},
                            {'data':{'$in':[pid]}}]}},
            {'$group': {'_id': '$scid','count': {'$sum': 1}}}]
    return pipeline
coll2 = db.get_collection('trkelle_view')
view_number = []
for pid in df['item']:
    x = list(coll2.aggregate(pip(pid))) #given a produdct id return a list of number a product is viewed.
    n = sum(item['count'] for item in x)
    view_number.append(n)

df['view'] = pd.Series(view_number,index = df.index)
#get the average frequency a viewed product in a widget is click
df['view_perclick'] = df['widget_click'].divide(df['view'])
df = df.replace([np.inf, -np.inf],0 )

df = df.sort_values('view_perclick',ascending=False)
df.to_csv('avg_click.csv')


#get the frequency a product is click and click-per load
clicks_ =  pd.io.parsers.read_csv('tag.csv')
product_view = clicks_[clicks_.tag == 'product-click']
product_view = product_view.drop_duplicates(keep = 'last')
count2 = Counter(list(product_view['data.pid']))
click_time = []
for pid in df['item']:
    click_time.append(count2.get(pid))
for i in range(len(click_time)):
    if click_time[i] == None:
        click_time[i] = 0
df['product_click'] = pd.Series(click_time,index = df.index)
#df.columns = ['widget_click','item','count','view','avg_wid_click','product_click']
df['avg_pd_click'] = pd.Series(df['product_click'].divide(df['view']))
df.fillna(0, inplace=True)
df = df.replace([np.inf, -np.inf],0 )

#sorted accordfing rtto 
df = df.sort_values('avg_pd_click',ascending=False)

df.to_csv('data1.csv')
end = time.time()
print('time usage:' ,end-start,'s')

#hover time
df = pd.read_csv('data1.csv')
load = pd.io.parsers.read_csv('tagpid.csv')
load1 = load.dropna()
hover_count = load1[load1.tag == 'p-hover']
count = Counter(list(hover_count['data.pid']))
hover_time = []
for pid in df['item']:
    if pid in count.keys():
        hover_time.append(count.get(pid))
    else:
        hover_time.append(1)
df['hover_time'] = pd.Series(hover_time,index = df.index)

df.to_csv('data1.csv')
