#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 17:32:52 2018

@author: xinjiwang
"""

import requests
import bs4
from bs4 import BeautifulSoup

soup = BeautifulSoup('')

soup = BeautifulSoup('www.shopcade.com/product/5a7ae8da5151686753610eb7')  
print(soup)
#//*[@id="zoom_it"]

page = requests.get('http://www.shopcade.com/product/5a7ae8da5151686753610eb7')
page.content
soup = BeautifulSoup(page.content, 'html.parser')
list(soup.children)
html = list(soup.children)[1]
p = list(html.children)[3]


soup.find_all('p', class_='img')
