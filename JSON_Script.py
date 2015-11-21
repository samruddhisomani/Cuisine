# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 01:45:31 2015

@author: Bejan Sadeghian
Learning JSON
"""

#import json
import pandas as pd

rawdata = pd.read_json(r'C:\Users\beins_000\Documents\GitHub\APM_Food_Project\train_json\train.json')

rawdata['cuisine'].value_counts()
rawdata.to_csv('Test.csv')