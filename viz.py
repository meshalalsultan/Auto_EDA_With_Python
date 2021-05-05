import pandas as pd 
import numpy as np 
import sweetviz
from sklearn.model_selection import train_test_split
'''
data = pd.read_csv('stock_data.csv')
print(data.head())

df=pd.read_csv('Groceries_dataset.csv')
print(df.head())
print(df.shape)
train = df.iloc[ :15000 , :]
test = df.iloc[ : , 15000:]

print(train.shape)
print(test.shape)

my_report = sweetviz.analyze([df,'df'],target_feat='Member_number')
my_report.show_html('Groceries_dataset_vis.html')
'''
heart = pd.read_csv('heart-disease.csv')

#print(heart.head())
#print(heart.shape)

my_report = sweetviz.analyze([heart,'heart'],target_feat='target')
my_report.show_html('heart-disease_vis.html')