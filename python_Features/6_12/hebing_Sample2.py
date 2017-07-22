# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 08:55:02 2017

@author: NEU001
"""
import pandas as pd

#goods数据
Biaotou_goods = pd.read_csv('F:/Data_Vip/data/Original_Data/goods_train_Biaotou.csv')
goods_train = pd.read_csv('F:/Data_Vip/data/Original_Data/goods_train.txt', header=None, sep='\t', names=list(Biaotou_goods))
#线下Sample2下单数据
Sample2_XiaDan_0318_0324 = pd.read_csv('F:/Data_Vip/data/Offline_Data/Sample2_XiaDan_0318_0324.csv')
#Sample2下单数据merge goods
Sample2_Features_XiaDan = pd.merge(Sample2_XiaDan_0318_0324,goods_train,how = 'left',on='spu_id')
del Sample2_XiaDan_0318_0324
del goods_train
#去掉两列，占空间，没用了
x_columns = [x for x in Sample2_Features_XiaDan.columns if x not in ['uid', 'spu_id']]
Sample2_Features_XiaDan = Sample2_Features_XiaDan[x_columns]

########################################## 加载1 3 7 14 28的数据 并merge   ########################################
#Sample2_Features_1 = pd.read_csv('F:/Data_Vip/data/python_Features/Sample2_Features_1.csv')
'''
Sample2_Features_3 = pd.read_csv('F:/Data_Vip/data/python_Features/Sample2_Features_3.csv')
Sample2_Features_XiaDan = pd.merge(Sample2_Features_XiaDan,Sample2_Features_3,how = 'left',on='u_spu_id')
del Sample2_Features_3

#Sample2_Features_7 = pd.read_csv('F:/Data_Vip/data/python_Features/Sample2_Features_7.csv')
Sample2_Features_14 = pd.read_csv('F:/Data_Vip/data/python_Features/Sample2_Features_14.csv')
Sample2_Features_XiaDan = pd.merge(Sample2_Features_XiaDan,Sample2_Features_14,how = 'left',on='u_spu_id')
del Sample2_Features_14

Sample2_Features_28 = pd.read_csv('F:/Data_Vip/data/python_Features/Sample2_Features_28.csv')
Sample2_Features_XiaDan = pd.merge(Sample2_Features_XiaDan,Sample2_Features_28,how = 'left',on='u_spu_id')
del Sample2_Features_28
'''
Sample2_Features_100 = pd.read_csv('F:/Data_Vip/data/python_Features/Sample2_Features_100.csv')
Sample2_Features_XiaDan = pd.merge(Sample2_Features_XiaDan,Sample2_Features_100,how = 'left',on='u_spu_id')
del Sample2_Features_100

#去掉一列，占空间，没用了
x_columns = [x for x in Sample2_Features_XiaDan.columns if x not in ['u_spu_id']]
Sample2_Features_XiaDan = Sample2_Features_XiaDan[x_columns]
Sample2_Features_XiaDan.to_csv('F:/Data_Vip/data/python_Features/Sample2_Features.csv', index=False)

