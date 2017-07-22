# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 08:55:02 2017

@author: NEU001
"""
import pandas as pd

#goods数据
Biaotou_goods = pd.read_csv('F:/Data_Vip/data/Original_Data/goods_train_Biaotou.csv')
goods_train = pd.read_csv('F:/Data_Vip/data/Original_Data/goods_train.txt', header=None, sep='\t', names=list(Biaotou_goods))
#线下Sample1下单数据
Sample1_XiaDan_0325_0331 = pd.read_csv('F:/Data_Vip/data/Offline_XiaDan/Offline_Sample1_XiaDan_0325_0331.csv')
#Sample1下单数据merge goods
Sample1_Features_XiaDan = pd.merge(Sample1_XiaDan_0325_0331,goods_train,how = 'left',on='spu_id')
del Sample1_XiaDan_0325_0331
del goods_train
#去掉两列，占空间，没用了
x_columns = [x for x in Sample1_Features_XiaDan.columns if x not in ['uid', 'spu_id']]
Sample1_Features_XiaDan = Sample1_Features_XiaDan[x_columns]

########################################## 加载1 3 7 14 28的数据 并merge   ########################################
#Sample1_Features_1 = pd.read_csv('F:/Data_Vip/data/python_Features/Sample1_Features_1.csv')

Sample1_Features_3 = pd.read_csv('F:/Data_Vip/data/python_Features/Sample1_Features_3.csv')
Sample1_Features_XiaDan = pd.merge(Sample1_Features_XiaDan,Sample1_Features_3,how = 'left',on='u_spu_id')
del Sample1_Features_3

#Sample1_Features_7 = pd.read_csv('F:/Data_Vip/data/python_Features/Sample1_Features_7.csv')
Sample1_Features_14 = pd.read_csv('F:/Data_Vip/data/python_Features/Sample1_Features_14.csv')
Sample1_Features_XiaDan = pd.merge(Sample1_Features_XiaDan,Sample1_Features_14,how = 'left',on='u_spu_id')
del Sample1_Features_14

Sample1_Features_28 = pd.read_csv('F:/Data_Vip/data/python_Features/Sample1_Features_28.csv')
Sample1_Features_XiaDan = pd.merge(Sample1_Features_XiaDan,Sample1_Features_28,how = 'left',on='u_spu_id')
del Sample1_Features_28

#去掉一列，占空间，没用了
x_columns = [x for x in Sample1_Features_XiaDan.columns if x not in ['u_spu_id']]
Sample1_Features_XiaDan = Sample1_Features_XiaDan[x_columns]
Sample1_Features_XiaDan.to_csv('F:/Data_Vip/data/python_Features/Sample1_Features.csv', index=False)
