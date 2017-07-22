# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 08:55:02 2017

@author: NEU001
"""
import pandas as pd

#goods数据
Biaotou_goods = pd.read_csv('F:/Data_Vip/data/Original_Data/goods_train_Biaotou.csv')
goods_train = pd.read_csv('F:/Data_Vip/data/Original_Data/goods_train.txt', header=None, sep='\t', names=list(Biaotou_goods))
#Online数据
Biaotou_uatest = pd.read_csv('F:/Data_Vip/data/Original_Data/user_action_test_items_hebing_Biaotou.csv')
Online = pd.read_csv('F:/Data_Vip/data/Original_Data/user_action_test_items_hebing.csv', 
                                     header=None, names=list(Biaotou_uatest),index_col = False)#加最后一项，索引就不是第一列了
#Online数据merge goods
Online_Features_XiaDan = pd.merge(Online,goods_train,how = 'left',on='spu_id')
del Online
del goods_train
#去掉两列，占空间，没用了
x_columns = [x for x in Online_Features_XiaDan.columns if x not in ['uid', 'spu_id']]
Online_Features_XiaDan = Online_Features_XiaDan[x_columns]

########################################## 加载1 3 7 14 28的数据 并merge   ########################################
#Online_Features_1 = pd.read_csv('F:/Data_Vip/data/python_Features/Online_Features_1.csv')
'''
Online_Features_3 = pd.read_csv('F:/Data_Vip/data/python_Features/Online_Features_3.csv')
Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_Features_3,how = 'left',on='u_spu_id')
del Online_Features_3

#Online_Features_7 = pd.read_csv('F:/Data_Vip/data/python_Features/Online_Features_7.csv')
Online_Features_14 = pd.read_csv('F:/Data_Vip/data/python_Features/Online_Features_14.csv')
Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_Features_14,how = 'left',on='u_spu_id')
del Online_Features_14

Online_Features_28 = pd.read_csv('F:/Data_Vip/data/python_Features/Online_Features_28.csv')
Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_Features_28,how = 'left',on='u_spu_id')
del Online_Features_28
'''
Online_Features_100 = pd.read_csv('F:/Data_Vip/data/python_Features/Online_Features_100.csv')
Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_Features_100,how = 'left',on='u_spu_id')
del Online_Features_100

#去掉一列，占空间，没用了
x_columns = [x for x in Online_Features_XiaDan.columns if x not in ['u_spu_id']]
Online_Features_XiaDan = Online_Features_XiaDan[x_columns]
Online_Features_XiaDan.to_csv('F:/Data_Vip/data/python_Features/Online_Features.csv', index=False)
