# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 10:49:27 2017

@author: NEU001
"""
import pandas as pd

#Biaotou_uatrain = pd.read_csv('F:/Data_Vip/data/Original_Data/user_action_train_hebing_Biaotou.csv')
user_action_train = pd.read_csv('F:/Data_Vip/data/Offline_Data/Sample1_user_action_train.csv')
#Sample1训练集
user_action_train = user_action_train[user_action_train.datas < '03-25']
#Sample1 训练集的uer的下单数
Sample1_train = user_action_train[['uid','action_type']]
group = Sample1_train.groupby('uid')
Sample1_train_xiadancishu = group.sum()
Sample1_train_xiadancishu = Sample1_train_xiadancishu.reset_index()
Sample1_train_xiadancishu.rename(columns={'action_type':'xiadanshu'}, inplace = True)
#没下过单的人
#Sample1_xiandan0 = Sample1_train_xiadancishu[Sample1_train_xiadancishu.action_type == 0]

#统计Sample1 训练集的user的点击天数（活跃天数）
Sample1_train = user_action_train[['uid','action_type','datas']]
del user_action_train
Sample1_train = Sample1_train[Sample1_train.action_type == 0]
group = Sample1_train.groupby(['uid','datas'])
Sample1_train = group.size()
Sample1_train = pd.DataFrame(Sample1_train).reset_index()
Sample1_train = Sample1_train[['uid','datas']]
group = Sample1_train.groupby('uid')
del Sample1_train
Sample1_train_dianjiTianShu = group.size()
Sample1_train_dianjiTianShu = pd.DataFrame(Sample1_train_dianjiTianShu,columns=['dianjitianshu']).reset_index()

#线下Sample1下单数据
Sample1_XiaDan_0325_0331 = pd.read_csv('F:/Data_Vip/data/Offline_Data/Sample1_XiaDan_0325_0331.csv')
Sample1_XiaDan_0325_0331 = Sample1_XiaDan_0325_0331[['uid','action_type']]
Sample1_XiaDan_0325_0331 = Sample1_XiaDan_0325_0331.sort_values(by='action_type', ascending=False)
Sample1_XiaDan_0325_0331 = Sample1_XiaDan_0325_0331.drop_duplicates(['uid'])

#merge 点击数
Sample1_XiaDan_0325_0331 = pd.merge(Sample1_XiaDan_0325_0331,Sample1_train_dianjiTianShu,how='left',on='uid')
del Sample1_train_dianjiTianShu
#merge 下单数
Sample1_XiaDan_0325_0331 = pd.merge(Sample1_XiaDan_0325_0331,Sample1_train_xiadancishu,how='left',on='uid')
del Sample1_train_xiadancishu

#看Sampe2只点击不买的人 的点击数和下单数
Sample1_XiaDan_a0 = Sample1_XiaDan_0325_0331[Sample1_XiaDan_0325_0331.action_type == 0]
#看Sampe2 的人 的点击数和下单数
Sample1_XiaDan_a1 = Sample1_XiaDan_0325_0331[Sample1_XiaDan_0325_0331.action_type == 1]

tmp0 = Sample1_XiaDan_a0[Sample1_XiaDan_a0.xiadanshu == 0]
tmp0 = tmp0[tmp0.dianjitianshu >50]

tmp1 = Sample1_XiaDan_a1[Sample1_XiaDan_a1.xiadanshu == 0]
tmp1 = tmp1[tmp1.dianjitianshu >50]