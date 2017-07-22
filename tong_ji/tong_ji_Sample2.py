# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 10:49:27 2017

@author: NEU001
"""
import pandas as pd

#Biaotou_uatrain = pd.read_csv('F:/Data_Vip/data/Original_Data/user_action_train_hebing_Biaotou.csv')
user_action_train = pd.read_csv('F:/Data_Vip/data/Offline_Data/Sample2_user_action_train.csv')
#Sample2训练集
user_action_train = user_action_train[user_action_train.datas < '03-18']
#######################################################################################################
#Sample2 训练集的uer的下单数
Sample2_train = user_action_train[['uid','action_type']]
group = Sample2_train.groupby('uid')
Sample2_train_xiadancishu = group.sum()
Sample2_train_xiadancishu = Sample2_train_xiadancishu.reset_index()
Sample2_train_xiadancishu.rename(columns={'action_type':'xiadanshu'}, inplace = True)
#没下过单的人
#Sample2_xiandan0 = Sample2_train_xiadancishu[Sample2_train_xiadancishu.action_type == 0]

#统计Sample2 训练集的user的点击天数（活跃天数）
Sample2_train = user_action_train[['uid','action_type','datas']]
Sample2_train = Sample2_train[Sample2_train.action_type == 0]
group = Sample2_train.groupby(['uid','datas'])
Sample2_train = group.size()
Sample2_train = pd.DataFrame(Sample2_train).reset_index()
Sample2_train = Sample2_train[['uid','datas']]
group = Sample2_train.groupby('uid')
del Sample2_train
Sample2_train_dianjiTianShu = group.size()
Sample2_train_dianjiTianShu = pd.DataFrame(Sample2_train_dianjiTianShu,columns=['dianjitianshu']).reset_index()

#点击量
tmp = user_action_train[['uid','action_type']]
tmp = tmp[tmp.action_type == 0]
del user_action_train
group = tmp.groupby('uid')
Sample2_dianJiLiang = group.size()
Sample2_dianJiLiang = pd.DataFrame(Sample2_dianJiLiang,columns=['dianJiLiang']).reset_index()

#平均每天的点击量
Sample2_dianJiLiangPerDay = Sample2_dianJiLiang['dianJiLiang']/Sample2_train_dianjiTianShu['dianjitianshu']
Sample2_dianJiLiangPerDay = pd.DataFrame(Sample2_dianJiLiangPerDay,columns=['dianJiLiangPerDay'])
Sample2_dianJiLiangPerDay = pd.concat([Sample2_train_dianjiTianShu['uid'],Sample2_dianJiLiangPerDay['dianJiLiangPerDay']],axis=1)

################################加载处理线下Sample2下单数据##############################################
Sample2_XiaDan_0318_0324 = pd.read_csv('F:/Data_Vip/data/Offline_Data/Sample2_XiaDan_0318_0324.csv')
Sample2_XiaDan_0318_0324 = Sample2_XiaDan_0318_0324[['uid','action_type']]
Sample2_XiaDan_0318_0324 = Sample2_XiaDan_0318_0324.sort_values(by='action_type', ascending=False)
Sample2_XiaDan_0318_0324 = Sample2_XiaDan_0318_0324.drop_duplicates(['uid'])

#######################################  merge  #########################################################
#merge 下单数
Sample2_XiaDan_0318_0324 = pd.merge(Sample2_XiaDan_0318_0324,Sample2_train_xiadancishu,how='left',on='uid')
del Sample2_train_xiadancishu
#merge 点击天数
Sample2_XiaDan_0318_0324 = pd.merge(Sample2_XiaDan_0318_0324,Sample2_train_dianjiTianShu,how='left',on='uid')
del Sample2_train_dianjiTianShu
#merge 点击量
Sample2_XiaDan_0318_0324 = pd.merge(Sample2_XiaDan_0318_0324, Sample2_dianJiLiang,how='left',on='uid')
del Sample2_dianJiLiang
#merge 平均点击量
Sample2_XiaDan_0318_0324 = pd.merge(Sample2_XiaDan_0318_0324, Sample2_dianJiLiangPerDay,how='left',on='uid')

#看Sampe2只点击不买的人 的点击数和下单数
Sample2_XiaDan_a0 = Sample2_XiaDan_0318_0324[Sample2_XiaDan_0318_0324.action_type == 0]
#看Sampe2 的人 的点击数和下单数
Sample2_XiaDan_a1 = Sample2_XiaDan_0318_0324[Sample2_XiaDan_0318_0324.action_type == 1]
'''
tmp0 = Sample2_XiaDan_a0[Sample2_XiaDan_a0.xiadanshu == 0]
tmp0 = tmp0[tmp0.dianjitianshu >50]

tmp1 = Sample2_XiaDan_a1[Sample2_XiaDan_a1.xiadanshu == 0]
tmp1 = tmp1[tmp1.dianjitianshu >50]
'''