# -*- coding: utf-8 -*-
"""
Created on Wed Jun 07 19:13:46 2017

@author: NEU001
"""
"用python尝试提取线上、线下前5天的特征"
import pandas as pd

#goods数据
Biaotou_goods = pd.read_csv('F:/Data_Vip/data/Original_Data/goods_train_Biaotou.csv')
goods_train = pd.read_csv('F:/Data_Vip/data/Original_Data/goods_train.txt', header=None, sep='\t', names=list(Biaotou_goods))
#Online数据
Biaotou_uatest = pd.read_csv('F:/Data_Vip/data/Original_Data/user_action_test_items_hebing_Biaotou.csv')
Online = pd.read_csv('F:/Data_Vip/data/Original_Data/user_action_test_items_hebing.csv', 
                                     header=None, names=list(Biaotou_uatest),index_col = False)#加最后一项，索引就不是第一列了
#Online数据merge goods
Online_Features = pd.merge(Online,goods_train,how = 'left',on='spu_id')
del Online
#线下训练的所有数据
Biaotou_uatrain = pd.read_csv('F:/Data_Vip/data/Original_Data/user_action_train_hebing_Biaotou.csv')
user_action_train = pd.read_csv('F:/Data_Vip/data/Original_Data/user_action_train_hebing.csv', header=None, names=list(Biaotou_uatrain))
############################################ 7天交互的 #############################################
Online_7 = user_action_train[user_action_train.datas > '02-24']#改了5周
del user_action_train
##################################Online的训练集merge了goods的属性 特征###################################
Online_7 = pd.merge(Online_7,goods_train,how = 'left',on='spu_id')
del goods_train
########################################### u-s 特征 ############################################
#前7天us的点击天数
Online_action0_7 = Online_7[Online_7.action_type == 0]
group = Online_action0_7.groupby('u_spu_id')
Online_us_dianJiTianShu__7 = pd.DataFrame(group.size(),columns=['us_dianJiTianShu_7']).reset_index()
#前7天us的下单天数
Online_action1_7 = Online_7[Online_7.action_type == 1]
group = Online_action1_7.groupby('u_spu_id')
Online_us_xiaDanTianShu_7 = pd.DataFrame(group.size(),columns=['us_xiaDanTianShu_7']).reset_index()
del Online_7

########################################### u 特征 ############################################
#前7天u点击的天数
group = Online_action0_7.groupby(['uid','datas'])
Online_u_datas_7 = pd.DataFrame(group.size()).reset_index()
group = Online_u_datas_7.groupby('uid')
Online_u_dianJiTianShu_7 = pd.DataFrame(group.size(),columns=['u_dianJiTianShu_7']).reset_index()
#前7天u下单的天数
group = Online_action1_7.groupby(['uid','datas'])
Online_u_datas_7 = pd.DataFrame(group.size()).reset_index()
group = Online_u_datas_7.groupby('uid')
Online_u_xiaDanTianShu_7 = pd.DataFrame(group.size(),columns=['u_xiaDanTianShu_7']).reset_index()
del Online_u_datas_7

########################################### spu 特征 ############################################
#前7天spu被点击天数
group = Online_action0_7.groupby(['spu_id','datas'])
Online_spu_datas_7 = pd.DataFrame(group.size()).reset_index()
group = Online_spu_datas_7.groupby('spu_id')
Online_spu_beiDianJiTianShu_7 = pd.DataFrame(group.size(),columns=['spu_beiDianJiTianShu_7']).reset_index()
#前7天spu被下单天数
group = Online_action1_7.groupby(['spu_id','datas'])
Online_spu_datas_7 = pd.DataFrame(group.size()).reset_index()
group = Online_spu_datas_7.groupby('spu_id')
Online_spu_beiXiaDanTianShu_7 = pd.DataFrame(group.size(),columns=['spu_beiXiaDanTianShu_7']).reset_index()
del Online_spu_datas_7

#前7天spu被点击人数
group = Online_action0_7.groupby(['spu_id','uid'])
Online_spu_uid_7 = pd.DataFrame(group.size()).reset_index()
group = Online_spu_uid_7.groupby('spu_id')
Online_spu_beiDianJiRenShu_7 = pd.DataFrame(group.size(),columns=['spu_beiDianJiRenShu_7']).reset_index()
#前7天spu被下单人数
group = Online_action1_7.groupby(['spu_id','uid'])
Online_spu_uid_7 = pd.DataFrame(group.size()).reset_index()
group = Online_spu_uid_7.groupby('spu_id')
Online_spu_beiXiaDanRenShu_7 = pd.DataFrame(group.size(),columns=['spu_beiXiaDanRenShu_7']).reset_index()
del Online_spu_uid_7

########################################### u- cate 特征 ############################################


del Online_action0_7
del Online_action1_7
#############################把7天的特征先merge了存下################################################
Online_Features = pd.merge(Online_Features,Online_us_dianJiTianShu__7,how = 'left',on='u_spu_id')
Online_Features = pd.merge(Online_Features,Online_us_xiaDanTianShu_7,how = 'left',on='u_spu_id')

Online_Features = pd.merge(Online_Features,Online_u_dianJiTianShu_7,how = 'left',on='uid')
Online_Features = pd.merge(Online_Features,Online_u_xiaDanTianShu_7,how = 'left',on='uid')

Online_Features = pd.merge(Online_Features,Online_spu_beiDianJiTianShu_7,how = 'left',on='spu_id')
Online_Features = pd.merge(Online_Features,Online_spu_beiXiaDanTianShu_7,how = 'left',on='spu_id')
Online_Features = pd.merge(Online_Features,Online_spu_beiDianJiRenShu_7,how = 'left',on='spu_id')
Online_Features = pd.merge(Online_Features,Online_spu_beiXiaDanRenShu_7,how = 'left',on='spu_id')

del Online_us_dianJiTianShu__7
del Online_us_xiaDanTianShu_7
del Online_u_dianJiTianShu_7
del Online_u_xiaDanTianShu_7
del Online_spu_beiDianJiTianShu_7
del Online_spu_beiXiaDanTianShu_7
del Online_spu_beiDianJiRenShu_7
del Online_spu_beiXiaDanRenShu_7

##################################补缺失值  或者最后统一补 #################################
Online_Features = Online_Features.fillna(value=0)

#导出数据
Online_Features.to_csv('F:/Data_Vip/data/python_Features/Online_Features.csv', index=False)  # 不带索引