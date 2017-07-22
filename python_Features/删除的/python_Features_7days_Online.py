# -*- coding: utf-8 -*-
"""
Created on Wed Jun 07 19:13:46 2017

@author: NEU001
"""
"用python尝试提取线上、线下前7天的特征"
import pandas as pd

#goods数据
Biaotou_goods = pd.read_csv('F:/Data_Vip/data/Original_Data/goods_train_Biaotou.csv')
goods_train = pd.read_csv('F:/Data_Vip/data/Original_Data/goods_train.txt', header=None, sep='\t', names=list(Biaotou_goods))
#线下Sample2下单数据
Sample2_XiaDan_0318_0324 = pd.read_csv('F:/Data_Vip/data/Offline_XiaDan/Offline_Sample2_XiaDan_0318_0324.csv')
#Sample2下单数据merge goods
Sample2_XiaDan_0318_0324 = pd.merge(Sample2_XiaDan_0318_0324,goods_train,how = 'left',on='spu_id')
#线下训练的所有数据
Biaotou_uatrain = pd.read_csv('F:/Data_Vip/data/Original_Data/user_action_train_hebing_Biaotou.csv')
user_action_train = pd.read_csv('F:/Data_Vip/data/Original_Data/user_action_train_hebing.csv', header=None, names=list(Biaotou_uatrain))
############################################ 7天交互的 #############################################
Sample2_7 = user_action_train[user_action_train.datas > '03-10']#改了5周'02-10'
Sample2_7 = Sample2_7[Sample2_7.datas < '03-18']
del user_action_train
###################################Sample2的训练集merge了goods的属性 特征###################################
Sample2_7 = pd.merge(Sample2_7,goods_train,how = 'left',on='spu_id')
del goods_train
########################################### u-s 特征 ############################################
#前7天us的点击天数
Sample2_action0_7 = Sample2_7[Sample2_7.action_type == 0]
group = Sample2_action0_7.groupby('u_spu_id')
Sample2_us_dianJiTianShu_7 = pd.DataFrame(group.size(),columns=['us_dianJiTianShu_7']).reset_index()
#前7天us的下单天数
Sample2_action1_7 = Sample2_7[Sample2_7.action_type == 1]
group = Sample2_action1_7.groupby('u_spu_id')
Sample2_us_xiaDanTianShu_7 = pd.DataFrame(group.size(),columns=['us_xiaDanTianShu_7']).reset_index()

########################################### u 特征 ############################################
#前7天u点击的天数
group = Sample2_action0_7.groupby(['uid','datas'])
tmp = pd.DataFrame(group.size()).reset_index()
group = tmp.groupby('uid')
Sample2_u_dianJiTianShu_7 = pd.DataFrame(group.size(),columns=['u_dianJiTianShu_7']).reset_index()
#前7天u下单的天数
group = Sample2_action1_7.groupby(['uid','datas'])
tmp = pd.DataFrame(group.size()).reset_index()
group = tmp.groupby('uid')
Sample2_u_xiaDanTianShu_7 = pd.DataFrame(group.size(),columns=['u_xiaDanTianShu_7']).reset_index()

########################################### spu 特征 ############################################
#前7天spu被点击天数
group = Sample2_action0_7.groupby(['spu_id','datas'])
tmp = pd.DataFrame(group.size()).reset_index()
group = tmp.groupby('spu_id')
Sample2_spu_beiDianJiTianShu_7 = pd.DataFrame(group.size(),columns=['spu_beiDianJiTianShu_7']).reset_index()
#前7天spu被下单天数
group = Sample2_action1_7.groupby(['spu_id','datas'])
tmp = pd.DataFrame(group.size()).reset_index()
group = tmp.groupby('spu_id')
Sample2_spu_beiXiaDanTianShu_7 = pd.DataFrame(group.size(),columns=['spu_beiXiaDanTianShu_7']).reset_index()

#前7天spu被点击人数
group = Sample2_action0_7.groupby(['spu_id','uid'])
tmp = pd.DataFrame(group.size()).reset_index()
group = tmp.groupby('spu_id')
Sample2_spu_beiDianJiRenShu_7 = pd.DataFrame(group.size(),columns=['spu_beiDianJiRenShu_7']).reset_index()
#前7天spu被下单人数
group = Sample2_action1_7.groupby(['spu_id','uid'])
tmp = pd.DataFrame(group.size()).reset_index()
group = tmp.groupby('spu_id')
Sample2_spu_beiXiaDanRenShu_7 = pd.DataFrame(group.size(),columns=['spu_beiXiaDanRenShu_7']).reset_index()

########################################### u- cate 特征 ############################################
#前7天u点击cate的天数
group = Sample2_action0_7.groupby(['uid','cate_id','datas'])
tmp = pd.DataFrame(group.size()).reset_index()
group = tmp.groupby(['uid','cate_id'])  #merge用
Sample2_u_cate_dianJiTianShu_7 = pd.DataFrame(group.size(),columns=['u_cate_dianJiTianShu_7']).reset_index()
#u下单cate的天数
group = Sample2_action1_7.groupby(['uid','cate_id','datas'])
tmp = pd.DataFrame(group.size()).reset_index()
group = tmp.groupby(['uid','cate_id']) #merge用
Sample2_u_cate_xiaDanTianShu_7 = pd.DataFrame(group.size(),columns=['u_cate_xiaDanTianShu_7']).reset_index()
#u对cate的点击量
group = Sample2_action0_7.groupby(['uid','cate_id']) #merge用
Sample2_u_cate_dianJiLiang_7 = pd.DataFrame(group.size(),columns=['u_cate_dianJiLiang_7']).reset_index()
#u对cate的下单量
group = Sample2_action1_7.groupby(['uid','cate_id']) #merge用
Sample2_u_cate_xiaDanLiang_7 = pd.DataFrame(group.size(),columns=['u_cate_xiaDanLiang_7']).reset_index()

########################################### cate 特征 ############################################
#cate被点击天数
group = Sample2_action0_7.groupby(['cate_id','datas'])
tmp = pd.DataFrame(group.size()).reset_index()
group = tmp.groupby('cate_id')
Sample2_cate_beiDianJiTianShu_7 = pd.DataFrame(group.size(),columns=['cate_beiDianJiTianShu_7']).reset_index()
#cate 被下单天数
group = Sample2_action1_7.groupby(['cate_id','datas'])
tmp = pd.DataFrame(group.size()).reset_index()
group = tmp.groupby('cate_id')
Sample2_cate_beiXiaDanTianShu_7 = pd.DataFrame(group.size(),columns=['cate_beiXiaDanTianShu_7']).reset_index()

#cate被点击人数
group = Sample2_action0_7.groupby(['cate_id','uid'])
tmp = pd.DataFrame(group.size()).reset_index()
group = tmp.groupby('cate_id')
Sample2_cate_beiDianJiRenShu_7 = pd.DataFrame(group.size(),columns=['cate_beiDianJiRenShu_7']).reset_index()
#cate被下单人数
group = Sample2_action1_7.groupby(['cate_id','uid'])
tmp = pd.DataFrame(group.size()).reset_index()
group = tmp.groupby('cate_id')
Sample2_cate_beiXiaDanRenShu_7 = pd.DataFrame(group.size(),columns=['cate_beiXiaDanRenShu_7']).reset_index()

########################################### u-brand 特征 ############################################
#u点击brand的天数
group = Sample2_action0_7.groupby(['uid','brand_id','datas'])
tmp = pd.DataFrame(group.size()).reset_index()
group = tmp.groupby(['uid','brand_id'])  #merge用
Sample2_u_brand_dianJiTianShu_7 = pd.DataFrame(group.size(),columns=['u_brand_dianJiTianShu_7']).reset_index()
#u下单brand的天数
group = Sample2_action1_7.groupby(['uid','brand_id','datas'])
tmp = pd.DataFrame(group.size()).reset_index()
group = tmp.groupby(['uid','brand_id']) #merge用
Sample2_u_brand_xiaDanTianShu_7 = pd.DataFrame(group.size(),columns=['u_brand_xiaDanTianShu_7']).reset_index()
#u对brand的点击量
group = Sample2_action0_7.groupby(['uid','brand_id']) #merge用
Sample2_u_brand_dianJiLiang_7 = pd.DataFrame(group.size(),columns=['u_brand_dianJiLiang_7']).reset_index()
#u对brand的下单量
group = Sample2_action1_7.groupby(['uid','brand_id']) #merge用
Sample2_u_brand_xiaDanLiang_7 = pd.DataFrame(group.size(),columns=['u_brand_xiaDanLiang_7']).reset_index()

########################################### brand 特征 ############################################
#brand被点击天数
group = Sample2_action0_7.groupby(['brand_id','datas'])
tmp = pd.DataFrame(group.size()).reset_index()
group = tmp.groupby('brand_id')
Sample2_brand_beiDianJiTianShu_7 = pd.DataFrame(group.size(),columns=['brand_beiDianJiTianShu_7']).reset_index()
#brand 被下单天数
group = Sample2_action1_7.groupby(['brand_id','datas'])
tmp = pd.DataFrame(group.size()).reset_index()
group = tmp.groupby('brand_id')
Sample2_brand_beiXiaDanTianShu_7 = pd.DataFrame(group.size(),columns=['brand_beiXiaDanTianShu_7']).reset_index()

#brand被点击人数
group = Sample2_action0_7.groupby(['brand_id','uid'])
tmp = pd.DataFrame(group.size()).reset_index()
group = tmp.groupby('brand_id')
Sample2_brand_beiDianJiRenShu_7 = pd.DataFrame(group.size(),columns=['brand_beiDianJiRenShu_7']).reset_index()
#brand被下单人数
group = Sample2_action1_7.groupby(['brand_id','uid'])
tmp = pd.DataFrame(group.size()).reset_index()
group = tmp.groupby('brand_id')
Sample2_brand_beiXiaDanRenShu_7 = pd.DataFrame(group.size(),columns=['brand_beiXiaDanRenShu_7']).reset_index()




del Sample2_action0_7
del Sample2_action1_7
#############################把7天的特征先merge了存下################################################
Sample2_feture_7 = Sample2_7[['u_spu_id','uid','spu_id','cate_id','brand_id']]
del Sample2_7
Sample2_feture_7 = Sample2_feture_7.drop_duplicates(['u_spu_id']) #去重，有重复的u-s
# u-s 特征 
Sample2_feture_7 = pd.merge(Sample2_feture_7,Sample2_us_dianJiTianShu_7,how = 'left',on='u_spu_id')
Sample2_feture_7 = pd.merge(Sample2_feture_7,Sample2_us_xiaDanTianShu_7,how = 'left',on='u_spu_id')
del Sample2_us_dianJiTianShu_7
del Sample2_us_xiaDanTianShu_7
# u 特征
Sample2_feture_7 = pd.merge(Sample2_feture_7,Sample2_u_dianJiTianShu_7,how = 'left',on='uid')
Sample2_feture_7 = pd.merge(Sample2_feture_7,Sample2_u_xiaDanTianShu_7,how = 'left',on='uid')
del Sample2_u_dianJiTianShu_7
del Sample2_u_xiaDanTianShu_7
# spu 特征
Sample2_feture_7 = pd.merge(Sample2_feture_7,Sample2_spu_beiDianJiTianShu_7,how = 'left',on='spu_id')
Sample2_feture_7 = pd.merge(Sample2_feture_7,Sample2_spu_beiXiaDanTianShu_7,how = 'left',on='spu_id')
Sample2_feture_7 = pd.merge(Sample2_feture_7,Sample2_spu_beiDianJiRenShu_7,how = 'left',on='spu_id')
Sample2_feture_7 = pd.merge(Sample2_feture_7,Sample2_spu_beiXiaDanRenShu_7,how = 'left',on='spu_id')
del Sample2_spu_beiDianJiTianShu_7
del Sample2_spu_beiXiaDanTianShu_7
del Sample2_spu_beiDianJiRenShu_7
del Sample2_spu_beiXiaDanRenShu_7
# u- cate 特征
Sample2_feture_7 = pd.merge(Sample2_feture_7,Sample2_u_cate_dianJiTianShu_7,how = 'left',on=['uid','cate_id'])
Sample2_feture_7 = pd.merge(Sample2_feture_7,Sample2_u_cate_xiaDanTianShu_7,how = 'left',on=['uid','cate_id'])
Sample2_feture_7 = pd.merge(Sample2_feture_7,Sample2_u_cate_dianJiLiang_7,how = 'left',on=['uid','cate_id'])
Sample2_feture_7 = pd.merge(Sample2_feture_7,Sample2_u_cate_xiaDanLiang_7,how = 'left',on=['uid','cate_id'])
del Sample2_u_cate_dianJiTianShu_7
del Sample2_u_cate_xiaDanTianShu_7
del Sample2_u_cate_dianJiLiang_7
del Sample2_u_cate_xiaDanLiang_7
## cate 特征
Sample2_feture_7 = pd.merge(Sample2_feture_7,Sample2_cate_beiDianJiTianShu_7,how = 'left',on='cate_id')
Sample2_feture_7 = pd.merge(Sample2_feture_7,Sample2_cate_beiXiaDanTianShu_7,how = 'left',on='cate_id')
Sample2_feture_7 = pd.merge(Sample2_feture_7,Sample2_cate_beiDianJiRenShu_7,how = 'left',on='cate_id')
Sample2_feture_7 = pd.merge(Sample2_feture_7,Sample2_cate_beiXiaDanRenShu_7,how = 'left',on='cate_id')
del Sample2_cate_beiDianJiTianShu_7
del Sample2_cate_beiXiaDanTianShu_7
del Sample2_cate_beiDianJiRenShu_7
del Sample2_cate_beiXiaDanRenShu_7
## u-brand 特征
Sample2_feture_7 = pd.merge(Sample2_feture_7,Sample2_u_brand_dianJiTianShu_7,how = 'left',on=['uid','brand_id'])
Sample2_feture_7 = pd.merge(Sample2_feture_7,Sample2_u_brand_xiaDanTianShu_7,how = 'left',on=['uid','brand_id'])
Sample2_feture_7 = pd.merge(Sample2_feture_7,Sample2_u_brand_dianJiLiang_7,how = 'left',on=['uid','brand_id'])
Sample2_feture_7 = pd.merge(Sample2_feture_7,Sample2_u_brand_xiaDanLiang_7,how = 'left',on=['uid','brand_id'])
del Sample2_u_brand_dianJiTianShu_7
del Sample2_u_brand_xiaDanTianShu_7
del Sample2_u_brand_dianJiLiang_7
del Sample2_u_brand_xiaDanLiang_7
# brand 特征 
Sample2_feture_7 = pd.merge(Sample2_feture_7,Sample2_brand_beiDianJiTianShu_7,how = 'left',on='brand_id')
Sample2_feture_7 = pd.merge(Sample2_feture_7,Sample2_brand_beiXiaDanTianShu_7,how = 'left',on='brand_id')
Sample2_feture_7 = pd.merge(Sample2_feture_7,Sample2_brand_beiDianJiRenShu_7,how = 'left',on='brand_id')
Sample2_feture_7 = pd.merge(Sample2_feture_7,Sample2_brand_beiXiaDanRenShu_7,how = 'left',on='brand_id')
del Sample2_brand_beiDianJiTianShu_7
del Sample2_brand_beiXiaDanTianShu_7
del Sample2_brand_beiDianJiRenShu_7
del Sample2_brand_beiXiaDanRenShu_7

################################# 把下单数据和训练数据统计的特征merge #####################################   
Sample2_feture_7_columns = [x for x in Sample2_feture_7.columns if x not in ['uid', 'spu_id','brand_id','cate_id']]
Sample2_feture_7 = Sample2_feture_7[Sample2_feture_7_columns] 
Sample2_Features_XiaDan_0318_0324 = pd.merge(Sample2_XiaDan_0318_0324,Sample2_feture_7,how = 'left',on='u_spu_id')
del Sample2_feture_7
del Sample2_XiaDan_0318_0324
##################################补缺失值  或者最后统一补 #################################
Sample2_Features_XiaDan_0318_0324 = Sample2_Features_XiaDan_0318_0324.fillna(value=0)

#导出数据
Sample2_Features_XiaDan_0318_0324.to_csv('F:/Data_Vip/data/python_Features/Sample2_Features_XiaDan_0318_0324.csv', index=False)  # 不带索引