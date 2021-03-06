# -*- coding: utf-8 -*-
"""
Created on Wed Jun 07 19:13:46 2017

@author: NEU001
"""
"用python尝试提取线上、线下前7天的特征"
import pandas as pd

def Online_Features_i_days(i):
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
    #线下训练的所有数据
    Biaotou_uatrain = pd.read_csv('F:/Data_Vip/data/Original_Data/user_action_train_hebing_Biaotou.csv')
    user_action_train = pd.read_csv('F:/Data_Vip/data/Original_Data/user_action_train_hebing.csv', header=None, names=list(Biaotou_uatrain))
    ############################################ 7天交互的 #############################################
    if i==1:
        Online_7 = user_action_train[user_action_train.datas == '03-31']
    if i==3:
        Online_7 = user_action_train[user_action_train.datas > '03-28']
    if i==7:
        Online_7 = user_action_train[user_action_train.datas > '03-24']
    if i==14:
        Online_7 = user_action_train[user_action_train.datas > '03-17']
    if i==28:
        Online_7 = user_action_train[user_action_train.datas > '03-03']
    del user_action_train
    ###################################Online的训练集merge了goods的属性 特征###################################
    Online_7 = pd.merge(Online_7,goods_train,how = 'left',on='spu_id')
    del goods_train
    ########################################### u-s 特征 ############################################
    #前7天us的点击天数
    Online_action0_7 = Online_7[Online_7.action_type == 0]
    group = Online_action0_7.groupby('u_spu_id')
    Online_us_dianJiTianShu_7 = pd.DataFrame(group.size(),columns=['us_dianJiTianShu_'+str(i)]).reset_index()
    #前7天us的下单天数
    Online_action1_7 = Online_7[Online_7.action_type == 1]
    group = Online_action1_7.groupby('u_spu_id')
    Online_us_xiaDanTianShu_7 = pd.DataFrame(group.size(),columns=['us_xiaDanTianShu_'+str(i)]).reset_index()
    del Online_7
    
    ########################################### u 特征 ############################################
    #前7天u点击的天数
    group = Online_action0_7.groupby(['uid','datas'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby('uid')
    Online_u_dianJiTianShu_7 = pd.DataFrame(group.size(),columns=['u_dianJiTianShu_'+str(i)]).reset_index()
    #前7天u下单的天数
    group = Online_action1_7.groupby(['uid','datas'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby('uid')
    Online_u_xiaDanTianShu_7 = pd.DataFrame(group.size(),columns=['u_xiaDanTianShu_'+str(i)]).reset_index()
    
    ########################################### spu 特征 ############################################
    #前7天spu被点击天数
    group = Online_action0_7.groupby(['spu_id','datas'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby('spu_id')
    Online_spu_beiDianJiTianShu_7 = pd.DataFrame(group.size(),columns=['spu_beiDianJiTianShu_'+str(i)]).reset_index()
    #前7天spu被下单天数
    group = Online_action1_7.groupby(['spu_id','datas'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby('spu_id')
    Online_spu_beiXiaDanTianShu_7 = pd.DataFrame(group.size(),columns=['spu_beiXiaDanTianShu_'+str(i)]).reset_index()
    
    #前7天spu被点击人数
    group = Online_action0_7.groupby(['spu_id','uid'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby('spu_id')
    Online_spu_beiDianJiRenShu_7 = pd.DataFrame(group.size(),columns=['spu_beiDianJiRenShu_'+str(i)]).reset_index()
    #前7天spu被下单人数
    group = Online_action1_7.groupby(['spu_id','uid'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby('spu_id')
    Online_spu_beiXiaDanRenShu_7 = pd.DataFrame(group.size(),columns=['spu_beiXiaDanRenShu_'+str(i)]).reset_index()
    
    ########################################### u- cate 特征 ############################################
    #前7天u点击cate的天数
    group = Online_action0_7.groupby(['uid','cate_id','datas'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby(['uid','cate_id'])  #merge用
    Online_u_cate_dianJiTianShu_7 = pd.DataFrame(group.size(),columns=['u_cate_dianJiTianShu_'+str(i)]).reset_index()
    #u下单cate的天数
    group = Online_action1_7.groupby(['uid','cate_id','datas'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby(['uid','cate_id']) #merge用
    Online_u_cate_xiaDanTianShu_7 = pd.DataFrame(group.size(),columns=['u_cate_xiaDanTianShu_'+str(i)]).reset_index()
    #u对cate的点击量
    group = Online_action0_7.groupby(['uid','cate_id']) #merge用
    Online_u_cate_dianJiLiang_7 = pd.DataFrame(group.size(),columns=['u_cate_dianJiLiang_'+str(i)]).reset_index()
    #u对cate的下单量
    group = Online_action1_7.groupby(['uid','cate_id']) #merge用
    Online_u_cate_xiaDanLiang_7 = pd.DataFrame(group.size(),columns=['u_cate_xiaDanLiang_'+str(i)]).reset_index()
    
    ########################################### cate 特征 ############################################
    #cate被点击天数
    group = Online_action0_7.groupby(['cate_id','datas'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby('cate_id')
    Online_cate_beiDianJiTianShu_7 = pd.DataFrame(group.size(),columns=['cate_beiDianJiTianShu_'+str(i)]).reset_index()
    #cate 被下单天数
    group = Online_action1_7.groupby(['cate_id','datas'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby('cate_id')
    Online_cate_beiXiaDanTianShu_7 = pd.DataFrame(group.size(),columns=['cate_beiXiaDanTianShu_'+str(i)]).reset_index()
    
    #cate被点击人数
    group = Online_action0_7.groupby(['cate_id','uid'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby('cate_id')
    Online_cate_beiDianJiRenShu_7 = pd.DataFrame(group.size(),columns=['cate_beiDianJiRenShu_'+str(i)]).reset_index()
    #cate被下单人数
    group = Online_action1_7.groupby(['cate_id','uid'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby('cate_id')
    Online_cate_beiXiaDanRenShu_7 = pd.DataFrame(group.size(),columns=['cate_beiXiaDanRenShu_'+str(i)]).reset_index()
    
    ########################################### u-brand 特征 ############################################
    #u点击brand的天数
    group = Online_action0_7.groupby(['uid','brand_id','datas'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby(['uid','brand_id'])  #merge用
    Online_u_brand_dianJiTianShu_7 = pd.DataFrame(group.size(),columns=['u_brand_dianJiTianShu_'+str(i)]).reset_index()
    #u下单brand的天数
    group = Online_action1_7.groupby(['uid','brand_id','datas'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby(['uid','brand_id']) #merge用
    Online_u_brand_xiaDanTianShu_7 = pd.DataFrame(group.size(),columns=['u_brand_xiaDanTianShu_'+str(i)]).reset_index()
    #u对brand的点击量
    group = Online_action0_7.groupby(['uid','brand_id']) #merge用
    Online_u_brand_dianJiLiang_7 = pd.DataFrame(group.size(),columns=['u_brand_dianJiLiang_'+str(i)]).reset_index()
    #u对brand的下单量
    group = Online_action1_7.groupby(['uid','brand_id']) #merge用
    Online_u_brand_xiaDanLiang_7 = pd.DataFrame(group.size(),columns=['u_brand_xiaDanLiang_'+str(i)]).reset_index()
    
    ########################################### brand 特征 ############################################
    #brand被点击天数
    group = Online_action0_7.groupby(['brand_id','datas'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby('brand_id')
    Online_brand_beiDianJiTianShu_7 = pd.DataFrame(group.size(),columns=['brand_beiDianJiTianShu_'+str(i)]).reset_index()
    #brand 被下单天数
    group = Online_action1_7.groupby(['brand_id','datas'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby('brand_id')
    Online_brand_beiXiaDanTianShu_7 = pd.DataFrame(group.size(),columns=['brand_beiXiaDanTianShu_'+str(i)]).reset_index()
    
    #brand被点击人数
    group = Online_action0_7.groupby(['brand_id','uid'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby('brand_id')
    Online_brand_beiDianJiRenShu_7 = pd.DataFrame(group.size(),columns=['brand_beiDianJiRenShu_'+str(i)]).reset_index()
    #brand被下单人数
    group = Online_action1_7.groupby(['brand_id','uid'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby('brand_id')
    Online_brand_beiXiaDanRenShu_7 = pd.DataFrame(group.size(),columns=['brand_beiXiaDanRenShu_'+str(i)]).reset_index()
    
    
    
    del tmp
    del Online_action0_7
    del Online_action1_7
    #############################把7天的特征先merge了存下################################################
    # u-s 特征 
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_us_dianJiTianShu_7,how = 'left',on='u_spu_id')
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_us_xiaDanTianShu_7,how = 'left',on='u_spu_id')
    del Online_us_dianJiTianShu_7
    del Online_us_xiaDanTianShu_7
    # u 特征
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_u_dianJiTianShu_7,how = 'left',on='uid')
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_u_xiaDanTianShu_7,how = 'left',on='uid')
    del Online_u_dianJiTianShu_7
    del Online_u_xiaDanTianShu_7
    # spu 特征
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_spu_beiDianJiTianShu_7,how = 'left',on='spu_id')
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_spu_beiXiaDanTianShu_7,how = 'left',on='spu_id')
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_spu_beiDianJiRenShu_7,how = 'left',on='spu_id')
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_spu_beiXiaDanRenShu_7,how = 'left',on='spu_id')
    del Online_spu_beiDianJiTianShu_7
    del Online_spu_beiXiaDanTianShu_7
    del Online_spu_beiDianJiRenShu_7
    del Online_spu_beiXiaDanRenShu_7
    # u- cate 特征
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_u_cate_dianJiTianShu_7,how = 'left',on=['uid','cate_id'])
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_u_cate_xiaDanTianShu_7,how = 'left',on=['uid','cate_id'])
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_u_cate_dianJiLiang_7,how = 'left',on=['uid','cate_id'])
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_u_cate_xiaDanLiang_7,how = 'left',on=['uid','cate_id'])
    del Online_u_cate_dianJiTianShu_7
    del Online_u_cate_xiaDanTianShu_7
    del Online_u_cate_dianJiLiang_7
    del Online_u_cate_xiaDanLiang_7
    ## cate 特征
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_cate_beiDianJiTianShu_7,how = 'left',on='cate_id')
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_cate_beiXiaDanTianShu_7,how = 'left',on='cate_id')
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_cate_beiDianJiRenShu_7,how = 'left',on='cate_id')
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_cate_beiXiaDanRenShu_7,how = 'left',on='cate_id')
    del Online_cate_beiDianJiTianShu_7
    del Online_cate_beiXiaDanTianShu_7
    del Online_cate_beiDianJiRenShu_7
    del Online_cate_beiXiaDanRenShu_7
    ## u-brand 特征
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_u_brand_dianJiTianShu_7,how = 'left',on=['uid','brand_id'])
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_u_brand_xiaDanTianShu_7,how = 'left',on=['uid','brand_id'])
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_u_brand_dianJiLiang_7,how = 'left',on=['uid','brand_id'])
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_u_brand_xiaDanLiang_7,how = 'left',on=['uid','brand_id'])
    del Online_u_brand_dianJiTianShu_7
    del Online_u_brand_xiaDanTianShu_7
    del Online_u_brand_dianJiLiang_7
    del Online_u_brand_xiaDanLiang_7
    # brand 特征 
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_brand_beiDianJiTianShu_7,how = 'left',on='brand_id')
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_brand_beiXiaDanTianShu_7,how = 'left',on='brand_id')
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_brand_beiDianJiRenShu_7,how = 'left',on='brand_id')
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_brand_beiXiaDanRenShu_7,how = 'left',on='brand_id')
    del Online_brand_beiDianJiTianShu_7
    del Online_brand_beiXiaDanTianShu_7
    del Online_brand_beiDianJiRenShu_7
    del Online_brand_beiXiaDanRenShu_7
    
    ##################################补缺失值  或者最后统一补 #################################
    Online_Features_XiaDan = Online_Features_XiaDan.fillna(value=0)
    #前3列没用了
    x_columns = [x for x in Online_Features_XiaDan.columns if x not in ['uid', 'spu_id','brand_id','cate_id']]
    Online_Features_XiaDan = Online_Features_XiaDan[x_columns]
    
    #导出数据
    Online_Features_XiaDan.to_csv('F:/Data_Vip/data/python_Features/Online_Features_'+str(i)+'.csv', index=False)  # 不带索引