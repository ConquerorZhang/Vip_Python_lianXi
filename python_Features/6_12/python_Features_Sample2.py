# -*- coding: utf-8 -*-
"""
Created on Wed Jun 07 19:13:46 2017

@author: NEU001
"""
"用python尝试提取线上、线下前7天的特征"
import pandas as pd

def Sample2_Features_i_days(i):
    #goods数据
    Biaotou_goods = pd.read_csv('F:/Data_Vip/data/Original_Data/goods_train_Biaotou.csv')
    goods_train = pd.read_csv('F:/Data_Vip/data/Original_Data/goods_train.txt', header=None, sep='\t', names=list(Biaotou_goods))
    #线下Sample2下单数据
    Sample2_XiaDan_0318_0324 = pd.read_csv('F:/Data_Vip/data/Offline_Data/Sample2_XiaDan_0318_0324.csv')
    #Sample2下单数据merge goods
    Sample2_Features_XiaDan = pd.merge(Sample2_XiaDan_0318_0324,goods_train,how = 'left',on='spu_id')
    del Sample2_XiaDan_0318_0324
    #线下训练的所有数据
    user_action_train = pd.read_csv('F:/Data_Vip/data/Offline_Data/Sample2_user_action_train.csv')
    ############################################ 7天交互的 #############################################
    if i==1:
        Sample2 = user_action_train[user_action_train.datas == '03-17']
    if i==3:
        Sample2 = user_action_train[user_action_train.datas > '03-14']
    if i==7:
        Sample2 = user_action_train[user_action_train.datas > '03-10']
    if i==14:
        Sample2 = user_action_train[user_action_train.datas > '03-03']
    if i==28:
        Sample2 = user_action_train[user_action_train.datas > '02-17']
    if i == 35:
        Sample2 = user_action_train[user_action_train.datas > '02-10']
    if i == 100:
        Sample2 = user_action_train
    del user_action_train
    ###################################Sample2的训练集merge了goods的属性 特征###################################
    Sample2 = pd.merge(Sample2,goods_train,how = 'left',on='spu_id')
    del goods_train
    ########################################### u-s 特征 ############################################
    #us的点击天数(us)
    Sample2_action0 = Sample2[Sample2.action_type == 0]
    group = Sample2_action0.groupby('u_spu_id')
    Sample2_us_dianJiTianShu = pd.DataFrame(group.size(),columns=['us_dianJiTianShu_'+str(i)]).reset_index()
    #us的下单天数(us)
    Sample2_action1 = Sample2[Sample2.action_type == 1]
    group = Sample2_action1.groupby('u_spu_id')
    Sample2_us_xiaDanTianShu = pd.DataFrame(group.size(),columns=['us_xiaDanTianShu_'+str(i)]).reset_index()
    #del Sample2
    
    ########################################### u 特征 ############################################
    #u出现的天数(u)
    group = Sample2.groupby(['uid','datas'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby('uid')
    Sample2_u_chuXianTianShu = pd.DataFrame(group.size(),columns=['u_chuXianTianShu_'+str(i)]).reset_index()
    
    #u点击的天数(u)
    group = Sample2_action0.groupby(['uid','datas'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby('uid')
    Sample2_u_dianJiTianShu = pd.DataFrame(group.size(),columns=['u_dianJiTianShu_'+str(i)]).reset_index()
    #u点击的不重复的spu个数(u)
    group = Sample2_action0.groupby(['uid','spu_id'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby('uid')
    Sample2_u_dianJiSpuGeShu = pd.DataFrame(group.size(),columns=['u_dianJiSpuGeShu_'+str(i)]).reset_index()
    #u点击量(u)
    group = Sample2_action0.groupby(['uid'])
    Sample2_u_dianJiLiang = pd.DataFrame(group.size(),columns=['u_dianJiLiang_'+str(i)]).reset_index()
    #u在点击过的天数里平均点击量(u)
    Sample2_u_pingJunDianJiLiang = Sample2_u_dianJiLiang['u_dianJiLiang_'+str(i)]/Sample2_u_dianJiTianShu['u_dianJiTianShu_'+str(i)]
    Sample2_u_pingJunDianJiLiang = pd.DataFrame(Sample2_u_pingJunDianJiLiang, columns=['u_pingJunDianJiLiang_'+str(i)])
    Sample2_u_pingJunDianJiLiang = pd.concat([Sample2_u_dianJiTianShu['uid'],Sample2_u_pingJunDianJiLiang['u_pingJunDianJiLiang_'+str(i)]], axis=1)
    #u对spu的平均点击天数（次数）(u)
    Sample2_u_pingJunDianJiSpu = Sample2_u_dianJiLiang['u_dianJiLiang_'+str(i)]/Sample2_u_dianJiSpuGeShu['u_dianJiSpuGeShu_'+str(i)]
    Sample2_u_pingJunDianJiSpu = pd.DataFrame(Sample2_u_pingJunDianJiSpu, columns=['u_pingJunDianJiSpu_'+str(i)])
    Sample2_u_pingJunDianJiSpu = pd.concat([Sample2_u_dianJiTianShu['uid'],Sample2_u_pingJunDianJiSpu['u_pingJunDianJiSpu_'+str(i)]], axis=1)
    
    #u下单的天数(u)
    group = Sample2_action1.groupby(['uid','datas'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby('uid')
    Sample2_u_xiaDanTianShu = pd.DataFrame(group.size(),columns=['u_xiaDanTianShu_'+str(i)]).reset_index()
    #u下单的不重复的spu个数(u)
    group = Sample2_action1.groupby(['uid','spu_id'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby('uid')
    Sample2_u_xiaDanSpuGeShu = pd.DataFrame(group.size(),columns=['u_xiaDanSpuGeShu_'+str(i)]).reset_index()
    #u下单量(u)
    group = Sample2_action1.groupby(['uid'])
    Sample2_u_xiaDanLiang = pd.DataFrame(group.size(),columns=['u_xiaDanLiang_'+str(i)]).reset_index()
    #u在下单过的天数里平均下单(u)
    Sample2_u_pingJunXiaDanLiang = Sample2_u_xiaDanLiang['u_xiaDanLiang_'+str(i)]/Sample2_u_xiaDanTianShu['u_xiaDanTianShu_'+str(i)]
    Sample2_u_pingJunXiaDanLiang = pd.DataFrame(Sample2_u_pingJunXiaDanLiang, columns=['u_pingJunXiaDanLiang_'+str(i)])
    Sample2_u_pingJunXiaDanLiang = pd.concat([Sample2_u_xiaDanTianShu['uid'],Sample2_u_pingJunXiaDanLiang['u_pingJunXiaDanLiang_'+str(i)]], axis=1)
    #u对spu的平均下单天数（次数）(u)
    Sample2_u_pingJunXiaDanSpu = Sample2_u_xiaDanLiang['u_xiaDanLiang_'+str(i)]/Sample2_u_xiaDanSpuGeShu['u_xiaDanSpuGeShu_'+str(i)]
    Sample2_u_pingJunXiaDanSpu = pd.DataFrame(Sample2_u_pingJunXiaDanSpu, columns=['u_pingJunXiaDanSpu_'+str(i)])
    Sample2_u_pingJunXiaDanSpu = pd.concat([Sample2_u_xiaDanTianShu['uid'],Sample2_u_pingJunXiaDanSpu['u_pingJunXiaDanSpu_'+str(i)]], axis=1)
        
    ########################################### spu 特征 ############################################
    #spu被点击天数（spu）
    group = Sample2_action0.groupby(['spu_id','datas'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby('spu_id')
    Sample2_spu_beiDianJiTianShu = pd.DataFrame(group.size(),columns=['spu_beiDianJiTianShu_'+str(i)]).reset_index()
    #spu被点击的不重复的人数（spu）
    group = Sample2_action0.groupby(['spu_id','uid'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby('spu_id')
    Sample2_spu_beiDianJiRenShu = pd.DataFrame(group.size(),columns=['spu_beiDianJiRenShu_'+str(i)]).reset_index()
    #spu被点击量（spu）
    group = Sample2_action0.groupby(['spu_id'])
    Sample2_spu_beiDianJiLiang = pd.DataFrame(group.size(),columns=['spu_beiDianJiLiang_'+str(i)]).reset_index()
    #spu平均每天被点击量（spu）
    Sample2_spu_pingJunBeiDianJiLiang = Sample2_spu_beiDianJiLiang['spu_beiDianJiLiang_'+str(i)]/Sample2_spu_beiDianJiTianShu['spu_beiDianJiTianShu_'+str(i)]
    Sample2_spu_pingJunBeiDianJiLiang = pd.DataFrame(Sample2_spu_pingJunBeiDianJiLiang, columns=['spu_pingJunBeiDianJiLiang_'+str(i)])
    Sample2_spu_pingJunBeiDianJiLiang = pd.concat([Sample2_spu_beiDianJiTianShu['spu_id'],Sample2_spu_pingJunBeiDianJiLiang['spu_pingJunBeiDianJiLiang_'+str(i)]],axis=1)
    #spu被点击过的人平均点击量（spu）
    '''
    Sample2_spu_pingJunBeiDianJi_meiRen = Sample2_spu_beiDianJiLiang['spu_beiDianJiLiang_'+str(i)]/Sample2_spu_beiDianJiTianShu['spu_beiDianJiTianShu_'+str(i)]
    Sample2_spu_pingJunBeiDianJi_meiRen = pd.DataFrame(Sample2_spu_pingJunBeiDianJi_meiRen, columns=['spu_pingJunBeiDianJi_meiRen_'+str(i)])
    Sample2_spu_pingJunBeiDianJi_meiRen = pd.concat([Sample2_spu_beiDianJiTianShu['spu_id'],Sample2_spu_pingJunBeiDianJi_meiRen['spu_pingJunBeiDianJi_meiRen_'+str(i)]],axis=1)
    '''
    
    #spu被下单天数（spu）
    group = Sample2_action1.groupby(['spu_id','datas'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby('spu_id')
    Sample2_spu_beiXiaDanTianShu = pd.DataFrame(group.size(),columns=['spu_beiXiaDanTianShu_'+str(i)]).reset_index()
    #spu被下单的不重复的人数（spu）
    group = Sample2_action1.groupby(['spu_id','uid'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby('spu_id')
    Sample2_spu_beiXiaDanRenShu = pd.DataFrame(group.size(),columns=['spu_beiXiaDanRenShu_'+str(i)]).reset_index()
    #spu被下单量（spu）
    group = Sample2_action1.groupby(['spu_id'])
    Sample2_spu_beiXiaDanLiang = pd.DataFrame(group.size(),columns=['spu_beiXiaDanLiang_'+str(i)]).reset_index()
    #spu平均每天被下单量（spu）
    Sample2_spu_pingJunBeiXiaDanLiang = Sample2_spu_beiXiaDanLiang['spu_beiXiaDanLiang_'+str(i)]/Sample2_spu_beiXiaDanTianShu['spu_beiXiaDanTianShu_'+str(i)]
    Sample2_spu_pingJunBeiXiaDanLiang = pd.DataFrame(Sample2_spu_pingJunBeiXiaDanLiang, columns=['spu_pingJunBeiXiaDanLiang_'+str(i)])
    Sample2_spu_pingJunBeiXiaDanLiang = pd.concat([Sample2_spu_beiXiaDanTianShu['spu_id'],Sample2_spu_pingJunBeiXiaDanLiang['spu_pingJunBeiXiaDanLiang_'+str(i)]],axis=1)
    #spu被点击过的人平均下单量（spu）
    '''
    Sample2_spu_pingJunBeiXiaDan_meiRen = Sample2_spu_beiXiaDanLiang['spu_beiXiaDanLiang_'+str(i)]/Sample2_spu_beiXiaDanTianShu['spu_beiXiaDanTianShu_'+str(i)]
    Sample2_spu_pingJunBeiXiaDan_meiRen = pd.DataFrame(Sample2_spu_pingJunBeiXiaDan_meiRen, columns=['spu_pingJunBeiXiaDan_meiRen_'+str(i)])
    Sample2_spu_pingJunBeiXiaDan_meiRen = pd.concat([Sample2_spu_beiXiaDanTianShu['spu_id'],Sample2_spu_pingJunBeiXiaDan_meiRen['spu_pingJunBeiXiaDan_meiRen_'+str(i)]],axis=1)
    '''
    
    ########################################### u- cate 特征 ############################################
    #u点击cate的天数
    group = Sample2_action0.groupby(['uid','cate_id','datas'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby(['uid','cate_id'])  #merge用
    Sample2_u_cate_dianJiTianShu = pd.DataFrame(group.size(),columns=['u_cate_dianJiTianShu_'+str(i)]).reset_index()
    #u对cate的点击量
    group = Sample2_action0.groupby(['uid','cate_id']) #merge用
    Sample2_u_cate_dianJiLiang = pd.DataFrame(group.size(),columns=['u_cate_dianJiLiang_'+str(i)]).reset_index()
    #u对cate的平均每天点击量
    Sample2_u_cate_pingJunDianJiLiang = Sample2_u_cate_dianJiLiang['u_cate_dianJiLiang_'+str(i)]/Sample2_u_cate_dianJiTianShu['u_cate_dianJiTianShu_'+str(i)]
    Sample2_u_cate_pingJunDianJiLiang = pd.DataFrame(Sample2_u_cate_pingJunDianJiLiang, columns=['u_cate_pingJunDianJiLiang_'+str(i)])
    Sample2_u_cate_pingJunDianJiLiang = pd.concat([Sample2_u_cate_dianJiTianShu[['uid','cate_id']],Sample2_u_cate_pingJunDianJiLiang['u_cate_pingJunDianJiLiang_'+str(i)]],axis=1)
    #u点击cate类下的spu不重复的个数
    group = Sample2_action0.groupby(['uid','cate_id','spu_id'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby(['uid','cate_id']) 
    Sample2_u_cate_dianJiSpuGeShu = pd.DataFrame(group.size(),columns=['u_cate_dianJiSpuGeShu_'+str(i)]).reset_index()
    
    #u下单cate的天数
    group = Sample2_action1.groupby(['uid','cate_id','datas'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby(['uid','cate_id']) #merge用
    Sample2_u_cate_xiaDanTianShu = pd.DataFrame(group.size(),columns=['u_cate_xiaDanTianShu_'+str(i)]).reset_index()
    #u对cate的下单量
    group = Sample2_action1.groupby(['uid','cate_id']) #merge用
    Sample2_u_cate_xiaDanLiang = pd.DataFrame(group.size(),columns=['u_cate_xiaDanLiang_'+str(i)]).reset_index()
    #u对cate的平均每天下单量
    Sample2_u_cate_pingJunXiaDanLiang = Sample2_u_cate_xiaDanLiang['u_cate_xiaDanLiang_'+str(i)]/Sample2_u_cate_xiaDanTianShu['u_cate_xiaDanTianShu_'+str(i)]
    Sample2_u_cate_pingJunXiaDanLiang = pd.DataFrame(Sample2_u_cate_pingJunXiaDanLiang, columns=['u_cate_pingJunXiaDanLiang_'+str(i)])
    Sample2_u_cate_pingJunXiaDanLiang = pd.concat([Sample2_u_cate_dianJiTianShu[['uid','cate_id']],Sample2_u_cate_pingJunXiaDanLiang['u_cate_pingJunXiaDanLiang_'+str(i)]],axis=1)
    #u下单cate类下的spu不重复的个数
    group = Sample2_action1.groupby(['uid','cate_id','spu_id'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby(['uid','cate_id']) 
    Sample2_u_cate_XiaDanSpuGeShu = pd.DataFrame(group.size(),columns=['u_cate_XiaDanSpuGeShu_'+str(i)]).reset_index()
    
    ########################################### cate 特征 ############################################

    #cate被点击天数
    '''
    group = Sample2_action0.groupby(['cate_id','datas'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby('cate_id')
    Sample2_cate_beiDianJiTianShu = pd.DataFrame(group.size(),columns=['cate_beiDianJiTianShu_'+str(i)]).reset_index()
    '''
    #cate 被下单天数
    group = Sample2_action1.groupby(['cate_id','datas'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby('cate_id')
    Sample2_cate_beiXiaDanTianShu = pd.DataFrame(group.size(),columns=['cate_beiXiaDanTianShu_'+str(i)]).reset_index()
    
    #cate被点击人数
    group = Sample2_action0.groupby(['cate_id','uid'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby('cate_id')
    Sample2_cate_beiDianJiRenShu = pd.DataFrame(group.size(),columns=['cate_beiDianJiRenShu_'+str(i)]).reset_index()
    #cate被下单人数
    group = Sample2_action1.groupby(['cate_id','uid'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby('cate_id')
    Sample2_cate_beiXiaDanRenShu = pd.DataFrame(group.size(),columns=['cate_beiXiaDanRenShu_'+str(i)]).reset_index()

    ########################################### u-brand 特征 ############################################
    #u点击brand的天数
    group = Sample2_action0.groupby(['uid','brand_id','datas'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby(['uid','brand_id'])  #merge用
    Sample2_u_brand_dianJiTianShu = pd.DataFrame(group.size(),columns=['u_brand_dianJiTianShu_'+str(i)]).reset_index()
    #u对brand的点击量
    group = Sample2_action0.groupby(['uid','brand_id']) #merge用
    Sample2_u_brand_dianJiLiang = pd.DataFrame(group.size(),columns=['u_brand_dianJiLiang_'+str(i)]).reset_index()
    #u对brand的平均每天点击量
    Sample2_u_brand_pingJunDianJiLiang = Sample2_u_brand_dianJiLiang['u_brand_dianJiLiang_'+str(i)]/Sample2_u_brand_dianJiTianShu['u_brand_dianJiTianShu_'+str(i)]
    Sample2_u_brand_pingJunDianJiLiang = pd.DataFrame(Sample2_u_brand_pingJunDianJiLiang, columns=['u_brand_pingJunDianJiLiang_'+str(i)])
    Sample2_u_brand_pingJunDianJiLiang = pd.concat([Sample2_u_brand_dianJiTianShu[['uid','brand_id']],Sample2_u_brand_pingJunDianJiLiang['u_brand_pingJunDianJiLiang_'+str(i)]],axis=1)
    #u点击brand类下的spu不重复的个数
    group = Sample2_action0.groupby(['uid','brand_id','spu_id'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby(['uid','brand_id']) 
    Sample2_u_brand_dianJiSpuGeShu = pd.DataFrame(group.size(),columns=['u_brand_dianJiSpuGeShu_'+str(i)]).reset_index()
    
    #u下单brand的天数
    group = Sample2_action1.groupby(['uid','brand_id','datas'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby(['uid','brand_id']) #merge用
    Sample2_u_brand_xiaDanTianShu = pd.DataFrame(group.size(),columns=['u_brand_xiaDanTianShu_'+str(i)]).reset_index()
    #u对brand的下单量
    group = Sample2_action1.groupby(['uid','brand_id']) #merge用
    Sample2_u_brand_xiaDanLiang = pd.DataFrame(group.size(),columns=['u_brand_xiaDanLiang_'+str(i)]).reset_index()
    #u对brand的平均每天下单量
    Sample2_u_brand_pingJunXiaDanLiang = Sample2_u_brand_xiaDanLiang['u_brand_xiaDanLiang_'+str(i)]/Sample2_u_brand_xiaDanTianShu['u_brand_xiaDanTianShu_'+str(i)]
    Sample2_u_brand_pingJunXiaDanLiang = pd.DataFrame(Sample2_u_brand_pingJunXiaDanLiang, columns=['u_brand_pingJunXiaDanLiang_'+str(i)])
    Sample2_u_brand_pingJunXiaDanLiang = pd.concat([Sample2_u_brand_xiaDanTianShu[['uid','brand_id']],Sample2_u_brand_pingJunXiaDanLiang['u_brand_pingJunXiaDanLiang_'+str(i)]],axis=1)
    #u下单brand类下的spu不重复的个数
    '''
    group = Sample2_action1.groupby(['uid','brand_id','spu_id'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby(['uid','brand_id']) 
    Sample2_u_brand_xiaDanSpuGeShu = pd.DataFrame(group.size(),columns=['u_brand_xiaDanSpuGeShu_'+str(i)]).reset_index()
    '''
    
    ########################################### brand 特征 ############################################

    #brand被点击天数
    group = Sample2_action0.groupby(['brand_id','datas'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby('brand_id')
    Sample2_brand_beiDianJiTianShu = pd.DataFrame(group.size(),columns=['brand_beiDianJiTianShu_'+str(i)]).reset_index()
    #brand 被下单天数
    group = Sample2_action1.groupby(['brand_id','datas'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby('brand_id')
    Sample2_brand_beiXiaDanTianShu = pd.DataFrame(group.size(),columns=['brand_beiXiaDanTianShu_'+str(i)]).reset_index()
    
    #brand被点击人数
    group = Sample2_action0.groupby(['brand_id','uid'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby('brand_id')
    Sample2_brand_beiDianJiRenShu = pd.DataFrame(group.size(),columns=['brand_beiDianJiRenShu_'+str(i)]).reset_index()
    #brand被下单人数
    group = Sample2_action1.groupby(['brand_id','uid'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby('brand_id')
    Sample2_brand_beiXiaDanRenShu = pd.DataFrame(group.size(),columns=['brand_beiXiaDanRenShu_'+str(i)]).reset_index()

    
    
    del tmp
    del Sample2_action0
    del Sample2_action1
    #############################把7天的特征先merge了存下################################################
    # u-s 特征 
    Sample2_Features_XiaDan = pd.merge(Sample2_Features_XiaDan,Sample2_us_dianJiTianShu,how = 'left',on='u_spu_id')
    Sample2_Features_XiaDan = pd.merge(Sample2_Features_XiaDan,Sample2_us_xiaDanTianShu,how = 'left',on='u_spu_id')
    del Sample2_us_dianJiTianShu
    del Sample2_us_xiaDanTianShu
    # u 特征
    Sample2_Features_XiaDan = pd.merge(Sample2_Features_XiaDan,Sample2_u_chuXianTianShu,how = 'left',on='uid')
    Sample2_Features_XiaDan = pd.merge(Sample2_Features_XiaDan,Sample2_u_dianJiTianShu,how = 'left',on='uid')
    Sample2_Features_XiaDan = pd.merge(Sample2_Features_XiaDan,Sample2_u_dianJiSpuGeShu,how = 'left',on='uid')
    Sample2_Features_XiaDan = pd.merge(Sample2_Features_XiaDan,Sample2_u_dianJiLiang,how = 'left',on='uid')
    Sample2_Features_XiaDan = pd.merge(Sample2_Features_XiaDan,Sample2_u_pingJunDianJiLiang,how = 'left',on='uid')
    Sample2_Features_XiaDan = pd.merge(Sample2_Features_XiaDan,Sample2_u_pingJunDianJiSpu,how = 'left',on='uid')
    Sample2_Features_XiaDan = pd.merge(Sample2_Features_XiaDan,Sample2_u_xiaDanTianShu,how = 'left',on='uid')
    Sample2_Features_XiaDan = pd.merge(Sample2_Features_XiaDan,Sample2_u_xiaDanSpuGeShu,how = 'left',on='uid')
    Sample2_Features_XiaDan = pd.merge(Sample2_Features_XiaDan,Sample2_u_xiaDanLiang,how = 'left',on='uid')
    Sample2_Features_XiaDan = pd.merge(Sample2_Features_XiaDan,Sample2_u_pingJunXiaDanLiang,how = 'left',on='uid')
    Sample2_Features_XiaDan = pd.merge(Sample2_Features_XiaDan,Sample2_u_pingJunXiaDanSpu,how = 'left',on='uid')
    del Sample2_u_chuXianTianShu
    del Sample2_u_dianJiTianShu
    del Sample2_u_dianJiSpuGeShu	
    del Sample2_u_dianJiLiang
    del Sample2_u_pingJunDianJiLiang
    del Sample2_u_pingJunDianJiSpu
    del Sample2_u_xiaDanTianShu	
    del Sample2_u_xiaDanSpuGeShu
    del Sample2_u_xiaDanLiang
    del Sample2_u_pingJunXiaDanLiang	
    del Sample2_u_pingJunXiaDanSpu
    # spu 特征
    Sample2_Features_XiaDan = pd.merge(Sample2_Features_XiaDan,Sample2_spu_beiDianJiTianShu,how = 'left',on='spu_id')
    Sample2_Features_XiaDan = pd.merge(Sample2_Features_XiaDan,Sample2_spu_beiDianJiRenShu,how = 'left',on='spu_id')
    Sample2_Features_XiaDan = pd.merge(Sample2_Features_XiaDan,Sample2_spu_beiDianJiLiang,how = 'left',on='spu_id')
    Sample2_Features_XiaDan = pd.merge(Sample2_Features_XiaDan,Sample2_spu_pingJunBeiDianJiLiang,how = 'left',on='spu_id')
    #Sample2_Features_XiaDan = pd.merge(Sample2_Features_XiaDan,Sample2_spu_pingJunBeiDianJi_meiRen,how = 'left',on='spu_id')
    Sample2_Features_XiaDan = pd.merge(Sample2_Features_XiaDan,Sample2_spu_beiXiaDanTianShu,how = 'left',on='spu_id')
    Sample2_Features_XiaDan = pd.merge(Sample2_Features_XiaDan,Sample2_spu_beiXiaDanRenShu,how = 'left',on='spu_id')
    Sample2_Features_XiaDan = pd.merge(Sample2_Features_XiaDan,Sample2_spu_beiXiaDanLiang,how = 'left',on='spu_id')
    Sample2_Features_XiaDan = pd.merge(Sample2_Features_XiaDan,Sample2_spu_pingJunBeiXiaDanLiang,how = 'left',on='spu_id')
    #Sample2_Features_XiaDan = pd.merge(Sample2_Features_XiaDan,Sample2_spu_pingJunBeiXiaDan_meiRen,how = 'left',on='spu_id')
    del Sample2_spu_beiDianJiTianShu
    del Sample2_spu_beiDianJiRenShu
    del Sample2_spu_beiDianJiLiang
    del Sample2_spu_pingJunBeiDianJiLiang
    #del Sample2_spu_pingJunBeiDianJi_meiRen
    del Sample2_spu_beiXiaDanTianShu
    del Sample2_spu_beiXiaDanRenShu
    del Sample2_spu_beiXiaDanLiang
    del Sample2_spu_pingJunBeiXiaDanLiang
    #del Sample2_spu_pingJunBeiXiaDan_meiRen
    # u- cate 特征
    Sample2_Features_XiaDan = pd.merge(Sample2_Features_XiaDan,Sample2_u_cate_dianJiTianShu,how = 'left',on=['uid','cate_id'])
    Sample2_Features_XiaDan = pd.merge(Sample2_Features_XiaDan,Sample2_u_cate_dianJiLiang,how = 'left',on=['uid','cate_id'])
    Sample2_Features_XiaDan = pd.merge(Sample2_Features_XiaDan,Sample2_u_cate_pingJunDianJiLiang,how = 'left',on=['uid','cate_id'])
    Sample2_Features_XiaDan = pd.merge(Sample2_Features_XiaDan,Sample2_u_cate_dianJiSpuGeShu,how = 'left',on=['uid','cate_id'])
    Sample2_Features_XiaDan = pd.merge(Sample2_Features_XiaDan,Sample2_u_cate_xiaDanTianShu,how = 'left',on=['uid','cate_id'])
    Sample2_Features_XiaDan = pd.merge(Sample2_Features_XiaDan,Sample2_u_cate_xiaDanLiang,how = 'left',on=['uid','cate_id'])
    Sample2_Features_XiaDan = pd.merge(Sample2_Features_XiaDan,Sample2_u_cate_pingJunXiaDanLiang,how = 'left',on=['uid','cate_id'])
    Sample2_Features_XiaDan = pd.merge(Sample2_Features_XiaDan,Sample2_u_cate_XiaDanSpuGeShu,how = 'left',on=['uid','cate_id'])
    del Sample2_u_cate_dianJiTianShu
    del Sample2_u_cate_dianJiLiang
    del Sample2_u_cate_pingJunDianJiLiang
    del Sample2_u_cate_dianJiSpuGeShu
    del Sample2_u_cate_xiaDanTianShu
    del Sample2_u_cate_xiaDanLiang
    del Sample2_u_cate_pingJunXiaDanLiang
    del Sample2_u_cate_XiaDanSpuGeShu
    
    ## cate 特征
    #Sample2_Features_XiaDan = pd.merge(Sample2_Features_XiaDan,Sample2_cate_beiDianJiTianShu,how = 'left',on='cate_id')
    Sample2_Features_XiaDan = pd.merge(Sample2_Features_XiaDan,Sample2_cate_beiXiaDanTianShu,how = 'left',on='cate_id')
    Sample2_Features_XiaDan = pd.merge(Sample2_Features_XiaDan,Sample2_cate_beiDianJiRenShu,how = 'left',on='cate_id')
    Sample2_Features_XiaDan = pd.merge(Sample2_Features_XiaDan,Sample2_cate_beiXiaDanRenShu,how = 'left',on='cate_id')
    #del Sample2_cate_beiDianJiTianShu
    del Sample2_cate_beiXiaDanTianShu
    del Sample2_cate_beiDianJiRenShu
    del Sample2_cate_beiXiaDanRenShu

    ## u-brand 特征
    Sample2_Features_XiaDan = pd.merge(Sample2_Features_XiaDan,Sample2_u_brand_dianJiTianShu,how = 'left',on=['uid','brand_id'])
    Sample2_Features_XiaDan = pd.merge(Sample2_Features_XiaDan,Sample2_u_brand_dianJiLiang,how = 'left',on=['uid','brand_id'])
    Sample2_Features_XiaDan = pd.merge(Sample2_Features_XiaDan,Sample2_u_brand_pingJunDianJiLiang,how = 'left',on=['uid','brand_id'])
    Sample2_Features_XiaDan = pd.merge(Sample2_Features_XiaDan,Sample2_u_brand_dianJiSpuGeShu,how = 'left',on=['uid','brand_id'])
    Sample2_Features_XiaDan = pd.merge(Sample2_Features_XiaDan,Sample2_u_brand_xiaDanTianShu,how = 'left',on=['uid','brand_id'])
    Sample2_Features_XiaDan = pd.merge(Sample2_Features_XiaDan,Sample2_u_brand_xiaDanLiang,how = 'left',on=['uid','brand_id'])
    Sample2_Features_XiaDan = pd.merge(Sample2_Features_XiaDan,Sample2_u_brand_pingJunXiaDanLiang,how = 'left',on=['uid','brand_id'])
    #Sample2_Features_XiaDan = pd.merge(Sample2_Features_XiaDan,Sample2_u_brand_xiaDanSpuGeShu,how = 'left',on=['uid','brand_id'])
    del Sample2_u_brand_dianJiTianShu
    del Sample2_u_brand_dianJiLiang
    del Sample2_u_brand_pingJunDianJiLiang
    del Sample2_u_brand_dianJiSpuGeShu
    del Sample2_u_brand_xiaDanTianShu
    del Sample2_u_brand_xiaDanLiang
    del Sample2_u_brand_pingJunXiaDanLiang
    #del Sample2_u_brand_xiaDanSpuGeShu
    
    # brand 特征 
    Sample2_Features_XiaDan = pd.merge(Sample2_Features_XiaDan,Sample2_brand_beiDianJiTianShu,how = 'left',on='brand_id')
    Sample2_Features_XiaDan = pd.merge(Sample2_Features_XiaDan,Sample2_brand_beiXiaDanTianShu,how = 'left',on='brand_id')
    Sample2_Features_XiaDan = pd.merge(Sample2_Features_XiaDan,Sample2_brand_beiDianJiRenShu,how = 'left',on='brand_id')
    Sample2_Features_XiaDan = pd.merge(Sample2_Features_XiaDan,Sample2_brand_beiXiaDanRenShu,how = 'left',on='brand_id')
    del Sample2_brand_beiDianJiTianShu
    del Sample2_brand_beiXiaDanTianShu
    del Sample2_brand_beiDianJiRenShu
    del Sample2_brand_beiXiaDanRenShu

    ##################################补缺失值  或者最后统一补 #################################
    Sample2_Features_XiaDan = Sample2_Features_XiaDan.fillna(value=0)
    #前3列没用了
    x_columns = [x for x in Sample2_Features_XiaDan.columns if x not in ['uid', 'spu_id','action_type','brand_id','cate_id']]
    Sample2_Features_XiaDan = Sample2_Features_XiaDan[x_columns]
    
    #导出数据
    Sample2_Features_XiaDan.to_csv('F:/Data_Vip/data/python_Features/Sample2_Features_'+str(i)+'.csv', index=False)  # 不带索引