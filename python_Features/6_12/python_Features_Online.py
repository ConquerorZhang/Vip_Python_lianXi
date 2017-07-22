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
        Online = user_action_train[user_action_train.datas == '03-31']
    if i==3:
        Online = user_action_train[user_action_train.datas > '03-28']
    if i==7:
        Online = user_action_train[user_action_train.datas > '03-24']
    if i==14:
        Online = user_action_train[user_action_train.datas > '03-17']
    if i==28:
        Online = user_action_train[user_action_train.datas > '03-03']
    if i==35:
        Online = user_action_train[user_action_train.datas > '02-24']
    if i==100:
        Online = user_action_train
    del user_action_train
    ###################################Online的训练集merge了goods的属性 特征###################################
    Online = pd.merge(Online,goods_train,how = 'left',on='spu_id')
    del goods_train
    ########################################### u-s 特征 ############################################
    #us的点击天数(us)
    Online_action0 = Online[Online.action_type == 0]
    group = Online_action0.groupby('u_spu_id')
    Online_us_dianJiTianShu = pd.DataFrame(group.size(),columns=['us_dianJiTianShu_'+str(i)]).reset_index()
    #us的下单天数(us)
    Online_action1 = Online[Online.action_type == 1]
    group = Online_action1.groupby('u_spu_id')
    Online_us_xiaDanTianShu = pd.DataFrame(group.size(),columns=['us_xiaDanTianShu_'+str(i)]).reset_index()
    #del Online
    
    ########################################### u 特征 ############################################
    #u出现的天数(u)
    group = Online.groupby(['uid','datas'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby('uid')
    Online_u_chuXianTianShu = pd.DataFrame(group.size(),columns=['u_chuXianTianShu_'+str(i)]).reset_index()
    
    #u点击的天数(u)
    group = Online_action0.groupby(['uid','datas'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby('uid')
    Online_u_dianJiTianShu = pd.DataFrame(group.size(),columns=['u_dianJiTianShu_'+str(i)]).reset_index()
    #u点击的spu个数(u)
    group = Online_action0.groupby(['uid','spu_id'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby('uid')
    Online_u_dianJiSpuGeShu = pd.DataFrame(group.size(),columns=['u_dianJiSpuGeShu_'+str(i)]).reset_index()
    #u点击量(u)
    group = Online_action0.groupby(['uid'])
    Online_u_dianJiLiang = pd.DataFrame(group.size(),columns=['u_dianJiLiang_'+str(i)]).reset_index()
    #u在点击过的天数里平均点击量(u)
    Online_u_pingJunDianJiLiang = Online_u_dianJiLiang['u_dianJiLiang_'+str(i)]/Online_u_dianJiTianShu['u_dianJiTianShu_'+str(i)]
    Online_u_pingJunDianJiLiang = pd.DataFrame(Online_u_pingJunDianJiLiang, columns=['u_pingJunDianJiLiang_'+str(i)])
    Online_u_pingJunDianJiLiang = pd.concat([Online_u_dianJiTianShu['uid'],Online_u_pingJunDianJiLiang['u_pingJunDianJiLiang_'+str(i)]], axis=1)
    #u对spu的平均点击天数（次数）(u)
    Online_u_pingJunDianJiSpu = Online_u_dianJiLiang['u_dianJiLiang_'+str(i)]/Online_u_dianJiSpuGeShu['u_dianJiSpuGeShu_'+str(i)]
    Online_u_pingJunDianJiSpu = pd.DataFrame(Online_u_pingJunDianJiSpu, columns=['u_pingJunDianJiSpu_'+str(i)])
    Online_u_pingJunDianJiSpu = pd.concat([Online_u_dianJiTianShu['uid'],Online_u_pingJunDianJiSpu['u_pingJunDianJiSpu_'+str(i)]], axis=1)
    
    #u下单的天数(u)
    group = Online_action1.groupby(['uid','datas'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby('uid')
    Online_u_xiaDanTianShu = pd.DataFrame(group.size(),columns=['u_xiaDanTianShu_'+str(i)]).reset_index()
    #u下单的不重复的spu个数(u)
    group = Online_action1.groupby(['uid','spu_id'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby('uid')
    Online_u_xiaDanSpuGeShu = pd.DataFrame(group.size(),columns=['u_xiaDanSpuGeShu_'+str(i)]).reset_index()
    #u下单量(u)
    group = Online_action1.groupby(['uid'])
    Online_u_xiaDanLiang = pd.DataFrame(group.size(),columns=['u_xiaDanLiang_'+str(i)]).reset_index()
    #u在下单过的天数里平均下单(u)
    Online_u_pingJunXiaDanLiang = Online_u_xiaDanLiang['u_xiaDanLiang_'+str(i)]/Online_u_xiaDanTianShu['u_xiaDanTianShu_'+str(i)]
    Online_u_pingJunXiaDanLiang = pd.DataFrame(Online_u_pingJunXiaDanLiang, columns=['u_pingJunXiaDanLiang_'+str(i)])
    Online_u_pingJunXiaDanLiang = pd.concat([Online_u_xiaDanTianShu['uid'],Online_u_pingJunXiaDanLiang['u_pingJunXiaDanLiang_'+str(i)]], axis=1)
    #u对spu的平均下单天数（次数）(u)
    Online_u_pingJunXiaDanSpu = Online_u_xiaDanLiang['u_xiaDanLiang_'+str(i)]/Online_u_xiaDanSpuGeShu['u_xiaDanSpuGeShu_'+str(i)]
    Online_u_pingJunXiaDanSpu = pd.DataFrame(Online_u_pingJunXiaDanSpu, columns=['u_pingJunXiaDanSpu_'+str(i)])
    Online_u_pingJunXiaDanSpu = pd.concat([Online_u_xiaDanTianShu['uid'],Online_u_pingJunXiaDanSpu['u_pingJunXiaDanSpu_'+str(i)]], axis=1)
        
    ########################################### spu 特征 ############################################
    #spu被点击天数（spu）
    group = Online_action0.groupby(['spu_id','datas'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby('spu_id')
    Online_spu_beiDianJiTianShu = pd.DataFrame(group.size(),columns=['spu_beiDianJiTianShu_'+str(i)]).reset_index()
    #spu被点击的不重复的人数（spu）
    group = Online_action0.groupby(['spu_id','uid'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby('spu_id')
    Online_spu_beiDianJiRenShu = pd.DataFrame(group.size(),columns=['spu_beiDianJiRenShu_'+str(i)]).reset_index()
    #spu被点击量（spu）
    group = Online_action0.groupby(['spu_id'])
    Online_spu_beiDianJiLiang = pd.DataFrame(group.size(),columns=['spu_beiDianJiLiang_'+str(i)]).reset_index()
    #spu平均每天被点击量（spu）
    Online_spu_pingJunBeiDianJiLiang = Online_spu_beiDianJiLiang['spu_beiDianJiLiang_'+str(i)]/Online_spu_beiDianJiTianShu['spu_beiDianJiTianShu_'+str(i)]
    Online_spu_pingJunBeiDianJiLiang = pd.DataFrame(Online_spu_pingJunBeiDianJiLiang, columns=['spu_pingJunBeiDianJiLiang_'+str(i)])
    Online_spu_pingJunBeiDianJiLiang = pd.concat([Online_spu_beiDianJiTianShu['spu_id'],Online_spu_pingJunBeiDianJiLiang['spu_pingJunBeiDianJiLiang_'+str(i)]],axis=1)
    #spu被点击过的人平均点击量（spu）
    Online_spu_pingJunBeiDianJi_meiRen = Online_spu_beiDianJiLiang['spu_beiDianJiLiang_'+str(i)]/Online_spu_beiDianJiTianShu['spu_beiDianJiTianShu_'+str(i)]
    Online_spu_pingJunBeiDianJi_meiRen = pd.DataFrame(Online_spu_pingJunBeiDianJi_meiRen, columns=['spu_pingJunBeiDianJi_meiRen_'+str(i)])
    Online_spu_pingJunBeiDianJi_meiRen = pd.concat([Online_spu_beiDianJiTianShu['spu_id'],Online_spu_pingJunBeiDianJi_meiRen['spu_pingJunBeiDianJi_meiRen_'+str(i)]],axis=1)
    
    #spu被下单天数（spu）
    group = Online_action1.groupby(['spu_id','datas'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby('spu_id')
    Online_spu_beiXiaDanTianShu = pd.DataFrame(group.size(),columns=['spu_beiXiaDanTianShu_'+str(i)]).reset_index()
    #spu被下单的不重复的人数（spu）
    group = Online_action1.groupby(['spu_id','uid'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby('spu_id')
    Online_spu_beiXiaDanRenShu = pd.DataFrame(group.size(),columns=['spu_beiXiaDanRenShu_'+str(i)]).reset_index()
    #spu被下单量（spu）
    group = Online_action1.groupby(['spu_id'])
    Online_spu_beiXiaDanLiang = pd.DataFrame(group.size(),columns=['spu_beiXiaDanLiang_'+str(i)]).reset_index()
    #spu平均每天被下单量（spu）
    Online_spu_pingJunBeiXiaDanLiang = Online_spu_beiXiaDanLiang['spu_beiXiaDanLiang_'+str(i)]/Online_spu_beiXiaDanTianShu['spu_beiXiaDanTianShu_'+str(i)]
    Online_spu_pingJunBeiXiaDanLiang = pd.DataFrame(Online_spu_pingJunBeiXiaDanLiang, columns=['spu_pingJunBeiXiaDanLiang_'+str(i)])
    Online_spu_pingJunBeiXiaDanLiang = pd.concat([Online_spu_beiXiaDanTianShu['spu_id'],Online_spu_pingJunBeiXiaDanLiang['spu_pingJunBeiXiaDanLiang_'+str(i)]],axis=1)
    #spu被点击过的人平均下单量（spu）
    Online_spu_pingJunBeiXiaDan_meiRen = Online_spu_beiXiaDanLiang['spu_beiXiaDanLiang_'+str(i)]/Online_spu_beiXiaDanTianShu['spu_beiXiaDanTianShu_'+str(i)]
    Online_spu_pingJunBeiXiaDan_meiRen = pd.DataFrame(Online_spu_pingJunBeiXiaDan_meiRen, columns=['spu_pingJunBeiXiaDan_meiRen_'+str(i)])
    Online_spu_pingJunBeiXiaDan_meiRen = pd.concat([Online_spu_beiXiaDanTianShu['spu_id'],Online_spu_pingJunBeiXiaDan_meiRen['spu_pingJunBeiXiaDan_meiRen_'+str(i)]],axis=1)

    ########################################### u- cate 特征 ############################################
    #u点击cate的天数
    group = Online_action0.groupby(['uid','cate_id','datas'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby(['uid','cate_id'])  #merge用
    Online_u_cate_dianJiTianShu = pd.DataFrame(group.size(),columns=['u_cate_dianJiTianShu_'+str(i)]).reset_index()
    #u对cate的点击量
    group = Online_action0.groupby(['uid','cate_id']) #merge用
    Online_u_cate_dianJiLiang = pd.DataFrame(group.size(),columns=['u_cate_dianJiLiang_'+str(i)]).reset_index()
    #u对cate的平均每天点击量
    Online_u_cate_pingJunDianJiLiang = Online_u_cate_dianJiLiang['u_cate_dianJiLiang_'+str(i)]/Online_u_cate_dianJiTianShu['u_cate_dianJiTianShu_'+str(i)]
    Online_u_cate_pingJunDianJiLiang = pd.DataFrame(Online_u_cate_pingJunDianJiLiang, columns=['u_cate_pingJunDianJiLiang_'+str(i)])
    Online_u_cate_pingJunDianJiLiang = pd.concat([Online_u_cate_dianJiTianShu[['uid','cate_id']],Online_u_cate_pingJunDianJiLiang['u_cate_pingJunDianJiLiang_'+str(i)]],axis=1)
    #u点击cate类下的spu不重复的个数
    group = Online_action0.groupby(['uid','cate_id','spu_id'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby(['uid','cate_id']) 
    Online_u_cate_dianJiSpuGeShu = pd.DataFrame(group.size(),columns=['u_cate_dianJiSpuGeShu_'+str(i)]).reset_index()
    
    #u下单cate的天数
    group = Online_action1.groupby(['uid','cate_id','datas'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby(['uid','cate_id']) #merge用
    Online_u_cate_xiaDanTianShu = pd.DataFrame(group.size(),columns=['u_cate_xiaDanTianShu_'+str(i)]).reset_index()
    #u对cate的下单量
    group = Online_action1.groupby(['uid','cate_id']) #merge用
    Online_u_cate_xiaDanLiang = pd.DataFrame(group.size(),columns=['u_cate_xiaDanLiang_'+str(i)]).reset_index()
    #u对cate的平均每天下单量
    Online_u_cate_pingJunXiaDanLiang = Online_u_cate_xiaDanLiang['u_cate_xiaDanLiang_'+str(i)]/Online_u_cate_xiaDanTianShu['u_cate_xiaDanTianShu_'+str(i)]
    Online_u_cate_pingJunXiaDanLiang = pd.DataFrame(Online_u_cate_pingJunXiaDanLiang, columns=['u_cate_pingJunXiaDanLiang_'+str(i)])
    Online_u_cate_pingJunXiaDanLiang = pd.concat([Online_u_cate_dianJiTianShu[['uid','cate_id']],Online_u_cate_pingJunXiaDanLiang['u_cate_pingJunXiaDanLiang_'+str(i)]],axis=1)
    #u下单cate类下的spu不重复的个数
    group = Online_action1.groupby(['uid','cate_id','spu_id'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby(['uid','cate_id']) 
    Online_u_cate_XiaDanSpuGeShu = pd.DataFrame(group.size(),columns=['u_cate_XiaDanSpuGeShu_'+str(i)]).reset_index()
    
    ########################################### cate 特征 ############################################

    #cate被点击天数
    group = Online_action0.groupby(['cate_id','datas'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby('cate_id')
    Online_cate_beiDianJiTianShu = pd.DataFrame(group.size(),columns=['cate_beiDianJiTianShu_'+str(i)]).reset_index()
    #cate 被下单天数
    group = Online_action1.groupby(['cate_id','datas'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby('cate_id')
    Online_cate_beiXiaDanTianShu = pd.DataFrame(group.size(),columns=['cate_beiXiaDanTianShu_'+str(i)]).reset_index()
    
    #cate被点击人数
    group = Online_action0.groupby(['cate_id','uid'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby('cate_id')
    Online_cate_beiDianJiRenShu = pd.DataFrame(group.size(),columns=['cate_beiDianJiRenShu_'+str(i)]).reset_index()
    #cate被下单人数
    group = Online_action1.groupby(['cate_id','uid'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby('cate_id')
    Online_cate_beiXiaDanRenShu = pd.DataFrame(group.size(),columns=['cate_beiXiaDanRenShu_'+str(i)]).reset_index()

    ########################################### u-brand 特征 ############################################
    #u点击brand的天数
    group = Online_action0.groupby(['uid','brand_id','datas'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby(['uid','brand_id'])  #merge用
    Online_u_brand_dianJiTianShu = pd.DataFrame(group.size(),columns=['u_brand_dianJiTianShu_'+str(i)]).reset_index()
    #u对brand的点击量
    group = Online_action0.groupby(['uid','brand_id']) #merge用
    Online_u_brand_dianJiLiang = pd.DataFrame(group.size(),columns=['u_brand_dianJiLiang_'+str(i)]).reset_index()
    #u对brand的平均每天点击量
    Online_u_brand_pingJunDianJiLiang = Online_u_brand_dianJiLiang['u_brand_dianJiLiang_'+str(i)]/Online_u_brand_dianJiTianShu['u_brand_dianJiTianShu_'+str(i)]
    Online_u_brand_pingJunDianJiLiang = pd.DataFrame(Online_u_brand_pingJunDianJiLiang, columns=['u_brand_pingJunDianJiLiang_'+str(i)])
    Online_u_brand_pingJunDianJiLiang = pd.concat([Online_u_brand_dianJiTianShu[['uid','brand_id']],Online_u_brand_pingJunDianJiLiang['u_brand_pingJunDianJiLiang_'+str(i)]],axis=1)
    #u点击brand类下的spu不重复的个数
    group = Online_action0.groupby(['uid','brand_id','spu_id'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby(['uid','brand_id']) 
    Online_u_brand_dianJiSpuGeShu = pd.DataFrame(group.size(),columns=['u_brand_dianJiSpuGeShu_'+str(i)]).reset_index()
    
    #u下单brand的天数
    group = Online_action1.groupby(['uid','brand_id','datas'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby(['uid','brand_id']) #merge用
    Online_u_brand_xiaDanTianShu = pd.DataFrame(group.size(),columns=['u_brand_xiaDanTianShu_'+str(i)]).reset_index()
    #u对brand的下单量
    group = Online_action1.groupby(['uid','brand_id']) #merge用
    Online_u_brand_xiaDanLiang = pd.DataFrame(group.size(),columns=['u_brand_xiaDanLiang_'+str(i)]).reset_index()
    #u对brand的平均每天下单量
    Online_u_brand_pingJunXiaDanLiang = Online_u_brand_xiaDanLiang['u_brand_xiaDanLiang_'+str(i)]/Online_u_brand_xiaDanTianShu['u_brand_xiaDanTianShu_'+str(i)]
    Online_u_brand_pingJunXiaDanLiang = pd.DataFrame(Online_u_brand_pingJunXiaDanLiang, columns=['u_brand_pingJunXiaDanLiang_'+str(i)])
    Online_u_brand_pingJunXiaDanLiang = pd.concat([Online_u_brand_xiaDanTianShu[['uid','brand_id']],Online_u_brand_pingJunXiaDanLiang['u_brand_pingJunXiaDanLiang_'+str(i)]],axis=1)
    #u下单brand类下的spu不重复的个数
    group = Online_action1.groupby(['uid','brand_id','spu_id'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby(['uid','brand_id']) 
    Online_u_brand_xiaDanSpuGeShu = pd.DataFrame(group.size(),columns=['u_brand_xiaDanSpuGeShu_'+str(i)]).reset_index()
    
    ########################################### brand 特征 ############################################

    #brand被点击天数
    group = Online_action0.groupby(['brand_id','datas'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby('brand_id')
    Online_brand_beiDianJiTianShu = pd.DataFrame(group.size(),columns=['brand_beiDianJiTianShu_'+str(i)]).reset_index()
    #brand 被下单天数
    group = Online_action1.groupby(['brand_id','datas'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby('brand_id')
    Online_brand_beiXiaDanTianShu = pd.DataFrame(group.size(),columns=['brand_beiXiaDanTianShu_'+str(i)]).reset_index()
    
    #brand被点击人数
    group = Online_action0.groupby(['brand_id','uid'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby('brand_id')
    Online_brand_beiDianJiRenShu = pd.DataFrame(group.size(),columns=['brand_beiDianJiRenShu_'+str(i)]).reset_index()
    #brand被下单人数
    group = Online_action1.groupby(['brand_id','uid'])
    tmp = pd.DataFrame(group.size()).reset_index()
    group = tmp.groupby('brand_id')
    Online_brand_beiXiaDanRenShu = pd.DataFrame(group.size(),columns=['brand_beiXiaDanRenShu_'+str(i)]).reset_index()

    
    
    del tmp
    del Online_action0
    del Online_action1
    #############################把7天的特征先merge了存下################################################
    # u-s 特征 
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_us_dianJiTianShu,how = 'left',on='u_spu_id')
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_us_xiaDanTianShu,how = 'left',on='u_spu_id')
    del Online_us_dianJiTianShu
    del Online_us_xiaDanTianShu
    # u 特征
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_u_chuXianTianShu,how = 'left',on='uid')
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_u_dianJiTianShu,how = 'left',on='uid')
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_u_dianJiSpuGeShu,how = 'left',on='uid')
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_u_dianJiLiang,how = 'left',on='uid')
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_u_pingJunDianJiLiang,how = 'left',on='uid')
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_u_pingJunDianJiSpu,how = 'left',on='uid')
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_u_xiaDanTianShu,how = 'left',on='uid')
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_u_xiaDanSpuGeShu,how = 'left',on='uid')
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_u_xiaDanLiang,how = 'left',on='uid')
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_u_pingJunXiaDanLiang,how = 'left',on='uid')
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_u_pingJunXiaDanSpu,how = 'left',on='uid')
    del Online_u_chuXianTianShu
    del Online_u_dianJiTianShu
    del Online_u_dianJiSpuGeShu	
    del Online_u_dianJiLiang
    del Online_u_pingJunDianJiLiang
    del Online_u_pingJunDianJiSpu
    del Online_u_xiaDanTianShu	
    del Online_u_xiaDanSpuGeShu
    del Online_u_xiaDanLiang
    del Online_u_pingJunXiaDanLiang	
    del Online_u_pingJunXiaDanSpu
    # spu 特征
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_spu_beiDianJiTianShu,how = 'left',on='spu_id')
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_spu_beiDianJiRenShu,how = 'left',on='spu_id')
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_spu_beiDianJiLiang,how = 'left',on='spu_id')
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_spu_pingJunBeiDianJiLiang,how = 'left',on='spu_id')
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_spu_pingJunBeiDianJi_meiRen,how = 'left',on='spu_id')
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_spu_beiXiaDanTianShu,how = 'left',on='spu_id')
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_spu_beiXiaDanRenShu,how = 'left',on='spu_id')
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_spu_beiXiaDanLiang,how = 'left',on='spu_id')
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_spu_pingJunBeiXiaDanLiang,how = 'left',on='spu_id')
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_spu_pingJunBeiXiaDan_meiRen,how = 'left',on='spu_id')
    del Online_spu_beiDianJiTianShu
    del Online_spu_beiDianJiRenShu
    del Online_spu_beiDianJiLiang
    del Online_spu_pingJunBeiDianJiLiang
    del Online_spu_pingJunBeiDianJi_meiRen
    del Online_spu_beiXiaDanTianShu
    del Online_spu_beiXiaDanRenShu
    del Online_spu_beiXiaDanLiang
    del Online_spu_pingJunBeiXiaDanLiang
    del Online_spu_pingJunBeiXiaDan_meiRen
    # u- cate 特征
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_u_cate_dianJiTianShu,how = 'left',on=['uid','cate_id'])
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_u_cate_dianJiLiang,how = 'left',on=['uid','cate_id'])
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_u_cate_pingJunDianJiLiang,how = 'left',on=['uid','cate_id'])
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_u_cate_dianJiSpuGeShu,how = 'left',on=['uid','cate_id'])
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_u_cate_xiaDanTianShu,how = 'left',on=['uid','cate_id'])
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_u_cate_xiaDanLiang,how = 'left',on=['uid','cate_id'])
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_u_cate_pingJunXiaDanLiang,how = 'left',on=['uid','cate_id'])
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_u_cate_XiaDanSpuGeShu,how = 'left',on=['uid','cate_id'])
    del Online_u_cate_dianJiTianShu
    del Online_u_cate_dianJiLiang
    del Online_u_cate_pingJunDianJiLiang
    del Online_u_cate_dianJiSpuGeShu
    del Online_u_cate_xiaDanTianShu
    del Online_u_cate_xiaDanLiang
    del Online_u_cate_pingJunXiaDanLiang
    del Online_u_cate_XiaDanSpuGeShu
    ## cate 特征

    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_cate_beiDianJiTianShu,how = 'left',on='cate_id')
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_cate_beiXiaDanTianShu,how = 'left',on='cate_id')
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_cate_beiDianJiRenShu,how = 'left',on='cate_id')
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_cate_beiXiaDanRenShu,how = 'left',on='cate_id')
    del Online_cate_beiDianJiTianShu
    del Online_cate_beiXiaDanTianShu
    del Online_cate_beiDianJiRenShu
    del Online_cate_beiXiaDanRenShu

    ## u-brand 特征
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_u_brand_dianJiTianShu,how = 'left',on=['uid','brand_id'])
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_u_brand_dianJiLiang,how = 'left',on=['uid','brand_id'])
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_u_brand_pingJunDianJiLiang,how = 'left',on=['uid','brand_id'])
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_u_brand_dianJiSpuGeShu,how = 'left',on=['uid','brand_id'])
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_u_brand_xiaDanTianShu,how = 'left',on=['uid','brand_id'])
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_u_brand_xiaDanLiang,how = 'left',on=['uid','brand_id'])
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_u_brand_pingJunXiaDanLiang,how = 'left',on=['uid','brand_id'])
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_u_brand_xiaDanSpuGeShu,how = 'left',on=['uid','brand_id'])
    del Online_u_brand_dianJiTianShu
    del Online_u_brand_dianJiLiang
    del Online_u_brand_pingJunDianJiLiang
    del Online_u_brand_dianJiSpuGeShu
    del Online_u_brand_xiaDanTianShu
    del Online_u_brand_xiaDanLiang
    del Online_u_brand_pingJunXiaDanLiang
    del Online_u_brand_xiaDanSpuGeShu
    # brand 特征 

    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_brand_beiDianJiTianShu,how = 'left',on='brand_id')
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_brand_beiXiaDanTianShu,how = 'left',on='brand_id')
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_brand_beiDianJiRenShu,how = 'left',on='brand_id')
    Online_Features_XiaDan = pd.merge(Online_Features_XiaDan,Online_brand_beiXiaDanRenShu,how = 'left',on='brand_id')
    del Online_brand_beiDianJiTianShu
    del Online_brand_beiXiaDanTianShu
    del Online_brand_beiDianJiRenShu
    del Online_brand_beiXiaDanRenShu

    ##################################补缺失值  或者最后统一补 #################################
    Online_Features_XiaDan = Online_Features_XiaDan.fillna(value=0)
    #前3列没用了
    x_columns = [x for x in Online_Features_XiaDan.columns if x not in ['uid', 'spu_id','brand_id','cate_id']]
    Online_Features_XiaDan = Online_Features_XiaDan[x_columns]
    
    #导出数据
    Online_Features_XiaDan.to_csv('F:/Data_Vip/data/python_Features/Online_Features_'+str(i)+'.csv', index=False)  # 不带索引