# -*- coding: utf-8 -*-

import pandas as pd
import lightgbm as lgb
#from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.model_selection import GridSearchCV
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import classification_report


# 加载数据
#Sample2_Features_XiaDan_0318_0324 = pd.read_csv('F:/Data_Vip/data/python_Features/Sample2_Features.csv')
Online_Features = pd.read_csv('F:/Data_Vip/data/python_Features/Online_Features.csv')

'''
train_All = Sample2_Features_XiaDan_0318_0324
del Sample2_Features_XiaDan_0318_0324
train_Positive = train_All[(train_All.action_type == 1)]  # 取正样本
train_Negative = train_All[(train_All.action_type == 0)]  # 取负样本
train_Negative_Part = train_Negative.sample(frac=0.1, replace=False)  #原来是0.07 # 按frac比例随机采样,无放回
print "训练集的正负样本比例: 1:%s" % (float(len(train_Negative_Part)) / len(train_Positive))
train_All = train_Positive.append(train_Negative_Part, ignore_index=False)  # 合并，(False:原来的索引,默认false)
train = train_All.sort_index()  # 按索引排序
'''


#训练集的X和Y
train = pd.read_csv('F:/Data_Vip/data/python_Features/Sample2_caiyang.csv')
Y_train = train['action_type']
X_columns = [x for x in train.columns if x not in ['u_spu_id', 'action_type']]
X_train = train[X_columns]

#模型

#gbm = lgb.LGBMClassifier(num_leaves=4,n_estimators=400, learning_rate=0.15, subsample=0.8,#  max_depth=4, max_bin=750,
#                         seed=10, nthread=3, objective="binary")
gbm = lgb.LGBMClassifier(num_leaves=16, n_estimators=400, learning_rate=0.05,# subsample=0.8,# max_depth=4,max_bin=750,
                         seed=10, nthread=3, objective="binary")

gbm.fit(X_train,Y_train)


#线上预测概率：
X_Online = Online_Features
X_columns = [x for x in X_Online.columns if x not in ['u_spu_id']]
X_Online = X_Online[X_columns]
weight_Online = gbm.predict_proba(X_Online)
weight_Online = pd.DataFrame(weight_Online[:,1])
weight_Online = weight_Online.round(3)#保留3位小数
weight_Online.to_csv('F:/Data_Vip/data/model_outPut/weight.txt', header=None, index=False)  # 不带索引

#把weight_Online处理一下
weight_Online = pd.read_csv('F:/Data_Vip/data/model_outPut/weight.txt',header=None,names=['weight'])
weight_sort=weight_Online.sort_values(by='weight',ascending=False).reset_index(drop=True)
jiezhi_05 = int(round(len(weight_sort)*0.011))
num = weight_sort.iloc[jiezhi_05,0]#按从大到小排序，前1.1%的
print num
num = 0.5 - num
weight_Online_05 = weight_Online + num#直接全加感觉不太好，本来是0的还是为0？？？
weight_Online_05[weight_Online_05.weight>1]=1
weight_Online_05[weight_Online.weight==0]=0 #本来是0的还是为0，会提高一点点
weight_Online_05.to_csv('F:/Data_Vip/data/model_outPut/weight_05.txt', header=None, index=False)  # 不带索引