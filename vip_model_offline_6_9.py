# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn import metrics
import lightgbm as lgb
#from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.model_selection import GridSearchCV
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import classification_report


# 加载数据
Sample1_Features_XiaDan_0325_0331 = pd.read_csv('F:/Data_Vip/data/python_Features/Sample1_Features.csv')
Sample2_Features_XiaDan_0318_0324 = pd.read_csv('F:/Data_Vip/data/python_Features/Sample2_Features.csv')

train_All = Sample2_Features_XiaDan_0318_0324
del Sample2_Features_XiaDan_0318_0324
train_Positive = train_All[(train_All.action_type == 1)]  # 取正样本
train_Negative = train_All[(train_All.action_type == 0)]  # 取负样本

#for FRAC in [0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6]:
#    print "采样率：",FRAC
while(True):
    train_Negative_Part = train_Negative.sample(frac=0.1, replace=False)  #原来是0.07 # 按frac比例随机采样,无放回
    print "训练集的正负样本比例: 1:%s" % (float(len(train_Negative_Part)) / len(train_Positive))
    train_All = train_Positive.append(train_Negative_Part, ignore_index=False)  # 合并，(False:原来的索引,默认false)
    train = train_All.sort_index()  # 按索引排序
    #保存 导出 较好的训练样本
    train.to_csv('F:/Data_Vip/data/python_Features/Sample2_caiyang.csv',index=False)  # 不带索引


    #训练集的X和Y
    #train = Sample2_Features_XiaDan_0318_0324
    Y_train = train['action_type']
    X_columns = [x for x in train.columns if x not in ['u_spu_id', 'action_type']]
    X_train = train[X_columns]
    #测试集的X
    test = Sample1_Features_XiaDan_0325_0331
    Y_test = test['action_type']
    X_columns = [x for x in test.columns if x not in ['u_spu_id', 'action_type']]
    X_test = test[X_columns]

    #模型
    '''
    gbm = GradientBoostingClassifier(n_estimators=100,learning_rate=0.1, min_samples_split=500,
                                     min_samples_leaf=50,max_depth=3,max_features='sqrt',subsample=0.8,
                                     random_state=10)#
    '''
    '''
    estimators = range(100,501,50)#[170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270]  #
    learn = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
    dep = [2,3,4,5]
    num_le = [4,8,16]
    for dp in dep:
        for es in estimators:
            for lr in learn:
                print 'le:...........................................', dp
                print 'est:...........................................', es
                print 'lr:...........................................', lr
    
    gbm = lgb.LGBMClassifier(num_leaves=8, max_depth=dp, n_estimators=es, learning_rate=lr, subsample=0.8,#  max_depth=dp,max_bin=750,
                             seed=10,nthread=3,objective="binary")
    '''
    gbm = lgb.LGBMClassifier(num_leaves=16, n_estimators=400, learning_rate=0.05, #subsample=0.8,# max_depth=4,max_bin=750,
                             seed=10, nthread=3, objective="binary")

    print '到gbm了：'
    gbm.fit(X_train,Y_train)
    print '算概率： '
    weight_test = gbm.predict_proba(X_test)
    '''
    print '保存feature_importances_： '
    # 保存feature_importances_
    Feature_Imp = gbm.feature_importances_
    Feature_Imp = pd.Series(Feature_Imp)
    Feature_Names = pd.Series(X_train.columns)  # 特征的列名称
    Feature_Imp = pd.concat([Feature_Names, Feature_Imp], axis=1)
    Feature_Imp.to_csv('F:/Data_Vip/data/model_outPut/Feature_importances.csv', header=False, index=False)  # 不带索引
    '''
    #线下评分：
    weight_Sample1 = pd.DataFrame(weight_test[:, 1])
    weight_Sample1.to_csv('F:/Data_Vip/data/model_outPut/weight_Sample1.csv', header=False, index=False)
    pre_weight = np.array(weight_test[:,1])
    true_weight = np.array(Y_test)
    score_Offline = np.sqrt(metrics.mean_squared_error(true_weight, pre_weight))
    print '线下测试集评分： ',score_Offline

    if score_Offline<0.14858:
        break



