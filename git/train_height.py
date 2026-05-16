#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
################ data preparcessing ###############################
age_df = pd.read_csv('predict_shortterm.csv')
age_df['age'] = age_df['age'].map(lambda x: x if x < 217 else np.nan )
age_df = age_df.dropna(axis= 'index', how='any')
age_df.reset_index(drop=True, inplace = True)


df_data = pd.DataFrame(data= age_df,
                       columns= ['病歷號','gender','age','身高', '體重','父身高','母身高','boneage','用藥資訊','BBW','after_month','after_H'])
def map_growth_hormone(info):
    if pd.isnull(info):
        return None  # 如果是缺失值，返回 None
    codes = info.split(',')  # 根据逗号分隔
    total = 0  # 初始化总和
    for code in codes:
        code = code.strip()  # 去掉多余的空格
        if code in ['907452', '525953', '526455']:  # 代表 -1
            total += -1
        elif code in ['407756', '407751', '827954', '407952']:  # 代表 1
            total += 1
    if total > 1 :
        return 1
    elif total < -1 :
        return -1
    elif total == 0 and len(codes) >= 2 :
        return 0
    elif total == 0 and len(codes) == 1 :
        return 2
    else:
        return total
def assign_GH_GnRHa(row):
    if row['growth_hormone'] == 2:
        return pd.Series([0, 0])  # GH = 0, GnRHa = 0
    elif row['growth_hormone'] == 1:
        return pd.Series([1, 0])  # GH = 1, GnRHa = 0
    elif row['growth_hormone'] == -1:
        return pd.Series([0, 1])  # GH = 0, GnRHa = 1
    elif row['growth_hormone'] == 0:
        return pd.Series([1, 1])  # GH = 1, GnRHa = 1
    else:
        return pd.Series([None, None])  # 防止有其他未知值
    


df_data['growth_hormone'] = df_data['用藥資訊'].map(map_growth_hormone)
# print(df_data['growth_hormone'].value_counts())
# print(df_data[df_data['用藥資訊'] == '0'])
# exit()
df_data['boneage'] = df_data['boneage'].map(lambda x: 216.0 if x > 216.0 else x )
df_data['bmi'] = [j / ( (float(i)/100.0) ** 2 ) for i,j in zip(df_data['身高'],df_data['體重'])]
df_data['BBW'] = df_data['BBW'].apply(lambda x: x * 1000 if x < 10 else x)
# df_data['parents_h'] =  [(float(i) +float(j))/2.0 for i,j in zip(df_data['父身高'],df_data['母身高'])]
df_data.rename(columns={'父身高': 'father_h', '母身高': 'mother_h'}, inplace=True)
# df_data['growth_hormone'] = df_data['用藥資訊'].map(lambda x: -1 if (( '907452' in x ) or ( '525953' in x ) or ( '526455' in x )) else ( 1 if (( '407756' in x ) or ( '407751' in x ) or ( '827954' in x ) or ( '407952' in x )) else 0 ))
# df_data['bone_err'] = [float(i) - float(j) for i,j in zip(df_data['pred_boneage'],df_data['boneage'])]
# df_data['GH'] = df_data['growth_hormone'].map(lambda x: x if x==1 else 0 )
# df_data['GnRHa'] = df_data['growth_hormone'].map(lambda x: 1.0 if x==-1.0 else 0.0)
df_data['grow'] =  [(float(i) -float(j)) for i,j in zip(df_data['after_H'],df_data['身高'])]
df_data[['GH', 'GnRHa']] = df_data.apply(assign_GH_GnRHa, axis=1)
# df_data['father_h'] = df_data.apply(lambda row: row['father_h'] * 1.5 if row['gender'] == 1 else row['father_h'], axis=1)
# df_data['mother_h'] = df_data.apply(lambda row: row['mother_h'] * 1.5 if row['gender'] == 0 else row['mother_h'], axis=1)
# print(df_data[(df_data['GH'] == 1) & (df_data['GnRHa'] == 1)])
# exit()
df_data['month_category'] = pd.cut(df_data['after_month'], 10)
# print(df_data[(df_data['after_month'] < 7)])
# print(df_data[(df_data['after_month'] < 13) & (df_data['after_month'] > 6)])
# print(df_data[(df_data['after_month'] < 25) & (df_data['after_month'] > 12)])
# print(df_data)
# exit()
################## data split to train and val ######################
from sklearn.model_selection import train_test_split

seed = 1999

# print(df_data['GnRHa'].unique())
# print(df_data[(df_data['GH'] == 0) & (df_data['GnRHa'] == 1)])
# exit()

# 0 : GH=0 and GnRHa=0 , 1: GH=1 and GnRHa=0 , 2: GH=0 and GnRHa=1
def split_by_gh(data, num):
    if num == 0 :
        new_df = data[(df_data['GH'] == 0) & (df_data['GnRHa'] == 0)]
    elif num == 1 :
        new_df = data[(df_data['GH'] == 1) & (df_data['GnRHa'] == 0)]
    else: 
        new_df = data[(df_data['GH'] == 0) & (df_data['GnRHa'] == 1)]
    return new_df
def split_patients(data, test_size): #data = df_data
  
    unique_patient_ids = data['病歷號'].unique()

    # train_size = int(len(name_arr) * (1.0-test_size))
    # test_size = len(name_arr) - train_size
    unique_patient_data = data.groupby('病歷號').first()  # 只保留每個病歷號的第一條記錄
    stratify_column = unique_patient_data['month_category']  # month_category 作為分層的基礎

    train_patients, test_patients = train_test_split(unique_patient_ids, 
                                    test_size = test_size, 
                                    random_state=seed)
    # print(train_patients, test_patients)
    # exit()
    train_df = data[data['病歷號'].isin(train_patients)]
    test_df = data[data['病歷號'].isin(test_patients)]
    return train_df, test_df

# df_data = split_by_gh(df_data,0)
# print(df_data)
# exit()
train_org_df, val_org_df = split_patients(df_data, 0.2)
# print(train_org_df, val_org_df)

train_df = pd.DataFrame(data= train_org_df, columns= ['gender','boneage','身高','GH','GnRHa','father_h','mother_h','BBW','after_month','grow'])
val_df = pd.DataFrame(data= val_org_df, columns= ['gender','boneage','身高','GH','GnRHa','father_h','mother_h','BBW','after_month','grow'])


train = train_org_df.drop(labels=['grow'],axis=1).values
test = val_org_df.drop(labels=['grow'],axis=1).values


X_train = train_df.drop(labels=['grow'],axis=1).values
y_train = train_df['grow'].values


X_test = val_df.drop(labels=['grow'],axis=1).values
y_test = val_df['grow'].values
# print(train_df['GnRHa'].value_counts())
print('train shape:', X_train.shape)
print('test shape:', X_test.shape)

# print(train_df.head(50))
# exit()
######################### train xgboost #####################################
import xgboost as xgb
from xgboost import XGBClassifier,XGBRegressor
from sklearn.ensemble import RandomForestRegressor
xgb.set_config(verbosity=0)                                                     ## 讓 XGBoost 只輸出錯誤和警告訊息就好

def train_xgboost(X_train, y_train, X_test, y_test, n, depth, lr):
    
    
    xgboostModel = XGBRegressor(n_estimators=n, learning_rate= lr, max_depth= depth)
    
    xgboostModel.fit(X_train, y_train)
    
    

    train_pred = xgboostModel.predict(X_train)
    test_pred = xgboostModel.predict(X_test)
    
    train_err = abs(train_pred-y_train)
    test_err = abs(test_pred-y_test)
    
    train_std = np.std(train_err)
    test_std = np.std(test_err)
    # xgboostModel.save_model('all_data_class0.json')

    return y_train, train_pred, y_test, test_pred, xgboostModel

def train_randomforest(X_train, y_train, X_test, y_test, n, depth):
    
    # 初始化 RandomForest 模型
    rf_model = RandomForestRegressor(n_estimators=n, max_depth=depth)
    
    # 訓練模型
    rf_model.fit(X_train, y_train)
    
    # 預測訓練集和測試集
    train_pred = rf_model.predict(X_train)
    test_pred = rf_model.predict(X_test)
    
    # 計算訓練和測試誤差
    train_err = abs(train_pred - y_train)
    test_err = abs(test_pred - y_test)
    
    # 計算誤差的標準差
    train_std = np.std(train_err)
    test_std = np.std(test_err)
    
    return y_train, train_pred, y_test, test_pred, rf_model
def mae(groundtruth, pred):  #groundtruth=y_train pred=train_pred
    err = 0.0
    for i in range(len(groundtruth)): #i=0
        
        err+=abs(float(groundtruth[i])-float(pred[i]))
    avg_err = err/float(len(groundtruth))
    
    return avg_err


def cal_MAE(month, y_test, test_pred, test):
    real_test=[]
    real_pred=[]
    
    for i in range(len(test)): #i=0
        # print(test[i][:])
        # exit()
        if test[i][10] < month:
            real_test.append(y_test[i])
            real_pred.append(test_pred[i])
                    
            
    test_err = abs(np.array(real_test)-np.array(real_pred))
    test_std = np.std(test_err)
    
    
    return  mae(real_test, real_pred), test_std

def cal_MAE2(month_l, month_u, y_test, test_pred, test):
    real_test=[]
    real_pred=[]
    
    for i in range(len(test)): #i=0
        # print(test[i][:])
        # exit()
        if test[i][10] < month_u and test[i][10] >= month_l:
            real_test.append(y_test[i])
            real_pred.append(test_pred[i])
                    
            
    test_err = abs(np.array(real_test)-np.array(real_pred))
    test_std = np.std(test_err)
    
    
    return  mae(real_test, real_pred), test_std

def cal_nFAH(y_test, test_pred, test):

    real_test=[]
    real_pred=[]
    
    val_org_df['yearold'] = [(float(i) + float(j) ) for i,j in zip(val_org_df['age'],val_org_df['after_month'])]

    
    df = pd.DataFrame(columns=[])
    # tt=[]
    
    val_org_df.reset_index(drop=True, inplace = True)
    
    
    for i in range(len(val_org_df)): #i=0
        # if val_org_df['gender'][i] == 1 and val_org_df['yearold'][i] > 191.0:
        if val_org_df['gender'][i] == 1 and val_org_df['yearold'][i] > 191.0 and val_org_df['age'][i] > 0.0 and val_org_df['age'][i] < 109.0:
            real_test.append(y_test[i])
            real_pred.append(test_pred[i])
            # df = df.append(val_org_df.iloc[i],ignore_index=True)
            if not val_org_df.iloc[i].isnull().all():
                df = pd.concat([df, pd.DataFrame([val_org_df.iloc[i]])], ignore_index=True)
        # elif val_org_df['gender'][i] == 0 and val_org_df['yearold'][i] > 167.0:
        elif val_org_df['gender'][i] == 0 and val_org_df['yearold'][i] > 167.0 and val_org_df['age'][i] > 0.0 and val_org_df['age'][i] < 109.0:
            real_test.append(y_test[i])
            real_pred.append(test_pred[i])
            # df = df.append(val_org_df.iloc[i],ignore_index=True)
            if not val_org_df.iloc[i].isnull().all():
                df = pd.concat([df, pd.DataFrame([val_org_df.iloc[i]])], ignore_index=True)
            
    
    test_err = abs(np.array(real_test)-np.array(real_pred))
    test_std = np.std(test_err)
    
    return  mae(real_test, real_pred), test_std


n_estimators = [int(x) for x in np.linspace(start=5, stop=30, num=14)]
max_depth = [int(x) for x in np.linspace(5, 30, num=14)]
learning_rate=[round(float(x),2) for x in np.linspace(start=0.08, stop=0.14, num=4)]

best_train_loss = 9999 
best_test_loss = 9999
best_train_loss2 = 9999 
best_test_loss2 = 9999
best_train_loss3 = 9999 
best_test_loss3 = 9999
best_train_loss4 = 9999 
best_test_loss4 = 9999
best_train_std = 0
best_test_std = 0



record = []
record2 = []
record3 = []
record4 =[]
record_str=''

for i in range(len(n_estimators)):
    
    for j in range(len(max_depth)):
        
        for k in range(len(learning_rate)):

            print( "Start Training -> ",  "n_estimators : " + str(n_estimators[i]) + ", max_depth : " + str(max_depth[j]) + ", learning_rate : " + str(learning_rate[k]))
            # print( "Start Training -> ",  "n_estimators : " + str(n_estimators[i]) + ", max_depth : " + str(max_depth[j]))
            # y_train, train_pred, y_test, test_pred, xgbmodel = train_randomforest(X_train, y_train, X_test, y_test, n_estimators[i], max_depth[j])
            y_train, train_pred, y_test, test_pred, xgbmodel = train_xgboost(X_train, y_train, X_test, y_test, n_estimators[i],max_depth[j],learning_rate[k])
            ################ feature selection ###################################################
            # feature_names = ['gender','boneage','身高','GH','GnRHa','father_h','mother_h','BBW','after_month']
            # # 取得模型的booster對象
            # booster = xgbmodel.get_booster()

            # # 使用booster的特徵順序來替換特徵名稱
            # booster.feature_names = feature_names

            # # 繪製特徵重要性圖，並使用正確的特徵名稱
            # xgb.plot_importance(booster, importance_type="weight", xlabel="Feature Importance", title="XGBoost Feature Importance")

            # plt.show()
            # exit()
            ####################################################################################
            # all data

            # train_loss, train_std = cal_MAE(100000, y_train, train_pred, train)
            # test_loss, test_std = cal_MAE(100000, y_test, test_pred, test)
            # record_str =  "train_loss: "+ str(round(train_loss,2))+ " ± "+ str(round(train_std,2)) + ' cm,    test_loss: '+ str(round(test_loss,2))+ " ± "+ str(round(test_std,2)) + ' cm'
            # print('-'*10 +'all data'+'-'*10)
            # print(record_str)            
            # if test_loss < best_test_loss:
            #     best_test_loss=test_loss
            #     xgbmodel.save_model('all_data_class0.json')
            #     print("***********n_estimators : " + str(n_estimators[i]) + ", max_depth : " + str(max_depth[j]) + ", learning_rate : " + str(learning_rate[k]))
    # print(best_test_loss)
            # 6 months
            train_loss, train_std = cal_MAE(7, y_train, train_pred, train)
            test_loss, test_std = cal_MAE(7, y_test, test_pred, test)
            # train_loss, train_std = cal_MAE2(0, 7, y_train, train_pred, train)
            # test_loss, test_std = cal_MAE2(0, 7, y_test, test_pred, test)
            record_str =  "train_loss: "+ str(round(train_loss,2))+ " ± "+ str(round(train_std,2)) + ' cm,    test_loss: '+ str(round(test_loss,2))+ " ± "+ str(round(test_std,2)) + ' cm'
            print('-'*10 +'6 months'+'-'*10)
            print( record_str)
            # 12 months

            train_loss2, train_std2 = cal_MAE(13, y_train, train_pred, train)
            test_loss2, test_std2 = cal_MAE(13, y_test, test_pred, test)
            # train_loss2, train_std2 = cal_MAE2(7, 13, y_train, train_pred, train)
            # test_loss2, test_std2 = cal_MAE2(7, 13, y_test, test_pred, test)
            record_str2 =  "train_loss: "+ str(round(train_loss2,2))+ " ± "+ str(round(train_std2,2)) + ' cm,    test_loss: '+ str(round(test_loss2,2))+ " ± "+ str(round(test_std2,2)) + ' cm'
            print('-'*10 +'12 months'+'-'*10)
            print( record_str2)
            # 24 months

            train_loss3, train_std3 = cal_MAE(25, y_train, train_pred, train)
            test_loss3, test_std3 = cal_MAE(25, y_test, test_pred, test)
            # train_loss3, train_std3 = cal_MAE2(13, 25, y_train, train_pred, train)
            # test_loss3, test_std3 = cal_MAE2(13, 25, y_test, test_pred, test)
            record_str3 =  "train_loss: "+ str(round(train_loss3,2))+ " ± "+ str(round(train_std3,2)) + ' cm,    test_loss: '+ str(round(test_loss3,2))+ " ± "+ str(round(test_std3,2)) + ' cm'
            print('-'*10 +'24 months'+'-'*10)
            print( record_str3)    
            # # nFAH

            # train_loss4, train_std4 = cal_nFAH(y_train, train_pred, train)
            # test_loss4, test_std4 = cal_nFAH(y_test, test_pred, test)
            # record_str4 =  "train_loss: "+ str(round(train_loss4,2))+ " ± "+ str(round(train_std4,2)) + ' cm,    test_loss: '+ str(round(test_loss4,2))+ " ± "+ str(round(test_std4,2)) + ' cm'
            # print('-'*10 +'nFAH'+'-'*10)
            # print( record_str4)     
            
            if test_loss < best_test_loss:
                best_test_loss=test_loss
                record.append( record_str + " / n_estimators : " + str(n_estimators[i]) + ", max_depth : " + str(max_depth[j]) + ", learning_rate : " + str(learning_rate[k]))
                # record.append( record_str + " / n_estimators : " + str(n_estimators[i]) + ", max_depth : " + str(max_depth[j]))

            if test_loss2 < best_test_loss2:
                best_test_loss2=test_loss2
                record2.append( record_str2 + " / n_estimators : " + str(n_estimators[i]) + ", max_depth : " + str(max_depth[j]) + ", learning_rate : " + str(learning_rate[k]))
                # record2.append( record_str2 + " / n_estimators : " + str(n_estimators[i]) + ", max_depth : " + str(max_depth[j]))

            if test_loss3 < best_test_loss3:
                best_test_loss3=test_loss3
                record3.append( record_str3 + " / n_estimators : " + str(n_estimators[i]) + ", max_depth : " + str(max_depth[j]) + ", learning_rate : " + str(learning_rate[k]))
                # record3.append( record_str3 + " / n_estimators : " + str(n_estimators[i]) + ", max_depth : " + str(max_depth[j]))

            # if test_loss4 < best_test_loss4:
            #     best_test_loss4=test_loss4
            #     record4.append( record_str4 + " / n_estimators : " + str(n_estimators[i]) + ", max_depth : " + str(max_depth[j]) + ", learning_rate : " + str(learning_rate[k]))

# file_path1 = "record1_data_parentweighted.txt"
# file_path2 = "record2_data_parentweighted.txt"
# file_path3 = "record3_data_parentweighted.txt"
# # file_path4 = "record4_data.txt"
# with open(file_path1, 'w') as file1:
#     for line in record:
#         file1.write(line + "\n")
# with open(file_path2, 'w') as file2:
#     for line in record2:
#         file2.write(line + "\n")

# with open(file_path3, 'w') as file3:
#     for line in record3:
#         file3.write(line + "\n")

# with open(file_path4, 'w') as file4:
#     for line in record4:
#         file4.write(line + "\n")

# xgboostModel.save_model("6-months Predict_yawen.json")