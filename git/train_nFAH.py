# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBClassifier,XGBRegressor
from sklearn.ensemble import RandomForestRegressor
################ data distribution ###############################
# df = pd.read_csv('predict_longterm_samegt.csv',index_col=0)

# # 使用 pd.cut() 函數來將年齡劃分為區間
# bins = list(range(0, 18*12+1, 12))  # 0 到 18 歲，每 12 個月一個區間
# labels = [f'{i}-{i+1}歲' for i in range(0, 18)]  # 年齡區間的標籤

# df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

# # 計算每個年齡區間的筆數
# age_group_counts = df['age_group'].value_counts().sort_index()

# # 顯示結果
# print(age_group_counts)
# exit()
################ data preparcessing ###############################
age_df = pd.read_csv('predict_longterm_samegt_timeslope_revision1219.csv')
age_df['age'] = age_df['age'].map(lambda x: x if x < 217 else np.nan )
age_df = age_df.dropna(axis= 'index', how='any')
age_df.reset_index(drop=True, inplace = True)

df_data = pd.DataFrame(data= age_df,
                       columns= ['病歷號','gender','age','身高', '體重','父身高','母身高','boneage','用藥資訊','slope','nfah_h'])
# df_data = pd.DataFrame(data= age_df,
#                        columns= ['病歷號','gender','age','身高', '體重','父身高','母身高','boneage','用藥資訊','BBW','nfan_h'])
# tentofif = df_data[(df_data['age'] <= 180) & (df_data['age'] >= 120)]
# df_data = pd.concat([df_data, tentofif]).drop_duplicates(keep=False)
# print(df_data)
# exit()
# print(df_data)
# exit()
def map_growth_hormone(info):
    if pd.isnull(info):
        return None  # 如果是缺失值，返回 None
    codes = info.split(',')  # 根据逗号分隔
    total = 0  # 初始化总和
    for code in codes:
        code = code.strip()  # 去掉多余的空格
        if code in ['907452', '525953', '526455']:  # GnRHa
            total += -1
        elif code in ['407756', '407751', '827954', '407952']:  # GH
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
# df_data['boneage'] = df_data['boneage'].map(lambda x: 216.0 if x > 216.0 else x )
df_data['bmi'] = [j / ( (float(i)/100.0) ** 2 ) for i,j in zip(df_data['身高'],df_data['體重'])]
# df_data['BBW'] = df_data['BBW'].apply(lambda x: x * 1000 if x < 10 else x)
# df_data['parents_h'] =  [(float(i) +float(j))/2.0 for i,j in zip(df_data['父身高'],df_data['母身高'])]
df_data.rename(columns={'父身高': 'father_h', '母身高': 'mother_h'}, inplace=True)
# df_data['growth_hormone'] = df_data['用藥資訊'].map(lambda x: -1 if (( '907452' in x ) or ( '525953' in x ) or ( '526455' in x )) else ( 1 if (( '407756' in x ) or ( '407751' in x ) or ( '827954' in x ) or ( '407952' in x )) else 0 ))
# df_data['bone_err'] = [float(i) - float(j) for i,j in zip(df_data['pred_boneage'],df_data['boneage'])]
# df_data['GH'] = df_data['growth_hormone'].map(lambda x: x if x==1 else 0 )
# df_data['GnRHa'] = df_data['growth_hormone'].map(lambda x: 1.0 if x==-1.0 else 0.0)
# df_data['grow'] =  [(float(i) -float(j)) for i,j in zip(df_data['after_H'],df_data['身高'])]
df_data[['GH', 'GnRHa']] = df_data.apply(assign_GH_GnRHa, axis=1)
# print(df_data)
# exit()
# print(df_data[(df_data['GH'] == 1) & (df_data['GnRHa'] == 1)])
# exit()
# df_data['month_category'] = pd.cut(df_data['after_month'], 10)
#----------------------  predict timeslope ------------------------------------------------------
# predict_t_model = XGBRegressor()
# predict_t_model.load_model("./time_slope/timeslope_revision1219_26_5_0.1.json")
# zero_slope_df = df_data[df_data['slope'] <= 0]
# X_to_predict = zero_slope_df[['gender','age','boneage','身高','GH','GnRHa','father_h','mother_h']]
# predicted_slope = predict_t_model.predict(X_to_predict)
# df_data.loc[df_data['slope'] <= 0, 'slope'] = predicted_slope
#--------------------------------------------------------------------------------------------------

# # # 將slope欄位的值替換為對應的中位數
# # df_data['slope'] = df_data['slope'].apply(replace_with_median)
# # print(df_data['slope'].value_counts())
# # exit()
#----------------------  predict timeslope ------------------------------------------------------
# import os
# import glob
# path = "./time_slope"
# output = "./nfah_predict_output"
# dir = os.listdir("./time_slope")
# file_pattern = os.path.join(path, "timeslope_revision1219*")
# all_model =[]

# for file in glob.glob(file_pattern):
#     tmp_df = df_data.copy()
#     ts_file = os.path.basename(file)
#     filename = os.path.join(path, ts_file)
#     print(f"############### model {filename} start #########################")


#     predict_t_model = XGBRegressor()
#     predict_t_model.load_model(f"{filename}")
#     # predict_t_model.load_model("./time_slope/timeslope_12_26_0.08.json")
#     zero_slope_df = tmp_df[tmp_df['slope'] <= 0]
#     X_to_predict = zero_slope_df[['gender','age','boneage','身高','GH','GnRHa','father_h','mother_h']]
#     predicted_slope = predict_t_model.predict(X_to_predict)
#     tmp_df.loc[tmp_df['slope'] <= 0, 'slope'] = predicted_slope
#     # print(predicted_slope)

# #     #------------------------------------------------------------------------------------------------

# #     # print(df_data[(df_data['after_month'] < 7)])
# #     # print(df_data[(df_data['after_month'] < 13) & (df_data['after_month'] > 6)])
# #     # print(df_data[(df_data['after_month'] < 25) & (df_data['after_month'] > 12)])
# #     # print(df_data)
# #     # exit()
#     ################## data split to train and val ######################
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
# train_org_df, val_org_df = split_patients(df_data, 0.2)

tmp_df = df_data.sample(frac=1, random_state=seed).reset_index(drop=True)
train_org_df, val_org_df = train_test_split(df_data,
                                test_size = 0.2, 
                                random_state=50)
# print(train_org_df, val_org_df)

train_df = pd.DataFrame(data= train_org_df, columns= ['gender','age','boneage','身高','GH','GnRHa','father_h','mother_h','slope','nfah_h'])
val_df = pd.DataFrame(data= val_org_df, columns= ['gender','age','boneage','身高','GH','GnRHa','father_h','mother_h','slope','nfah_h'])

# train_df = pd.DataFrame(data= train_org_df, columns= ['gender','age','boneage','身高','GH','GnRHa','father_h','mother_h','nfah_h'])
# val_df = pd.DataFrame(data= val_org_df, columns= ['gender','age','boneage','身高','GH','GnRHa','father_h','mother_h','nfah_h'])

train = train_org_df.drop(labels=['nfah_h'],axis=1).values
test = val_org_df.drop(labels=['nfah_h'],axis=1).values


X_train = train_df.drop(labels=['nfah_h'],axis=1).values
y_train = train_df['nfah_h'].values


X_test = val_df.drop(labels=['nfah_h'],axis=1).values
y_test = val_df['nfah_h'].values
# print(train_df['GnRHa'].value_counts())
# print(train_org_df,val_org_df)
print('train shape:', X_train.shape)
print('test shape:', X_test.shape)
# exit()
######################### train xgboost #####################################
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
    rf_model = RandomForestRegressor(n_estimators=n, max_depth=depth,random_state=seed)
    
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

def cal_MAE_all(y_test, test_pred, test):
    real_test=[]
    real_pred=[]
    
    for i in range(len(test)): #i=0
        # print(test[i][:])
        # exit()
        # if test[i][10] < month:
        real_test.append(y_test[i])
        real_pred.append(test_pred[i])
                    
            
    test_err = abs(np.array(real_test)-np.array(real_pred))
    test_std = np.std(test_err)
    
    
    return  mae(real_test, real_pred), test_std

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

############################# xgbooost ###################################################################
# for i in range(len(n_estimators)):
    
#     for j in range(len(max_depth)):
        
#         for k in range(len(learning_rate)):

#             print( "Start Training -> ",  "n_estimators : " + str(n_estimators[i]) + ", max_depth : " + str(max_depth[j]) + ", learning_rate : " + str(learning_rate[k]))
#             # print( "Start Training -> ",  "n_estimators : " + str(n_estimators[i]) + ", max_depth : " + str(max_depth[j]))
#             # y_train, train_pred, y_test, test_pred, xgbmodel = train_randomforest(X_train, y_train, X_test, y_test, n_estimators[i], max_depth[j])
#             y_train, train_pred, y_test, test_pred, xgbmodel = train_xgboost(X_train, y_train, X_test, y_test, n_estimators[i], max_depth[j], learning_rate[k])
#             # y_train, train_pred, y_test, test_pred, xgbmodel = train_xgboost(X_train, y_train, X_test, y_test, 30,5, 0.12)
            
#             # # nFAH

#             train_loss4, train_std4 = cal_MAE_all(y_train, train_pred, train)
#             test_loss4, test_std4 = cal_MAE_all(y_test, test_pred, test)
#             record_str4 =  "train_loss: "+ str(round(train_loss4,2))+ " ± "+ str(round(train_std4,2)) + ' cm,    test_loss: '+ str(round(test_loss4,2))+ " ± "+ str(round(test_std4,2)) + ' cm'
#             print('-'*10 +'nFAH'+'-'*10)
#             print( record_str4)    
#             # print(outputpath)
#             # exit()
#             # xgbmodel.save_model("xgboost_nfah_timeslpe_slopepass0_1.43.json")

#             if test_loss4 < best_test_loss4:
#                 best_test_loss4=test_loss4
#                 record4.append( record_str4 + " / n_estimators : " + str(n_estimators[i]) + ", max_depth : " + str(max_depth[j]) + ", learning_rate : " + str(learning_rate[k]))
################################ rf ##########################################################################
for i in range(len(n_estimators)):
    
    for j in range(len(max_depth)):

        print( "Start Training -> ",  "n_estimators : " + str(n_estimators[i]) + ", max_depth : " + str(max_depth[j]))
        # print( "Start Training -> ",  "n_estimators : " + str(n_estimators[i]) + ", max_depth : " + str(max_depth[j]))
        # y_train, train_pred, y_test, test_pred, xgbmodel = train_randomforest(X_train, y_train, X_test, y_test, n_estimators[i], max_depth[j])
        y_train, train_pred, y_test, test_pred, xgbmodel = train_randomforest(X_train, y_train, X_test, y_test, 26, 20)
        
        # # nFAH

        train_loss4, train_std4 = cal_MAE_all(y_train, train_pred, train)
        test_loss4, test_std4 = cal_MAE_all(y_test, test_pred, test)
        record_str4 =  "train_loss: "+ str(round(train_loss4,2))+ " ± "+ str(round(train_std4,2)) + ' cm,    test_loss: '+ str(round(test_loss4,2))+ " ± "+ str(round(test_std4,2)) + ' cm'
        print('-'*10 +'nFAH'+'-'*10)
        print( record_str4)   
        exit()
        # print(outputpath)
        # exit()
        # xgbmodel.save_model("xgboost_nfah_timeslpe_slopepass0_1.43.json")

        if test_loss4 < best_test_loss4:
            best_test_loss4=test_loss4
            record4.append( record_str4 + " / n_estimators : " + str(n_estimators[i]) + ", max_depth : " + str(max_depth[j]))

    # outputpath = os.path.join(output, f"record4_{ts_file[:-5]}.txt")

# file_path4 = f"./rf_revision1219_slope_seed50.txt"

# with open(file_path4, 'w') as file4:
#     for line in record4:
#         file4.write(line + "\n")
# exit()
# 00000000000000000000000000000000000000000000000000000000000000000000000000000000000000
#     record_str4 = ""
#     train_loss4 = ""
#     train_std4 = ""
#     test_loss4 = ""
#     test_std4 = ""

#     y_train, train_pred, y_test, test_pred, xgbmodel = train_xgboost(X_train, y_train, X_test, y_test, 30,6, 0.12)
#     train_loss4, train_std4 = cal_MAE_all(y_train, train_pred, train)
#     test_loss4, test_std4 = cal_MAE_all(y_test, test_pred, test)
#     record_str4 =  "train_loss: "+ str(round(train_loss4,2))+ " ± "+ str(round(train_std4,2)) + ' cm,    test_loss: '+ str(round(test_loss4,2))+ " ± "+ str(round(test_std4,2)) + ' cm'
#     print(record_str4)
#     all_model.append(filename+" "+record_str4)
#     # exit()
# outputpath = os.path.join(output, "allmodel_30_6_0.12.txt")
# with open(outputpath, 'w') as file4:
#     for line in all_model:
#         file4.write(line + "\n")
#%%
# from xgboost import XGBClassifier,XGBRegressor
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy import stats
# new_xgb = XGBRegressor()
# new_xgb.load_model("xgboost_nfah_timeslpe_slopepass0_1.43.json")

# # predict_df = pd.DataFrame({
# #     'gender': [1, 0, 0],
# #     'age': [96, 129, 134],
# #     'boneage': [67.6618, 144.3429, 141.6373],
# #     '身高': [107.9, 131.5, 134],
# #     'GH' : [1,1,1],
# #     'GnRHa' : [0,0,0],
# #     'father_h' : [178, 160, 160],
# #     'mother_h' : [153, 157, 157]
# # })
# predict_df = pd.read_csv('predict_longterm_samegt_timeslope2.csv',index_col=0)
# predict_df = predict_df.reset_index(drop=True)

# def map_growth_hormone(info):
#     if pd.isnull(info):
#         return None  # 如果是缺失值，返回 None
#     codes = info.split(',')  # 根据逗号分隔
#     total = 0  # 初始化总和
#     for code in codes:
#         code = code.strip()  # 去掉多余的空格
#         if code in ['907452', '525953', '526455']:  # 代表 -1
#             total += -1
#         elif code in ['407756', '407751', '827954', '407952']:  # 代表 1
#             total += 1
#     if total > 1 :
#         return 1
#     elif total < -1 :
#         return -1
#     elif total == 0 and len(codes) >= 2 :
#         return 0
#     elif total == 0 and len(codes) == 1 :
#         return 2
#     else:
#         return total
# def assign_GH_GnRHa(row):
#     if row['growth_hormone'] == 2:
#         return pd.Series([0, 0])  # GH = 0, GnRHa = 0
#     elif row['growth_hormone'] == 1:
#         return pd.Series([1, 0])  # GH = 1, GnRHa = 0
#     elif row['growth_hormone'] == -1:
#         return pd.Series([0, 1])  # GH = 0, GnRHa = 1
#     elif row['growth_hormone'] == 0:
#         return pd.Series([1, 1])  # GH = 1, GnRHa = 1
#     else:
#         return pd.Series([None, None])  # 防止有其他未知值
    


# predict_df['growth_hormone'] = predict_df['用藥資訊'].map(map_growth_hormone)
# predict_df.rename(columns={'父身高': 'father_h', '母身高': 'mother_h'}, inplace=True)
# predict_df[['GH', 'GnRHa']] = predict_df.apply(assign_GH_GnRHa, axis=1)
# #-------------------------------------------------------------------------------------------------
# predict_t_model = XGBRegressor()
# predict_t_model.load_model("./time_slope/timeslope_26_8_0.1.json")
# zero_slope_df = predict_df[predict_df['slope'] <= 0]
# X_to_predict = zero_slope_df[['gender','age','boneage','身高','GH','GnRHa','father_h','mother_h']]
# predicted_slope = predict_t_model.predict(X_to_predict)
# predict_df.loc[predict_df['slope'] <= 0, 'slope'] = predicted_slope
# #---------------------------------------------------------------------------------------------------
# # print(predict_df)
# # exit()
# for i in range(len(predict_df)) :
#     row = predict_df.iloc[i]
#     # feature = np.array([row[['gender', 'pred_boneage', '身高','GH','GnRHa', 'parents_h', 'after_month']].to_numpy()])
#     feature = np.array([row[['gender','age','boneage','身高','GH','GnRHa','father_h','mother_h','slope']].to_numpy()])
#     # feature = np.array([row[['gender','pred_boneage','身高','GH','GnRHa','parents_h','after_age']].to_numpy()])
#     # k = np.array([[0, 105.9699,125.4, 0.0, 0.0,158.5, 1]])
    
#     predict = new_xgb.predict(feature)
#     # print(predict_grow)
#     # predict_height = new_xgb.predict(feature)
#     predict_df.at[i, 'predict_H_nfan'] = predict

# # print(predict_df)
# # exit()
# predict_df.to_csv("result_longterm_samegt_timeslope2_1.43.csv")
# %%
