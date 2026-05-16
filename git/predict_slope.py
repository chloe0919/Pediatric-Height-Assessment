import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier,XGBRegressor
df = pd.read_csv("predict_longterm_samegt_timeslope_revision1219.csv", index_col=0)
df = df.dropna()
df = df[df['slope'] > 0]
# print(df)
# exit()
df['age'] = df['age'].map(lambda x: x if x < 217 else np.nan )
df = df.dropna(axis= 'index', how='any')
df.reset_index(drop=True, inplace = True)

df_data = pd.DataFrame(data= df,
                       columns= ['病歷號','gender','age','身高', '體重','父身高','母身高','boneage','用藥資訊','slope'])
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
# df_data['boneage'] = df_data['boneage'].map(lambda x: 216.0 if x > 216.0 else x )
df_data['bmi'] = [j / ( (float(i)/100.0) ** 2 ) for i,j in zip(df_data['身高'],df_data['體重'])]
# df_data['BBW'] = df_data['BBW'].apply(lambda x: x * 1000 if x < 10 else x)
# df_data['parents_h'] =  [(float(i) +float(j))/2.0 for i,j in zip(df_data['父身高'],df_data['母身高'])]
df_data.rename(columns={'父身高': 'father_h', '母身高': 'mother_h'}, inplace=True)

df_data[['GH', 'GnRHa']] = df_data.apply(assign_GH_GnRHa, axis=1)

#########################################################################################################################
def split_patients(data, test_size): #data = df_data
  
    unique_patient_ids = data['病歷號'].unique()
    
    # train_size = int(len(name_arr) * (1.0-test_size))
    # test_size = len(name_arr) - train_size
    
    train_patients, test_patients = train_test_split(unique_patient_ids, 
                                    test_size = test_size, 
                                    random_state=1999)
    # print(train_patients, test_patients)
    # exit()
    train_df = data[data['病歷號'].isin(train_patients)]
    test_df = data[data['病歷號'].isin(test_patients)]
    return train_df, test_df

# train_org_df, val_org_df = split_patients(df_data, 0.2)
df_data = df_data.sample(frac=1).reset_index(drop=True)
train_org_df, val_org_df = train_test_split(df_data, 
                                test_size = 0.2, 
                                random_state=42)
# print(train_org_df, val_org_df)

train_df = pd.DataFrame(data= train_org_df, columns= ['gender','age','boneage','身高','GH','GnRHa','father_h','mother_h','slope'])
val_df = pd.DataFrame(data= val_org_df, columns= ['gender','age','boneage','身高','GH','GnRHa','father_h','mother_h','slope'])

train = train_org_df.drop(labels=['slope'],axis=1).values
test = val_org_df.drop(labels=['slope'],axis=1).values


X_train = train_df.drop(labels=['slope'],axis=1).values
y_train = train_df['slope'].values


X_test = val_df.drop(labels=['slope'],axis=1).values
y_test = val_df['slope'].values

# print(train_df,val_df)
# exit()
print('train shape:', X_train.shape)
print('test shape:', X_test.shape)
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
            y_train, train_pred, y_test, test_pred, xgbmodel = train_xgboost(X_train, y_train, X_test, y_test, 26,8, 0.1)
            # y_train, train_pred, y_test, test_pred, xgbmodel = train_xgboost(X_train, y_train, X_test, y_test, n_estimators[i],max_depth[j], learning_rate[k])
            
            # # nFAH

            train_loss4, train_std4 = cal_MAE_all(y_train, train_pred, train)
            test_loss4, test_std4 = cal_MAE_all(y_test, test_pred, test)
            record_str4 =  "train_loss: "+ str(round(train_loss4,6))+ " ± "+ str(round(train_std4,6)) + ' cm,    test_loss: '+ str(round(test_loss4,6))+ " ± "+ str(round(test_std4,6)) + ' cm'
            print('-'*10 +'nFAH'+'-'*10) 
            print(record_str4)
            exit()
            xgbmodel.save_model(f"./time_slope/timeslope_revision1219_{n_estimators[i]}_{max_depth[j]}_{ learning_rate[k]}.json")

            if test_loss4 < best_test_loss4:
                best_test_loss4=test_loss4
                record4.append( record_str4 + " / n_estimators : " + str(n_estimators[i]) + ", max_depth : " + str(max_depth[j]) + ", learning_rate : " + str(learning_rate[k]))


# file_path4 = "predict_slope_revision1219.txt"

# with open(file_path4, 'w') as file4:
#     for line in record4:
#         file4.write(line + "\n")
#%%
# predict_t_model = XGBRegressor()
# predict_t_model.load_model("xgboost_predict_timeslope_slopepass0.json")
# zero_slope_df = df_data[df_data['slope'] == 0]
# X_to_predict = zero_slope_df[['gender','age','boneage','身高','GH','GnRHa','father_h','mother_h']]
# predicted_slope = predict_t_model.predict(X_to_predict)
# # print(predicted_slope)
# df_data.loc[df_data['slope'] == 0, 'slope'] = predicted_slope