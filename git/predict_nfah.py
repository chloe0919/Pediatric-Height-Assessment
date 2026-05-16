from xgboost import XGBClassifier,XGBRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
new_xgb = XGBRegressor()
new_xgb.load_model("xgboost_nfah_timeslpe_slopepass0_1.43.json")

# predict_df = pd.DataFrame({
#     'gender': [1, 0, 0],
#     'age': [96, 129, 134],
#     'boneage': [67.6618, 144.3429, 141.6373],
#     '身高': [107.9, 131.5, 134],
#     'GH' : [1,1,1],
#     'GnRHa' : [0,0,0],
#     'father_h' : [178, 160, 160],
#     'mother_h' : [153, 157, 157]
# })
predict_df = pd.read_csv('predict_longterm_samegt_timeslope2.csv',index_col=0)
predict_df = predict_df.reset_index(drop=True)

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
    


predict_df['growth_hormone'] = predict_df['用藥資訊'].map(map_growth_hormone)
predict_df.rename(columns={'父身高': 'father_h', '母身高': 'mother_h'}, inplace=True)
predict_df[['GH', 'GnRHa']] = predict_df.apply(assign_GH_GnRHa, axis=1)
predict_t_model = XGBRegressor()
predict_t_model.load_model("./time_slope/timeslope_26_8_0.1.json")
zero_slope_df = predict_df[predict_df['slope'] <= 0]
X_to_predict = zero_slope_df[['gender','age','boneage','身高','GH','GnRHa','father_h','mother_h']]
predicted_slope = predict_t_model.predict(X_to_predict)
predict_df.loc[predict_df['slope'] <= 0, 'slope'] = predicted_slope
#---------------------------------------------------------------------------------------------------
# print(predict_df)
# exit()
for i in range(len(predict_df)) :
    row = predict_df.iloc[i]
    # feature = np.array([row[['gender', 'pred_boneage', '身高','GH','GnRHa', 'parents_h', 'after_month']].to_numpy()])
    feature = np.array([row[['gender','age','boneage','身高','GH','GnRHa','father_h','mother_h','slope']].to_numpy()])
    # feature = np.array([row[['gender','pred_boneage','身高','GH','GnRHa','parents_h','after_age']].to_numpy()])
    # k = np.array([[0, 105.9699,125.4, 0.0, 0.0,158.5, 1]])
    
    predict = new_xgb.predict(feature)
    # print(predict_grow)
    # predict_height = new_xgb.predict(feature)
    predict_df.at[i, 'predict_H_nfan'] = predict

# print(predict_df)
# exit()
predict_df.to_csv("result_longterm_samegt_timeslope2_1.43.csv")

#### draw result mae (group by after_month) #################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
predict_df['absolute_error'] = abs(predict_df['predict_H_nfan'] - predict_df['nfan_h'])
print(predict_df['age'].max())
predict_df['age_year'] = predict_df['age'] // 12
grouped_error = predict_df.groupby('age_year')['absolute_error'].mean().reset_index()
print(grouped_error)
exit()

# 設定圖表
plt.figure(figsize=(10, 6))
plt.plot(grouped_error['age_year'], grouped_error['absolute_error'], 'o-', color='black', alpha=0.7, label='Absolute Mean Error')
plt.fill_between(grouped_error['age_year'], grouped_error['absolute_error'], alpha=0.1, color='gray')

# 設定 X 軸和 Y 軸
plt.xticks(np.arange(0, 25, 1))  # X 軸 0-25 每個數字
max_mae = grouped_error['absolute_error'].max()
y_max = int(np.ceil(max_mae))  # 找到 Y 軸最大整十數 
plt.yticks(np.arange(0, y_max + 1, 1))  # Y 軸從 0 到最大整十數

# 添加標題和標籤
plt.xlabel("Age")
plt.ylabel("Absolute Mean Error")
plt.legend()

# 顯示圖表
plt.tight_layout()
plt.show()
plt.savefig("longterm_mse_byage.png")
