from xgboost import XGBClassifier,XGBRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

new_xgb6m = XGBRegressor()
new_xgb6m.load_model("xgboost_6m_all_data.json")
new_xgb1y = XGBRegressor()
new_xgb1y.load_model("xgboost_1y_all_data.json")
new_xgb2y = XGBRegressor()
new_xgb2y.load_model("xgboost_2y_all_data.json")
################################################################################
# importance = new_xgbclass2.get_booster().get_score(importance_type='gain')
# feature_names = ['gender', 'pred_boneage', '身高', 'parents_h', 'after_month']
# feature_map = {f"f{i}": feature_names[i] for i in range(len(feature_names))}

# importance_named = {feature_map[k]: v for k, v in importance.items()}

# importance_df = pd.DataFrame(importance_named.items(), columns=['Feature', 'Importance'])

# importance_df = importance_df.sort_values(by='Importance', ascending=False)

# print(importance_df)

# importance_df.plot(kind='barh', x='Feature', y='Importance', legend=False)
# plt.show()
# exit()
################################################################################
predict_df = pd.read_csv('predict_shortterm.csv',index_col=0)

predict_df['age'] = predict_df['age'].map(lambda x: x if x < 217 else np.nan )
predict_df = predict_df.dropna(axis= 'index', how='any')
predict_df.reset_index(drop=True, inplace = True)

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
# print(df_data['growth_hormone'].value_counts())
# print(df_data[df_data['用藥資訊'] == '0'])
# exit()
predict_df['boneage'] = predict_df['boneage'].map(lambda x: 216.0 if x > 216.0 else x )
predict_df['bmi'] = [j / ( (float(i)/100.0) ** 2 ) for i,j in zip(predict_df['身高'],predict_df['體重'])]
predict_df['BBW'] = predict_df['BBW'].apply(lambda x: x * 1000 if x < 10 else x)
# df_data['parents_h'] =  [(float(i) +float(j))/2.0 for i,j in zip(df_data['父身高'],df_data['母身高'])]
predict_df.rename(columns={'父身高': 'father_h', '母身高': 'mother_h'}, inplace=True)
# predict_df['grow'] =  [(float(i) -float(j)) for i,j in zip(predict_df['after_H'],predict_df['身高'])]
predict_df[['GH', 'GnRHa']] = predict_df.apply(assign_GH_GnRHa, axis=1)
# predict_df['month_category'] = pd.cut(predict_df['after_month'], 10)
# output_data = pd.DataFrame(data= predict_df,
#                        columns= ['病歷號','gender','age','身高', '體重','父身高','母身高','boneage','用藥資訊','BBW','after_month','after_H'])


output_data = predict_df.copy()
output_data = output_data[(output_data['after_month'] >= 0) & (output_data['after_month'] < 25)]
output_data = output_data.reset_index(drop=True)

for i in range(len(output_data)):
    row = output_data.iloc[i]
    
    # 將該行數據轉換為特徵數組
    feature = row[['gender','boneage','身高','GH','GnRHa','father_h','mother_h','BBW','after_month']].to_numpy().reshape(1, -1)
    # 根據GH和GnRHa的值選擇模型進行預測
    if row['after_month'] < 7:
        predict_grow = new_xgb6m.predict(feature)
        # predict_grow1 = new_xgbclass2.predict(feature)
    elif row['after_month'] < 13 and row['after_month'] >= 7:
        predict_grow = new_xgb1y.predict(feature)
        # predict_grow1 = new_xgbclass2.predict(feature)
    elif row['after_month'] < 25 and row['after_month'] >= 13:
        predict_grow = new_xgb2y.predict(feature)
        # predict_grow1 = new_xgbclass2.predict(feature)
    else:
        predict_grow = None  # 如果不符合上述條件

    # 將預測結果存入新的欄位
    predict_height = feature[0][2] + predict_grow
    # predict_height1 = feature[0][2] + predict_grow1
    output_data.at[i, 'predict_H'] = predict_height
    # output_data.at[i, 'predict_H1'] = predict_height1
print(output_data)
output_data.to_csv("result_shortterm_under2y.csv")

#### draw result mae (group by after_month) #################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
output_data['absolute_error'] = abs(output_data['predict_H'] - output_data['after_H'])

grouped_error = output_data.groupby('after_month')['absolute_error'].mean().reset_index()
# 設定圖表
plt.figure(figsize=(10, 6))
plt.plot(grouped_error['after_month'], grouped_error['absolute_error'], 'o-', color='black', alpha=0.7, label='Absolute Mean Error')
plt.fill_between(grouped_error['after_month'], grouped_error['absolute_error'], alpha=0.1, color='gray')

# 設定 X 軸和 Y 軸
plt.xticks(np.arange(0, 25, 1))  # X 軸 0-25 每個數字
max_mae = grouped_error['absolute_error'].max()
y_max = int(np.ceil(max_mae))  # 找到 Y 軸最大整十數 
plt.yticks(np.arange(0, y_max + 1, 1))  # Y 軸從 0 到最大整十數

# 添加標題和標籤
plt.xlabel("Months Under 2 Years")
plt.ylabel("Absolute Mean Error")
plt.legend()

# 顯示圖表
plt.tight_layout()
plt.show()
plt.savefig("shortterm_mae_under2y.png")






#### find outliers #######################################################
# output_data = output_data.drop_duplicates()
# output_data = output_data.groupby(['病歷號', 'gender', '身高', 'boneage', 'mother_h', 'father_h', '門診日', 'BBW', 'after_month', 'GH', 'GnRHa'], as_index=False).agg({
#     'after_H': 'mean',
#     'age': 'first',
#     '體重': 'first',
#     'pred_boneage': 'first',
#     '生日': 'first',
#     '用藥資訊': 'first',
#     'growth_hormone': 'first',
#     'bmi': 'first'
# })
# output_data['error'] = output_data['predict_H'] - output_data['after_H']
# mean = output_data['error'].mean()
# std = output_data['error'].std()
# lower_threshold = mean - 2 * std
# upper_threshold = mean + 2 * std
# output_data['Unexpected_tall'] = (output_data['error'] > upper_threshold)
# output_data['Unexpected_short'] = (output_data['error'] < lower_threshold)
# # print(output_data[output_data['Unexpected_short'] == True])
# # print(output_data[output_data['Unexpected_tall'] == True])
# # print(output_data[(output_data['Unexpected_tall'] == False) & (output_data['Unexpected_short'] == False)])
# # exit()
# output_data.to_csv("outlier_10mto14m.csv")
####################################################################
# from scipy import stats
# t_statistic, p_value = stats.ttest_ind(output_data['predict_H'], output_data['predict_H1'])
# print(f"T-statistic: {t_statistic}")
# print(f"P-value: {p_value}")
####################################################################
# output_data.to_csv("output_height.csv")