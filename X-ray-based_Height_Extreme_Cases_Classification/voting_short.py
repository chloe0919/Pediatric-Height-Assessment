import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torchvision import transforms
import os
import math
from torch.utils.data import Dataset, DataLoader
import cv2
import pandas as pd
import glob
import random
#from age_predictor_model import Bottleneck, AgePredictor
from tqdm import tqdm
import scipy.io as io
import torch.nn.functional as F
import copy
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statistics
from sklearn.model_selection import train_test_split
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc, confusion_matrix
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torchvision import models

df_data = pd.read_csv("height_1y_60days.csv",index_col=0)
df_data = df_data.dropna()
# print(df_data['age'].value_counts())
# exit()
# ## 保留長的比平均矮的
# df_data = df_data[df_data['height_1y'] < df_data['height_1y_avg']]
dataset_path = 'new_data/'


df_data['path'] = df_data['醫令名稱'].map(lambda x: os.path.join(dataset_path, 
                                                       '{}.png'.format(x)))

df_data['exists'] = df_data['path'].map(os.path.exists)
df_data = df_data.dropna(axis= 'index', how='any')
df_data.reset_index(drop=True, inplace = True)
# 計算每個年齡層的 height_1y 平均值和標準差
df_data['mean_height'] = df_data.groupby('age')['height_1y'].transform('mean')
df_data['std_height'] = df_data.groupby('age')['height_1y'].transform('std')

df_data['unexpected_short'] = (df_data['height_1y'] < (df_data['mean_height'] - 1.5 * df_data['std_height'])).astype(int)
df_data['unexpected_tall'] = (df_data['height_1y'] > (df_data['mean_height'] + 1.5 * df_data['std_height'])).astype(int)
df_data = df_data.drop(columns=['height_1y_avg','mean_height', 'std_height'])

# df_data = df_data[df_data['unexpected_tall'] == 0] ### 挑出正常和太矮的
# df_data = df_data.drop(columns=['unexpected_tall'])
# df_data.rename(columns={'unexpected_short': 'gt'}, inplace=True)
df_data['gt'] = df_data['unexpected_short'].map(lambda x: 1 if x==1 else 0)
# print(df_data['age'].value_counts())
# exit()
# df_data = df_data[df_data['unexpected_short'] == False] ### 挑出正常和太高的
# df_data = df_data.drop(columns=['unexpected_short'])
# print(df_data['unexpected_short'].value_counts())
# exit()
###########################################################################################
from sklearn.model_selection import train_test_split
# df_data = df_data[df_data['gt'] != 2]
seed = 42
label_path = 'new_label_data/'
image_name=[file[:-4] for file in os.listdir(dataset_path)]
img_size = 512
# print(df_data['gt'].value_counts())
n_classes = 10
gt_0_data = df_data[df_data['gt'] == 0].sample(frac=1, random_state=42)
gt_0_classes = np.array_split(gt_0_data, n_classes)

# gt_0_data = df_data[df_data['gt'] == 0]
# grouped_by_age = gt_0_data.groupby('age')
# gt_0_classes = [None] * n_classes
# for age, group in grouped_by_age:
#     subarrays = np.array_split(group, n_classes)
#     for i in range(n_classes):
#         if gt_0_classes[i] is None:
#             # 如果還沒有合併過，直接將 subarray 設為這個子 DataFrame
#             gt_0_classes[i] = subarrays[i]
#         else:
#             # 否則將對應的子 DataFrame 合併到已經合併的結果中
#             gt_0_classes[i] = pd.concat([gt_0_classes[i], subarrays[i]], ignore_index=True)
n = 0
validation_df = []
prev_val_positive = None
for gt_0_class in gt_0_classes:
    n+=1
    other_classes = df_data[df_data['gt'] != 0]
    combined_data = pd.concat([gt_0_class, other_classes])
    subdf = combined_data.copy()
    # print(subdf[subdf['gt'] == 0])
    # exit()
    train_org_df, val_org_df = train_test_split(subdf, 
                                    test_size = 0.2, 
                                    random_state=seed,
                                    stratify = subdf['gt'])#df_data['outliers'])
    validation_df.append(val_org_df)
    ####-------------- check validation same -----------------------------------
    # current_val_positive = val_org_df[val_org_df['gt'] == 1]
    # if n > 1 and prev_val_positive is not None:
    #     is_same = current_val_positive.sort_values(by=list(current_val_positive.columns)).reset_index(drop=True).equals(
    #         prev_val_positive.sort_values(by=list(prev_val_positive.columns)).reset_index(drop=True)
    #     )
    #     print(f"Iteration {n}: Positive samples are {'the same' if is_same else 'different'} as previous iteration")
    
    # # 更新前一次迭代的結果
    # prev_val_positive = current_val_positive.copy()
    ##---------------------------------------------------------------------------
validation_df = pd.concat(validation_df, ignore_index=True)
validation_df = validation_df.drop_duplicates(keep='first')
# print(validation_df['gt'].value_counts())
# exit()
# print(validation_df)
# exit()
#########################################################################
class BonesDataset(Dataset):
    def __init__(self,dataframe, image_dir, transform):
        self.dataframe = dataframe
        self.image_dir = dataset_path
        self.transform = transform
        self.label_image_dir = label_path
    def __getitem__(self, index):
        img_name = self.image_dir + str(self.dataframe.iloc[index]['醫令名稱']) + '.png'
        label_name = self.label_image_dir + str(self.dataframe.iloc[index]['醫令名稱']) + '.png'
        img = Image.open(img_name).convert('RGB')
        img = img.resize((img_size,img_size))
        image_array = np.array(img)

        ## Unet ##
        label = Image.open(label_name).convert('RGB')
        label = label.resize((img_size,img_size))
        label_array = np.array(label)
        label_array[label_array == 255] = 1

        seg_array = image_array*label_array
        seg_array = np.dot(seg_array[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        seg = Image.fromarray(seg_array)
        
        ## CLAHE ##
        clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(12, 12))
        clahe_mask_array = clahe.apply(seg_array)
        clahe_mask = Image.fromarray(clahe_mask_array)

        
        ## Threshold ##
        clahe_mask_thre_array = clahe_mask_array.copy()                             ## = clahe_mask 會直接改到 clahe_mask 的值
        clahe_mask_thre_array[clahe_mask_thre_array < 180] = 0
        clahe_mask_thre = Image.fromarray(clahe_mask_thre_array)

        ## concat ##
        merge = Image.merge( 'RGB', (seg, clahe_mask, clahe_mask_thre))
        ## feature ##
        # if img_name == 'new_data/BONEAGE_P2_10956.png' :
        #     print(img_name)
        #     merge.show()
        #     exit()
        gender = np.atleast_1d(self.dataframe.iloc[index]['gender'])
        age = np.atleast_1d(self.dataframe.iloc[index]['age'])
        gt = np.atleast_1d(self.dataframe.iloc[index]['gt'])

        if self.transform is not None:
            merge = self.transform(merge)
        
        # print(merge.dtype,gender.dtype,bone_age.dtype)
        sample = {'name' : img_name, 'image': merge, 'gender': torch.from_numpy(gender).long(),'age': torch.from_numpy(age).float(),
                    'gt': torch.from_numpy(gt).long()}
        return sample
    
    def __len__(self):
        return len(self.dataframe)
    
# train_transform = transforms.Compose([
#     # transforms.Pad(padding=60),
#     transforms.RandomAffine(degrees=(-20,20), scale=(0.9, 1.2), translate = (0, 0.1)),
#     transforms.ColorJitter(brightness=(0.6, 1.4)),
#     transforms.Resize((512, 512)),
    
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ]) 
valid_transform = transforms.Compose([
    
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]) 

# train_dataset = BonesDataset(dataframe = train_org_df, image_dir=dataset_path, transform = train_transform)
val_dataset = BonesDataset(dataframe = validation_df, image_dir = dataset_path, transform = valid_transform)
# train_data_loader = DataLoader(train_dataset,batch_size=128,shuffle=False,drop_last=True)
val_data_loader = DataLoader(val_dataset,batch_size=128,shuffle=False)
######################## model #######################################################################################
import torch
import torch.nn as nn
#from age_predictor_model import Bottleneck, AgePredictor
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error
# from model import attention_module as block

class BN_Conv2d(nn.Module):
    """
    BN_CONV, default activation is ReLU
    """

    def __init__(self, in_channels: object, out_channels: object, kernel_size: object, stride: object, padding: object,
                  dilation=1, groups=1, bias=False, activation=True) -> object:
        super(BN_Conv2d, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation, groups=groups, bias=bias),
                  nn.BatchNorm2d(out_channels)]
        if activation:
            layers.append(nn.ReLU(inplace=False))
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)
class Stem_v4_Res2(nn.Module):
    """
    stem block for Inception-v4 and Inception-RestNet-v2
    """

    def __init__(self):
        super(Stem_v4_Res2, self).__init__()
        self.step1 = nn.Sequential(
            BN_Conv2d(3, 32, 3, 2, 0, bias=False),
            BN_Conv2d(32, 32, 3, 1, 0, bias=False),
            BN_Conv2d(32, 64, 3, 1, 1, bias=False)
        )
        self.step2_pool = nn.MaxPool2d(3, 2, 0)
        self.step2_conv = BN_Conv2d(64, 96, 3, 2, 0, bias=False)
        self.step3_1 = nn.Sequential(
            BN_Conv2d(160, 64, 1, 1, 0, bias=False),
            BN_Conv2d(64, 96, 3, 1, 0, bias=False)
        )
        self.step3_2 = nn.Sequential(
            BN_Conv2d(160, 64, 1, 1, 0, bias=False),
            BN_Conv2d(64, 64, (7, 1), (1, 1), (3, 0), bias=False),
            BN_Conv2d(64, 64, (1, 7), (1, 1), (0, 3), bias=False),
            BN_Conv2d(64, 96, 3, 1, 0, bias=False)
        )
        self.step4_pool = nn.MaxPool2d(3, 2, 0)
        self.step4_conv = BN_Conv2d(192, 192, 3, 2, 0, bias=False)

    def forward(self, x):
        out = self.step1(x)
        tmp1 = self.step2_pool(out)
        tmp2 = self.step2_conv(out)
        out = torch.cat((tmp1, tmp2), 1)
        tmp1 = self.step3_1(out)
        tmp2 = self.step3_2(out)
        out = torch.cat((tmp1, tmp2), 1)
        tmp1 = self.step4_pool(out)
        tmp2 = self.step4_conv(out)
        # print(tmp1.shape)
        # print(tmp2.shape)
        out = torch.cat((tmp1, tmp2), 1)
        return out
class Inception_A_res(nn.Module):
    """
    Inception-A block for Inception-ResNet-v1\
    and Inception-ResNet-v2 net
    """

    def __init__(self, in_channels, b1, b2_n1, b2_n3, b3_n1, b3_n3_1, b3_n3_2, n1_linear):
        super(Inception_A_res, self).__init__()
        self.branch1 = BN_Conv2d(in_channels, b1, 1, 1, 0, bias=False)
        self.branch2 = nn.Sequential(
            BN_Conv2d(in_channels, b2_n1, 1, 1, 0, bias=False),
            BN_Conv2d(b2_n1, b2_n3, 3, 1, 1, bias=False),
        )
        self.branch3 = nn.Sequential(
            BN_Conv2d(in_channels, b3_n1, 1, 1, 0, bias=False),
            BN_Conv2d(b3_n1, b3_n3_1, 3, 1, 1, bias=False),
            BN_Conv2d(b3_n3_1, b3_n3_2, 3, 1, 1, bias=False)
        )
        self.conv_linear = nn.Conv2d(b1+b2_n3+b3_n3_2, n1_linear, 1, 1, 0, bias=True)

        self.short_cut = nn.Sequential()
        if in_channels != n1_linear:
            self.short_cut = nn.Sequential(
                nn.Conv2d(in_channels, n1_linear, 1, 1, 0, bias=False),
                nn.BatchNorm2d(n1_linear)
            )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out = torch.cat((out1, out2, out3), 1)
        out = self.conv_linear(out)
        out += self.short_cut(x)
        return F.relu(out)
class Reduction_A(nn.Module):
    """
    Reduction-A block for Inception-v4, Inception-ResNet-v1, Inception-ResNet-v2 nets
    """

    def __init__(self, in_channels, k, l, m, n):
        super(Reduction_A, self).__init__()
        self.branch2 = BN_Conv2d(in_channels, n, 3, 2, 0, bias=False)
        self.branch3 = nn.Sequential(
            BN_Conv2d(in_channels, k, 1, 1, 0, bias=False),
            BN_Conv2d(k, l, 3, 1, 1, bias=False),
            BN_Conv2d(l, m, 3, 2, 0, bias=False)
        )

    def forward(self, x):
        out1 = F.max_pool2d(x, 3, 2, 0)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        return torch.cat((out1, out2, out3), 1)
    
class Inception_B_res(nn.Module):
    """
    Inception-A block for Inception-ResNet-v1\
    and Inception-ResNet-v2 net
    """

    def __init__(self, in_channels, b1, b2_n1, b2_n1x7, b2_n7x1, n1_linear):
        super(Inception_B_res, self).__init__()
        self.branch1 = BN_Conv2d(in_channels, b1, 1, 1, 0, bias=False)
        self.branch2 = nn.Sequential(
            BN_Conv2d(in_channels, b2_n1, 1, 1, 0, bias=False),
            BN_Conv2d(b2_n1, b2_n1x7, (1, 7), (1, 1), (0, 3), bias=False),
            BN_Conv2d(b2_n1x7, b2_n7x1, (7, 1), (1, 1), (3, 0), bias=False)
            # BN_Conv2d(b2_n1, b2_n1x7, (7, 7), (1, 1), (3, 3), bias=False),
            # BN_Conv2d(b2_n1x7, b2_n7x1, (7, 7), (1, 1), (3, 3), bias=False)
        )
        self.conv_linear = nn.Conv2d(b1 + b2_n7x1, n1_linear, 1, 1, 0, bias=False)
        self.short_cut = nn.Sequential()
        if in_channels != n1_linear:
            self.short_cut = nn.Sequential(
                nn.Conv2d(in_channels, n1_linear, 1, 1, 0, bias=False),
                nn.BatchNorm2d(n1_linear)
            )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out = torch.cat((out1, out2), 1)
        out = self.conv_linear(out)
        out += self.short_cut(x)
        return F.relu(out)
    
class Reduction_B_Res(nn.Module):
    """
    Reduction-B block for Inception-ResNet-v1 \
    and Inception-ResNet-v2  net
    """

    def __init__(self, in_channels, b2_n1, b2_n3, b3_n1, b3_n3, b4_n1, b4_n3_1, b4_n3_2):
        super(Reduction_B_Res, self).__init__()
        self.branch2 = nn.Sequential(
            BN_Conv2d(in_channels, b2_n1, 1, 1, 0, bias=False),
            BN_Conv2d(b2_n1, b2_n3, 3, 2, 0, bias=False),
        )
        self.branch3 = nn.Sequential(
            BN_Conv2d(in_channels, b3_n1, 1, 1, 0, bias=False),
            BN_Conv2d(b3_n1, b3_n3, 3, 2, 0, bias=False)
        )
        self.branch4 = nn.Sequential(
            BN_Conv2d(in_channels, b4_n1, 1, 1, 0, bias=False),
            BN_Conv2d(b4_n1, b4_n3_1, 3, 1, 1, bias=False),
            BN_Conv2d(b4_n3_1, b4_n3_2, 3, 2, 0, bias=False)
        )

    def forward(self, x):
        out1 = F.max_pool2d(x, 3, 2, 0)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        return torch.cat((out1, out2, out3, out4), 1)
    
class Inception_C_res(nn.Module):
    """
    Inception-C block for Inception-ResNet-v1\
    and Inception-ResNet-v2 net
    """

    def __init__(self, in_channels, b1, b2_n1, b2_n1x3, b2_n3x1, n1_linear):
        super(Inception_C_res, self).__init__()
        self.branch1 = BN_Conv2d(in_channels, b1, 1, 1, 0, bias=False)
        self.branch2 = nn.Sequential(
            BN_Conv2d(in_channels, b2_n1, 1, 1, 0, bias=False),
            BN_Conv2d(b2_n1, b2_n1x3, (1, 3), (1, 1), (0, 1), bias=False),
            BN_Conv2d(b2_n1x3, b2_n3x1, (3, 1), (1, 1), (1, 0), bias=False)
            # BN_Conv2d(b2_n1, b2_n1x3, (3, 3), (1, 1), (1, 1), bias=False),
            # BN_Conv2d(b2_n1x3, b2_n3x1, (3, 3), (1, 1), (1, 1), bias=False)
        )
        self.conv_linear = nn.Conv2d(b1 + b2_n3x1, n1_linear, 1, 1, 0, bias=False)
        self.short_cut = nn.Sequential()
        if in_channels != n1_linear:
            self.short_cut = nn.Sequential(
                nn.Conv2d(in_channels, n1_linear, 1, 1, 0, bias=False),
                nn.BatchNorm2d(n1_linear)
            )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out = torch.cat((out1, out2), 1)
        out = self.conv_linear(out)
        out += self.short_cut(x)
        return F.relu(out)
    
class Inception(nn.Module):
    """
    implementation of Inception-v4, Inception-ResNet-v1, Inception-ResNet-v2
    """

    def __init__(self, version, num_classes):
        super(Inception, self).__init__()
        self.version = version
        self.stem = Stem_v4_Res2()
        self.inception_A = self.__make_inception_A()
        self.Reduction_A = self.__make_reduction_A()
        self.inception_B = self.__make_inception_B()
        self.Reduction_B = self.__make_reduction_B()
        self.inception_C = self.__make_inception_C()

        self.fc = nn.Linear(2144, 1000)
        # self.gen_fc_1 = nn.Linear(1,32)
        # self.gen_relu = nn.ReLU()

        # self.age_fc_1 = nn.Linear(1,32)
        # self.age_relu = nn.ReLU()

        # self.cat_fc = nn.Linear(32+32+968,1032)
        # self.cat_relu = nn.ReLU()

        # self.fc2 = nn.Linear(1032, 1032)
        # self.final_fc = nn.Linear(1032,num_classes)
        self.fc1 = nn.Linear(1000, 512)
        self.fc2 = nn.Linear(512, 128)
        self.final_fc = nn.Linear(128,num_classes)

        self.relu = nn.ReLU()
        
    def __make_inception_A(self):
        layers = []
        for _ in range(5):
            layers.append(Inception_A_res(384, 32, 32, 32, 32, 48, 64, 384))
        return nn.Sequential(*layers)

    def __make_reduction_A(self):
            return Reduction_A(384, 256, 256, 384, 384) # 1152

    def __make_inception_B(self):
        layers = []
        for _ in range(10):
            layers.append(Inception_B_res(1152, 192, 128, 160, 192, 1152))  # 1152
        return nn.Sequential(*layers)
    
    def __make_reduction_B(self):
        return Reduction_B_Res(1152, 256, 384, 256, 288, 256, 288, 320)  # 2144

    def __make_inception_C(self):
        layers = []
        for _ in range(5):
            layers.append(Inception_C_res(2144, 192, 192, 224, 256, 2144))
        return nn.Sequential(*layers)

    
    def forward(self, x):
        out = self.stem(x)
        out = self.inception_A(out)
        out = self.Reduction_A(out)
        out = self.inception_B(out)
        out = self.Reduction_B(out)
        out = self.inception_C(out)


        out = F.avg_pool2d(out, 8)
        out = F.dropout(out, 0.2, training=self.training)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = out.view(out.size(0), -1)
        
        return out 
    
class CombinedModel(nn.Module):
    def __init__(self, bone_age_model_path, num_classes):
        super(CombinedModel, self).__init__()

        model = torch.load(bone_age_model_path)
        

        # self.bone_age_model = nn.Sequential(*list(model.children()))
   
        self.bone_age_model = model

        self.bone_age_model.eval()
        
        
        self.age_fc = nn.Linear(1,32)
        # self.gender_fc = nn.Linear(2, 32)


        self.cat_fc = nn.Sequential(nn.Linear(2144,478),nn.ReLU(),nn.BatchNorm1d(478))
        
        # self.fc1 = nn.Sequential(nn.Linear(478+2+32, 256),nn.ReLU(),nn.BatchNorm1d(256))
        self.fc1 = nn.Sequential(nn.Linear(478+2+32, 256),nn.ReLU(),nn.BatchNorm1d(256))
        # self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Sequential(nn.Linear(256, 128),nn.ReLU(),nn.BatchNorm1d(128))
        self.final_fc = nn.Linear(128,1)

        self.relu = nn.ReLU()
        self.bone_age_model.fc = self.cat_fc
        del self.bone_age_model.fc1 
        del self.bone_age_model.fc2
        del self.bone_age_model.final_fc 
        del self.bone_age_model.relu
        for param in self.bone_age_model.parameters():
            param.requires_grad = False
    def forward(self, xray_image,y,a):
        # 使用骨齡模型提取 X-ray 特徵
        with torch.no_grad():
            out = self.bone_age_model(xray_image)
        # print(y,a)
        # exit()
        #--------------------------------------------------------
        gender_one_hot = F.one_hot(y, num_classes=2).float()
        y = self.relu(gender_one_hot)
        y = y.view(y.size(0), -1)
        a = self.relu(self.age_fc(a))
        a = a.view(a.size(0), -1)
        
        z = torch.cat((out,y,a), dim = 1)
        #------------------------------------------------------
        z = self.fc1(z)
        # z = self.relu(z)
        z = F.dropout(z, 0.2, training=self.training)
        # z = self.fc2(z)
        # z = self.relu(z)
        # z = F.dropout(z, 0.2, training=self.training)
        z = self.fc3(z)
        # z = self.relu(z)
        z = F.dropout(z, 0.2, training=self.training)
        z = self.final_fc(z)
        
        return z 
class finetuneModel(nn.Module):
    def __init__(self,bone_age_model_path, modeldir, num_classes):
        super(finetuneModel, self).__init__()
        model = torch.load(bone_age_model_path)
   
        self.bone_age_model = model

        self.bone_age_model.eval()
        self.cat_fc = nn.Sequential(nn.Linear(2144,510),nn.ReLU(),nn.BatchNorm1d(510))
        self.relu = nn.ReLU()

        self.bone_age_model.fc = self.cat_fc
        del self.bone_age_model.fc1 
        del self.bone_age_model.fc2
        del self.bone_age_model.final_fc 
        del self.bone_age_model.relu
        for param in self.bone_age_model.parameters():
            param.requires_grad = False

        # 讀取所有模型
        self.base_models = []
        for model_path in sorted(modeldir):
            model = torch.load(model_path)
            model.eval()
            self.base_models.append(model)
    
        self.fc1 = nn.Sequential(nn.Linear(510+2+9, 256),nn.ReLU(),nn.BatchNorm1d(256))
        # self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Sequential(nn.Linear(256, 128),nn.ReLU(),nn.BatchNorm1d(128))
        self.final_fc = nn.Linear(128,1)
        
    def forward(self, x, y, a):
        with torch.no_grad():
            out = self.bone_age_model(x)

        gender_one_hot = F.one_hot(y, num_classes=2).float()
        gender = self.relu(gender_one_hot)
        gender = gender.view(gender.size(0), -1)
        # 獲取所有基礎模型的預測
        base_predictions = []
        with torch.no_grad():
            for model in self.base_models:
                pred = torch.sigmoid(model(x, y , a))  # 獲取預測機率
                base_predictions.append(pred)
         
        # 將所有預測拼接起來
        predictions = torch.cat(base_predictions, dim=1)
        
        # 將原始特徵和預測結果拼接
        combined = torch.cat([out, gender, predictions], dim=1)
        
        z = self.fc1(combined)

        z = F.dropout(z, 0.2, training=self.training)

        z = self.fc3(z)
        z = F.dropout(z, 0.2, training=self.training)
        z = self.final_fc(z)
        
        return z
#########################################################################################
import torch
from tqdm import tqdm
from collections import Counter
def test_hardvoting(data_loader, device, modeldir):
        running_acc = 0.0
        total = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader):
                images = batch['image'].to(device)
                name = batch['name'].to(device)
                gender = batch['gender'].to(device)
                age = batch['age'].to(device)
                gt = batch['gt'].to(device).float()
                # gt = gt.squeeze()
                # Run the model on the test set to predict labels
                batch_predictions = []
                batch_probs = []  
                for i in modeldir:
                    print(f"************** test {i}*****************")
                    smallmodel = torch.load(i)
                    smallmodel.eval()
                    outputs = smallmodel(images)
                    outputs = torch.sigmoid(outputs)
                    batch_probs.append(outputs)
                    preds = torch.where(outputs >= 0.5, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))  #hard
                    batch_predictions.append(preds.cpu().numpy())                                                             #hard
                    # print(preds.squeeze())
                    # print(gt.squeeze())
                avg_prob = torch.mean(torch.stack(batch_probs), dim=0)
                batch_predictions = list(zip(*batch_predictions))
                # print(batch_predictions[1])
                # exit()
                final_batch_preds = []
                for sample_preds in batch_predictions:
                    sample_preds = [pred.item() if isinstance(pred, np.ndarray) else pred for pred in sample_preds]
                    vote_count = Counter(sample_preds)
                    voted_pred = Counter(sample_preds).most_common(1)[0][0]  # 选择出现次数最多的类别
                    final_batch_preds.append(voted_pred)               
                final_batch_preds = torch.tensor(final_batch_preds, device=device)       


                total += gt.size(0)
                running_acc += (final_batch_preds == gt.squeeze()).sum().item()                                                     #hard
                # print(final_batch_preds,gt.squeeze())
                # print(running_acc)
                # exit()
                # running_acc += (preds == gt).sum().item()
                all_preds.extend(final_batch_preds.cpu().numpy())
                all_labels.extend(gt.cpu().numpy())
                all_probs.extend(avg_prob.cpu().numpy())

            fp_votes = []
            for idx, (pred, label) in enumerate(zip(final_batch_preds, gt)):
                if pred == 1 and label == 0:  # 假陽性：預測為1但實際為0
                    vote_count = Counter(batch_predictions[idx])
                    fp_votes.append(vote_count[1])  # 記錄投票為1的數量

            print("False Positive Vote Distribution:", Counter(fp_votes))
            # Calculate the average loss and accuracy over the epoch
            acc = running_acc / total
            # Calculate additional metrics
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            precision = precision_score(all_labels, all_preds)
            recall = recall_score(all_labels, all_preds)
            specificity = recall_score(all_labels, all_preds, pos_label=0)
            f1 = f1_score(all_labels, all_preds)
            fpr, tpr, _ = roc_curve(all_labels, all_probs)
            auc_ = auc(fpr, tpr)
            conf_matrix = confusion_matrix(all_labels, all_preds)
            # recall = recall_score(all_labels, all_preds, average='macro')
            # f1 = f1_score(all_labels, all_preds, average='macro')    
        return acc, precision, recall, specificity, f1, fpr, tpr, auc_, conf_matrix
def test_softvoting(data_loader, device, modeldir, df):
    running_acc = 0.0
    total = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    prediction = []
    prob = []
    patient_ids = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader):
            images = batch['image'].to(device)
            name = batch['name']
            gender = batch['gender'].to(device)
            age = batch['age'].to(device)
            gt = batch['gt'].to(device).float()
            name = [x.split('/')[-1] for x in name]
            name = [x[:-4] for x in name]

            # gt = gt.squeeze()
            # Run the model on the test set to predict labels
            batch_probs = []
            for i in modeldir:
                print(f"************** test {i}*****************")
                smallmodel = torch.load(i)
                smallmodel.eval()
                outputs = smallmodel(images,gender,age)
                outputs = torch.sigmoid(outputs)
                batch_probs.append(outputs) 
                # print(preds.squeeze())
                # print(gt.squeeze())
            avg_prob = torch.mean(torch.stack(batch_probs), dim=0)
            final_batch_preds = torch.where(avg_prob >= 0.5, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))

            ## 收集預測 ---------------------------------------------------

            prediction.extend(final_batch_preds.cpu().numpy().flatten())
            prob.extend(avg_prob.cpu().numpy().flatten())
            patient_ids.extend(name)
            ## ------------------------------------------------------------

            total += gt.size(0)
            running_acc += (final_batch_preds == gt).sum().item()
            # print(final_batch_preds,gt.squeeze())
            # print(running_acc)
            # exit()
            # running_acc += (preds == gt).sum().item()
            all_preds.extend(final_batch_preds.cpu().numpy())
            all_labels.extend(gt.cpu().numpy())
            all_probs.extend(avg_prob.cpu().numpy())

        
        acc = running_acc / total
        # Calculate additional metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        specificity = recall_score(all_labels, all_preds, pos_label=0)
        f1 = f1_score(all_labels, all_preds)
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        auc_ = auc(fpr, tpr)
        conf_matrix = confusion_matrix(all_labels, all_preds)
        results_df = pd.DataFrame({
           '醫令名稱': patient_ids,
           'prediction_class': prediction,
           'prediction_prob': prob
       })
        updated_df = df.merge(results_df, on='醫令名稱', how='left')
    return acc, precision, recall, specificity, f1, fpr, tpr, auc_, conf_matrix, updated_df
# model = torch.load('ncku_baprediction_agemonth_attention.pth')
modelpath = './model_pth_short/'
pattern = modelpath + 'new0.5_train_outliers_gender_age_60days_subdataset*.pth'
modeldir = glob.glob(pattern)
# print(len(modeldir))
# exit()
# val_acc, val_precision, val_recall, val_f1,voting_fpr, voting_tpr, voting_auc  = test_hardvoting(val_data_loader, device, modeldir)
# print(val_acc, val_precision, val_recall, val_f1, voting_fpr, voting_tpr,voting_auc)


### draw roc ################################################################
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

plt.figure(figsize=(10, 8))

# 繪製10個基礎模型的ROC
for i in range(1, 11):
   metrics = np.load(f'./matrics/short_model_{i}_val_metrics_age.npy', allow_pickle=True)
   best_metric = max(metrics, key=lambda x: x['f1_score'])
   
   fpr = best_metric['fpr']
   tpr = best_metric['tpr']
   auc_ = best_metric['auc']
#    print(f'-----------------------model{i}-------------------------------')
#    print(fpr,tpr,auc_)
   
   plt.plot(fpr, tpr, alpha=0.5, label=f'Model {i} (AUC = {auc_:.3f})')

# 加入voting結果的ROC
voting_acc, voting_precision, voting_recall, voting_specificity, voting_f1,voting_fpr, voting_tpr, voting_auc, voting_cm, result_df  = test_softvoting(val_data_loader, device, modeldir, validation_df)
# print(result_df)
# result_df.to_csv("prediction_short.csv")
print('Accuracy: %.2f' % (voting_acc))
print('Precision: %.2f' % (voting_precision))
print('Recall: %.2f:' % (voting_recall))
print('Specificity: %.2f' % (voting_specificity))
print('F1: %.2f' % (voting_f1))
plt.plot(voting_fpr, voting_tpr, 'r--', linewidth=2, 
        label=f'Ensemble (AUC = {voting_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves: Individual Models vs Ensemble')
plt.legend(loc="lower right")
plt.grid(True)
# plt.savefig("ROC_short_softvoting.png")
plt.show()

# confusion matrix
import seaborn as sns
# confusion_matrix = np.array([[56, 16], [7, 49]])  # 替換為您的數據
class_names = ['Negative', 'Positive']  # 類別名稱

# 創建圖表
plt.figure(figsize=(8, 6))
sns.heatmap(voting_cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)

# 添加標籤和標題
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
# plt.savefig("voting_cm_short.png")
plt.show()
