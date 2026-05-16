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
# print(df_data)
# print(df_data[(df_data['unexpected_tall'] == 0) & (df_data['unexpected_short'] == 0)])
# print(df_data[df_data['unexpected_tall'] == 1])
# print(df_data[df_data['unexpected_short'] == 1])
# exit()
df_data['gt'] = df_data['unexpected_short'].map(lambda x: 1 if x==1 else 0)
# print(df_data['gt'].value_counts())
# exit()
# df_data = df_data[df_data['unexpected_tall'] == 0] ### 挑出正常和太矮的
# df_data = df_data.drop(columns=['unexpected_tall'])
# df_data.rename(columns={'unexpected_short': 'gt'}, inplace=True)
# print(df_data['age'].value_counts())
# exit()
# df_data = df_data[df_data['unexpected_short'] == False] ### 挑出正常和太高的
# df_data = df_data.drop(columns=['unexpected_short'])
# print(df_data['unexpected_short'].value_counts())
# exit()
# df_data['age_group'] = pd.cut(df_data['age'],
#                         bins=[-1, 5, 10, 14, 100],
#                         labels=[0, 1, 2, 3])
######################## model #######################################################################################
import torch
import torch.nn as  nn
import torch.nn.functional as F


# class Bottleneck(nn.Module):
#     expansion = 4
#     def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
#         super(Bottleneck, self).__init__()
        
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
#         self.batch_norm1 = nn.BatchNorm2d(out_channels)
        
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
#         self.batch_norm2 = nn.BatchNorm2d(out_channels)
        
#         self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
#         self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion)
        
#         self.i_downsample = i_downsample
#         self.stride = stride
#         self.relu = nn.ReLU()
        
#     def forward(self, x):
#         identity = x.clone()
#         x = self.relu(self.batch_norm1(self.conv1(x)))
        
#         x = self.relu(self.batch_norm2(self.conv2(x)))
        
#         x = self.conv3(x)
#         x = self.batch_norm3(x)
        
#         #downsample if needed
#         if self.i_downsample is not None:
#             identity = self.i_downsample(identity)
#         #add identity
#         x+=identity
#         x=self.relu(x)
        
#         return x

# class Block(nn.Module):
#     expansion = 1
#     def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
#         super(Block, self).__init__()
       

#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
#         self.batch_norm1 = nn.BatchNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
#         self.batch_norm2 = nn.BatchNorm2d(out_channels)

#         self.i_downsample = i_downsample
#         self.stride = stride
#         self.relu = nn.ReLU()

#     def forward(self, x):
#       identity = x.clone()

#       x = self.relu(self.batch_norm2(self.conv1(x)))
#       x = self.batch_norm2(self.conv2(x))

#       if self.i_downsample is not None:
#           identity = self.i_downsample(identity)
#       print(x.shape)
#       print(identity.shape)
#       x += identity
#       x = self.relu(x)
#       return x


        
        
# class ResNet(nn.Module):
#     def __init__(self, ResBlock, layer_list, num_classes, num_channels=3):
#         super(ResNet, self).__init__()
#         self.in_channels = 64
        
#         self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.batch_norm1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU()
#         self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)
        
#         self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
#         self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
#         self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
#         self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)
        
#         self.avgpool = nn.AdaptiveAvgPool2d((1,1))
#         self.gender_fc = nn.Linear(1, 32)
#         self.age_fc = nn.Linear(1, 32)
#         # self.fc = nn.Sequential(nn.Linear(512*ResBlock.expansion, 1024),
#         #                         nn.ReLU(),
#         #                         nn.Linear(1024, num_classes))
        
#         self.fc = nn.Sequential(
#             nn.Linear(512 * ResBlock.expansion + 32 + 32, 1064),
#             nn.ReLU(),
#             nn.Dropout(p=0.5),  # 添加 Dropout 層，p 是丟棄的概率
#             nn.Linear(1064, num_classes)
#         )
#     def forward(self, x, y, a):
#         x = self.relu(self.batch_norm1(self.conv1(x)))
#         x = self.max_pool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
        
#         x = self.avgpool(x)
#         x = x.reshape(x.shape[0], -1)
#         # print(x.shape)
#         gender_features = self.gender_fc(y.unsqueeze(1)).squeeze(1)
#         age_features = self.age_fc(y.unsqueeze(1)).squeeze(1)
#         # print(height_features.shape)
#         # exit()
#         out = torch.cat((x, gender_features, age_features), dim=1)
#         out = self.fc(out)
#         return out
        
#     def _make_layer(self, ResBlock, blocks, planes, stride=1):
#         ii_downsample = None
#         layers = []
        
#         if stride != 1 or self.in_channels != planes*ResBlock.expansion:
#             ii_downsample = nn.Sequential(
#                 nn.Conv2d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride),
#                 nn.BatchNorm2d(planes*ResBlock.expansion)
#             )
            
#         layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
#         self.in_channels = planes*ResBlock.expansion
        
#         for i in range(blocks-1):
#             layers.append(ResBlock(self.in_channels, planes))
            
#         return nn.Sequential(*layers)
    
# def ResNet101(num_classes, channels=3):
#     return ResNet(Bottleneck, [3,4,23,3], num_classes, channels)
# def ResNet50(num_classes, channels=3):
#     return ResNet(Bottleneck, [3,4,6,3], num_classes, channels)
#############################################################################
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

        # gender_one_hot = F.one_hot(y, num_classes=2).float()
        # y = self.relu(gender_one_hot)
        # y = y.view(y.size(0), -1)
        # out = torch.cat((out,y), dim = 1)
        # z = torch.cat((out,y,a), dim = 1)
        # z = self.cat_fc(z)
        # z = self.cat_relu(z)
        # z = self.fc2(z)
        # z = self.final_fc(z)


        # z = self.fc1(out)
        # z = self.relu(z)
        # z = F.dropout(z, 0.2, training=self.training)
        # z = self.fc2(z)
        # z = self.relu(z)
        # z = F.dropout(z, 0.2, training=self.training)
        # z = self.final_fc(z)
        
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
        
        self.fc1 = nn.Sequential(nn.Linear(478+2, 256),nn.ReLU(),nn.BatchNorm1d(256))
        # self.fc1 = nn.Sequential(nn.Linear(478, 256),nn.ReLU(),nn.BatchNorm1d(256))
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
    def forward(self, xray_image, y, a):
        # 使用骨齡模型提取 X-ray 特徵
        with torch.no_grad():
            out = self.bone_age_model(xray_image)
        # print(y,a)
        # exit()
        #-------------------------------------------------------------
        gender_one_hot = F.one_hot(y, num_classes=2).float()
        # age_one_hot = F.one_hot(a, num_classes=4).float()
        # age_one_hot = F.one_hot(y, num_classes=2).float()
        # print(gender_one_hot)
        # exit()
        # y = self.relu(self.gender_fc(gender_one_hot))
        y = self.relu(gender_one_hot)
        y = y.view(y.size(0), -1)
        # a = self.relu(self.age_fc(a))
        # a = a.view(a.size(0), -1)
        # a = a.float()
        # a = self.relu(self.age_fc(a))
        # a = self.relu(age_one_hot)
        # a = a.view(a.size(0), -1)
        # tensor = torch.zeros((a.size(0),20)).to(a.device)
 
        # tensor[torch.arange(a.size(0)).to(a.device),a.long().squeeze()] = 1.
    
        # a = tensor.view(tensor.size(0), -1)
        
        z = torch.cat((out,y), dim = 1)
        # z = F.dropout(z, 0.2, training=self.training)
        #---------------------------------------------------------------------
   
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

#### 分層抽樣 -----------------------------------------------------------------------------------
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
# #------------------ check ------------------
# # for i in gt_0_classes:
# #     i = pd.DataFrame(i)
# #     print(i['age'].value_counts() )
# #------------------------------------------
#### ---------------------------------------------------------------------------------------------
n = 0
# positive_train, positive_val = train_test_split(
#     df_data[df_data['gt'] != 0],
#     test_size=0.2,
#     random_state=seed
# )
for gt_0_class in gt_0_classes:
    n+=1
    # neg_train, neg_val = train_test_split(
    #     gt_0_class,
    #     test_size=0.2,
    #     random_state=seed
    # )
    # train_org_df = pd.concat([positive_train, neg_train])
    # train_org_df = train_org_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    # val_org_df = pd.concat([positive_val, neg_val])
    # val_org_df = val_org_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    other_classes = df_data[df_data['gt'] != 0]
    combined_data = pd.concat([gt_0_class, other_classes])
    subdf = combined_data.copy()
    # # print(gt_0_class,other_classes)
    # # exit()
    # # print(subdf[subdf['gt'] == 0])
    # # exit()
    train_org_df, val_org_df = train_test_split(subdf, 
                                    test_size = 0.2, 
                                    random_state=seed,
                                    stratify = subdf['gt'])
     # print(train_org_df[train_org_df['gt'] == 0],train_org_df[train_org_df['gt'] == 1])
    # print(val_org_df[val_org_df['gt'] == 0],val_org_df[val_org_df['gt'] == 1])
    # exit()
    # print(val_org_df[val_org_df['gt'] == 0])
    # val_org_df, test_org_df = train_test_split(tmp_df, 
    #                                 test_size = 0.5, 
    #                                 random_state=seed,
    #                                 stratify = tmp_df['gt'])
    # print(train_org_df,val_org_df)
    # print(train_org_df['gt'].value_counts())
    # print(val_org_df['gt'].value_counts())
    # print(test_org_df['gt'].value_counts())
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
        
    train_transform = transforms.Compose([
        # transforms.Pad(padding=60),
        transforms.RandomAffine(degrees=(-20,20), scale=(0.9, 1.2), translate = (0, 0.1)),
        transforms.ColorJitter(brightness=(0.6, 1.4)),
        # transforms.Resize((512, 512)),
        
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]) 
    valid_transform = transforms.Compose([
        
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]) 

    train_dataset = BonesDataset(dataframe = train_org_df, image_dir=dataset_path, transform = train_transform)
    val_dataset = BonesDataset(dataframe = val_org_df, image_dir = dataset_path, transform = valid_transform)
    train_data_loader = DataLoader(train_dataset,batch_size=128,shuffle=False,drop_last=True)
    val_data_loader = DataLoader(val_dataset,batch_size=128,shuffle=False,drop_last=True)
    # print(f'------------------- start subdataset {n} --------------------------------------------')
    print("Total image:",len(train_dataset)+len(val_dataset))
    # image = train_dataset[0]['image']
    # image.show()
    # exit()
    ############################### train ############################################################################
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    from sklearn.metrics import roc_curve, auc, confusion_matrix
    from torchvision import models
    # criterion = nn.L1Loss()
    # resnet34 = models.resnet34()
    # num_ftrs = resnet34.fc.in_features
    # resnet34.fc = nn.Linear(num_ftrs, 2)
    # model = resnet34.to(device)

    # model = ResNet50(1)
    modelpath = 'ncku_baprediction_xrayonly_epoch300_mae6.2.pth'
    bone_age_model = CombinedModel(modelpath, num_classes=1)
    model = bone_age_model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.09)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8,  mode='min',patience=15, verbose=1, cooldown=5, min_lr=0.000001)
    criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience=5)
    # optimizer = torch.optim.SGD(age_predictor.parameters(), lr=0.001, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=200) 
    best_valLoss = 9999
    num_epochs = 200
    txtpath = f'./ablation/short/xray+gender/train_ablation_xray_gender_subdataset{n}.txt'
    txtpath2 = f'./ablation/short/xray+gender/val_ablation_xray_gender_subdataset{n}.txt'
    # txtpath = f'./result_train_short/train_outliers.txt'
    # txtpath2 = f'./result_val_short/val_outliers.txt'
    f = open(txtpath, 'w')
    ff = open(txtpath2, 'w')
    from torch.autograd import Variable

    # Function to save the model
    def saveModel():
        path = f"./ablation/short/xray+gender/model_ablation_xray_gender_subdataset{n}.pth"
        # path = f"./model_pth_short/train_outliers.pth"
        torch.save(model, path)

    # Function to test the model with the test dataset and print the accuracy for the test images
    def test(data_loader, device, epoch):
        model.eval()
        running_acc = 0.0
        total = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in tqdm(data_loader):
                images = batch['image'].to(device)
                gender = batch['gender'].to(device)
                age = batch['age'].to(device)
                gt = batch['gt'].to(device).float()
                # gt = gt.squeeze()
                # Run the model on the test set to predict labels
                outputs = model(images,gender,age)
                outputs = torch.sigmoid(outputs)
                # print(outputs)
                # exit()
                
                # The label with the highest energy will be our prediction
                # _, preds = torch.max(outputs.data, 1)
                preds = torch.where(outputs >= 0.5, torch.tensor(1.0), torch.tensor(0.0))
                print(preds.squeeze())
                print(gt.squeeze())
                # exit()
                # Accumulate the total number of labels and correct predictions
                total += gt.size(0)
                running_acc += (preds== gt).sum().item()
                # running_acc += (preds == gt).sum().item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(gt.cpu().numpy())

            # Calculate the average loss and accuracy over the epoch
            acc = running_acc / total

            # Calculate additional metrics
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            precision = precision_score(all_labels, all_preds)
            recall = recall_score(all_labels, all_preds)
            specificity = recall_score(all_labels, all_preds, pos_label=0)
            f1 = f1_score(all_labels, all_preds)

            fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
            auc_score = auc(fpr, tpr)
            conf_matrix = confusion_matrix(all_labels, all_preds)
            epoch_metrics = {
                'epoch': epoch + 1,
                'accuracy': acc,
                'precision': precision,
                'recall': recall,
                'specificity': specificity,
                'f1_score': f1,
                'auc': auc_score,
                'fpr': fpr,
                'tpr': tpr,
                'confusion_matrix': conf_matrix,
            }
            
        return acc, precision, recall, specificity, f1, epoch_metrics


    best_f1 = 0.0
    model.to(device)
    train_metrics_history = []
    val_metrics_history = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_acc = 0.0
        total = 0.0
        all_preds = []
        all_labels = []

        model.train()
        for batch in tqdm(train_data_loader):

            image = batch['image'].to(device)
            gender = batch['gender'].to(device)
            age = batch['age'].to(device)
            gt = batch['gt'].to(device).float()
            # gt = gt.squeeze()
            # print(image,gt)
            # exit()
            optimizer.zero_grad()
            # outputs = model(image)
            outputs = model(image,gender,age)

            # outputs = outputs.requires_grad_(True)
            preds = torch.where(F.sigmoid(outputs) >= 0.5, torch.tensor(1.0), torch.tensor(0.0))
            # loss = criterion(outputs.squeeze(), gt)
            loss = criterion(outputs, gt)

            loss.backward()
        
            optimizer.step()

            running_loss += loss.item()
            # _, preds = torch.max(outputs.data, 1)
            
            running_acc += (preds== gt).sum().item()
            total += gt.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(gt.cpu().numpy())

        train_acc = running_acc / total

        # Calculate additional metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        specificity = recall_score(all_labels, all_preds, pos_label=0)
        f1 = f1_score(all_labels, all_preds)

        fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
        auc_score = auc(fpr, tpr)
        conf_matrix = confusion_matrix(all_labels, all_preds)
        train_epoch_metrics  = {
            'epoch': epoch + 1,
            'accuracy': train_acc,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1,
            'auc': auc_score,
            'fpr': fpr,
            'tpr': tpr,
            'confusion_matrix': conf_matrix,
        }
        
        train_metrics_history.append(train_epoch_metrics)

        val_acc, val_pre, val_recall, val_spe, val_f1, val_epoch_metrics = test(val_data_loader, device, epoch)
        val_metrics_history.append(val_epoch_metrics)
        print('Epoch [%d], Loss: %.3f' % (epoch + 1, running_loss / len(train_data_loader)))
        print('Train Accuracy: %.2f, Val Accuracy: %.2f' % (train_acc, val_acc))
        print('Train Precision: %.2f, Val Precision: %.2f' % (precision, val_pre))
        print('Train Recall: %.2f, Val Recall: %.2f' % (recall, val_recall))
        print('Train Specificity: %.2f, Val Specificity: %.2f' % (specificity, val_spe))
        print('Train F1: %.2f, Val F1: %.2f' % (f1, val_f1))

        if val_f1 >= best_f1:
            saveModel()
            print('Epoch [%d],save model' % (epoch + 1))
            best_f1 = val_f1

        f.write(str(train_acc) + " " + str(precision) + " " + str(recall) + "  "+ str(specificity) + "  " + str(f1)+'\n')
        f.flush()
        ff.write(str(val_acc) + " " + str(val_pre) + " " + str(val_recall) + " " + str(val_spe) + "  "+str(val_f1)+'\n')
        ff.flush()
    np.save(f'./ablation/short/xray+gender/short_model_{n}_train_metrics_ablation_xray_gender.npy', train_metrics_history)
    np.save(f'./ablation/short/xray+gender/short_model_{n}_val_metrics_ablation_xray_gender.npy', val_metrics_history)

    f.close() 
    ff.close()   
