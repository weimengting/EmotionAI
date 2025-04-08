import os
import pandas as pd
import shutil

import torch
import torchvision.models as models
import torch.nn as nn


# 在备份里面修改, A为0,B为1
dict = {'1A':0, '1B':1, '1NA':2}

def add_labels():
    excel = '/media/mengting/data2/SS_dataset/cropped_sum.xlsx'
    df = pd.read_excel(excel)
    data_root = '/media/mengting/data2/SS_dataset/SS_video_clips_with_labels2'
    subjects = os.listdir(data_root)
    for subject in subjects:
        cur_subject_path = os.path.join(data_root, subject)
        videos = os.listdir(cur_subject_path)
        for video in videos:
            cur_video_path = os.path.join(cur_subject_path, video)
            if os.path.isdir(cur_video_path):
                condition = (df['vides'] == int(subject)) & (df['clips'] == int(video))
                label = df.loc[condition, 'condition'].item()

                dst_path = cur_video_path + '_' + str(dict[label])
                os.rename(cur_video_path, dst_path)

def remove_files():
    data_root = '/media/mengting/data2/SS_dataset/SS_video_clips_with_labels2'
    subjects = os.listdir(data_root)
    for subject in subjects:
        cur_subject_path = os.path.join(data_root, subject)
        videos = os.listdir(cur_subject_path)
        for video in videos:
            cur_video_path = os.path.join(cur_subject_path, video)
            if os.path.isdir(cur_video_path):
                if cur_video_path.endswith('_2'):
                    #shutil.rmtree(cur_video_path)
                    print(cur_video_path)



subjects = ['1', '10', '11', '12', '13', '14', '16', '17', '18', '19', '2', '20', '21', '22', '23',
            '24', '25', '26', '27', '28', '29', '3', '30', '31', '32', '33', '34', '37', '38', '4',
            '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '54', '55',
            '56', '57', '58', '59', '6', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '7',
            '70', '71', '72', '73', '74', '8', '9']
if __name__ == '__main__':
    #remove_files()
    #add_labels()
    #print(len(subjects))
    model = models.vgg16(pretrained=True)
    num_features = model.classifier[6].in_features  # 获取最后一层的输入特征数量
    model.classifier[6] = nn.Linear(num_features, 2)
    x = torch.rand(4, 3, 256, 256)
    y = model(x)
    print(y.shape)
    print(model)