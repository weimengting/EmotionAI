import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC
import argparse
from video_dataset import obtain_subjects, VideoDataset


# -------- LBP-TOP 特征提取 --------
def lbp_top(video_frames, radius=1, n_points=8):
    T, H, W = video_frames.shape
    features = []

    # XY-plane (spatial)
    for t in range(T):
        lbp_xy = local_binary_pattern(video_frames[t], n_points, radius, method='uniform')
        hist_xy, _ = np.histogram(lbp_xy.ravel(), bins=np.arange(0, n_points + 3), density=True)
        features.extend(hist_xy)

    # XT-plane (horizontal time)
    for y in range(H):
        plane_xt = video_frames[:, y, :]
        lbp_xt = local_binary_pattern(plane_xt, n_points, radius, method='uniform')
        hist_xt, _ = np.histogram(lbp_xt.ravel(), bins=np.arange(0, n_points + 3), density=True)
        features.extend(hist_xt)

    # YT-plane (vertical time)
    for x in range(W):
        plane_yt = video_frames[:, :, x]
        lbp_yt = local_binary_pattern(plane_yt, n_points, radius, method='uniform')
        hist_yt, _ = np.histogram(lbp_yt.ravel(), bins=np.arange(0, n_points + 3), density=True)
        features.extend(hist_yt)

    return np.array(features)


# 需要根据fold准备训练和测试的数据集
def prepare_dataset(fold, sign):
    train_subjects, test_subjects = obtain_subjects(fold)
    train_dataset = VideoDataset(
        video_folders=["/home/mengting/projects/EmotionAI/pose_cropped",
                       "/home/mengting/projects/EmotionAI/spon_cropped"
                       ],
        image_size=224,
        sign=sign,
        sublst=train_subjects,
    )
    train_X, train_y = [], []
    for i in range(train_dataset.__len__()):
        tgt, label = train_dataset.__getitem__(i)
        features = lbp_top(tgt)
        train_X.append(features)
        train_y.append(label)
    train_X, train_y = np.array(train_X), np.array(train_y)
    test_dataset = VideoDataset(
        video_folders=["/home/mengting/projects/EmotionAI/pose_cropped",
                       "/home/mengting/projects/EmotionAI/spon_cropped"
                       ],
        image_size=224,
        sign=sign,
        sublst=test_subjects,
    )
    test_X, test_y = [], []
    for i in range(test_dataset.__len__()):
        tgt, label = test_dataset.__getitem__(i)
        features = lbp_top(tgt)
        test_X.append(features)
        test_y.append(label)
    test_X, test_y = np.array(test_X), np.array(test_y)


    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    np.savez(f'fold_{fold}_{sign}.npz', train_X=train_X, train_y=train_y, test_X=test_X, test_y=test_y)


def classify(fold, sign):
    data = np.load(f'fold_{fold}_{sign}.npz')
    print(data)
    train_X, train_y, test_X, test_y = data["train_X"], data["train_y"], data["test_X"], data["test_y"]
    clf = SVC(kernel='linear')
    clf.fit(train_X, train_y)
    y_pred = clf.predict(test_X)

    print("Accuracy:", accuracy_score(test_y, y_pred))
    print("F1-score", f1_score(test_y, y_pred, average='macro'))

# multi
# fold 1: Accuracy: 0.23772609819121446
# F1-score 0.16971855743311146
# fold 2: Accuracy: 0.298372513562387
# F1-score 0.18230155398807205
# fold3: Accuracy: 0.27314814814814814
# F1-score 0.1717267134723861
# fold4： Accuracy: 0.27491408934707906
# F1-score 0.20930371029850112
# fold 5: Accuracy: 0.2632978723404255
# F1-score 0.16847450776055753
# fold6： Accuracy: 0.2350597609561753
# F1-score 0.12649522174276032
# fold7: Accuracy: 0.23809523809523808
# F1-score 0.16441122933865396
# fold 8: Accuracy: 0.26239067055393583
# F1-score 0.13991509331928667
# fold 9： Accuracy: 0.27191413237924866
# F1-score 0.16104059639804483
# fold10: Accuracy: 0.3220338983050847
# F1-score 0.20218316415243054

# binary
# fold1： Accuracy: 0.5478036175710594
# F1-score 0.5434013146806
# fold2: Accuracy: 0.7269439421338155
# F1-score 0.7088103499381027
# fold3: Accuracy: 0.6203703703703703
# F1-score 0.611851332398317
# fold4: Accuracy: 0.647766323024055
# F1-score 0.6372707290156054
# fold 5: Accuracy: 0.625
# F1-score 0.6168007459288332
# fold 6: Accuracy: 0.7609561752988048
# F1-score 0.7229172799528996
# fold 7: Accuracy: 0.6813186813186813
# F1-score 0.654545983447995
# fold 8: Accuracy: 0.5743440233236151
# F1-score 0.5639324277255311
# fold 9: Accuracy: 0.6726296958855098
# F1-score 0.6493616000274212
# fold 10: Accuracy: 0.6864406779661016
# F1-score 0.6589528775362743

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=4)
    parser.add_argument("--sign", type=str, default="binary")  # multi or binary
    args = parser.parse_args()
    prepare_dataset(args.fold, args.sign)
    classify(args.fold, args.sign)