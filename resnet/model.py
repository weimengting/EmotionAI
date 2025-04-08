import torch
import torch.nn as nn
import torchvision.models as models

class ResNetLSTMClassifier(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=1, num_classes=2):
        super(ResNetLSTMClassifier, self).__init__()

        # 使用预训练的 ResNet18 并移除最后的 FC 层
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]  # 去掉最后的全连接层
        self.feature_extractor = nn.Sequential(*modules)
        self.feature_dim = resnet.fc.in_features  # ResNet 输出的特征维度（通常是 512）

        # LSTM 层：输入为每帧提取的 ResNet 特征
        self.lstm = nn.LSTM(input_size=self.feature_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True)

        # 输出层
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape  # B=batch, T=frames
        x = x.view(B * T, C, H, W)  # 展平为批量帧输入到 ResNet

        features = self.feature_extractor(x)  # (B*T, 512, 1, 1)
        features = features.view(B, T, -1)  # (B, T, 512)

        # 输入 LSTM
        lstm_out, _ = self.lstm(features)  # 输出 shape: (B, T, hidden_dim)
        final_feature = lstm_out[:, -1, :]  # 取最后一帧输出作为代表

        logits = self.classifier(final_feature)  # (B, num_classes)
        return logits

if __name__ == '__main__':

    model = ResNetLSTMClassifier()
    x = torch.rand((1, 16, 3, 224, 224))
    y = model(x)
    print(y.shape)
