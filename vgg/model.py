import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class VGG_LSTM(nn.Module):
    def __init__(self, num_classes=2, hidden_dim=256, num_layers=1):
        super(VGG_LSTM, self).__init__()
        self.vgg = models.vgg16(pretrained=True)
        # 去掉原来的分类头，提取特征向量
        self.vgg.classifier = nn.Sequential(*list(self.vgg.classifier.children())[:-1])
        self.feature_dim = 4096  # VGG16 最后一个全连接层的输出维度

        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        features = self.vgg(x)  # shape: [B*T, 4096]
        features = features.view(B, T, -1)  # shape: [B, T, 4096]

        lstm_out, _ = self.lstm(features)  # shape: [B, T, hidden_dim]
        last_output = lstm_out[:, -1, :]  # 取最后一个时间步的输出

        logits = self.classifier(last_output)  # shape: [B, num_classes]
        return logits


if __name__ == '__main__':

    model = VGG_LSTM(num_classes=2)
    x = torch.rand((1, 16, 3, 224, 224))
    y = model(x)
    print(y.shape)
