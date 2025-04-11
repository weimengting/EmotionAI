import torch
import torch.nn as nn

from transformers import VivitForVideoClassification


class VIVIT(nn.Module):
    def __init__(self, num_classes=2):
        super(VIVIT, self).__init__()

        self.model = VivitForVideoClassification.from_pretrained("./vivit-b-16x2-kinetics400")  # (video_length, )
        self.model.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = torch.cat([x, x], dim=1)
        return self.model(x)




if __name__ == '__main__':
    x = torch.rand((1, 16, 3, 224, 224))
    model = VIVIT(num_classes=2)
    with torch.no_grad():
        outputs = model(x)
        logits = outputs.logits
    print(logits.shape)
    # model predicts one of the 400 Kinetics-400 classes

    # print(y.shape)