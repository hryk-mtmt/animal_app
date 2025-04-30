from torchvision import transforms
import pytorch_lightning as pl
import torch.nn as nn

from torchvision.models import resnet18

# 前処理
'''
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.458, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
'''
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomCrop(224, padding=4),
    transforms.Normalize(mean=[0.458, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ネットワーク定義
'''
class Net(pl.LightningModule):

    def __init__(self):
        super().__init__()

        #学習時に使ったのと同じ学習済みモデルを定義
        self.feature = resnet18(pretrained=True)
        self.fc = nn.Linear(1000, 2)

    def forward(self, X):
        #学習時に使ったのと同じ順伝播
        h = self.feature(X)
        h = self.fc(h)
        return h
'''

class Net(nn.Module):  #ResNet
    def __init__(self):
        super(Net, self).__init__()
        # 事前学習済みのResNet18モデルをロード
        self.resnet = resnet18(pretrained=True)
        # 最後の全結合層(fc)を変更
        self.resnet.fc = nn.Linear(in_features=self.resnet.fc.in_features, out_features=2)

    def forward(self, x):
        return self.resnet(x)    
    