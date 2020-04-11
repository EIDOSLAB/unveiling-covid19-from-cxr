import torch
import torchvision
import torch.nn.functional as F
from torch import nn

def get_covid_classifier(checkpoint=None):
    covid_classifier = CovidClassifier(encoder=None, pretrained=checkpoint is None)

    if checkpoint is not None:
        checkpoint = torch.load(checkpoint, map_location={'cuda:0': 'cpu'})
        print(f'Loaded covid classifier from epoch {checkpoint["epoch"]}')
        print(f'Classifier corda version {checkpoint["corda-version"]}')

        covid_classifier.load_state_dict(checkpoint['model'])

    return covid_classifier

class CovidClassifier(torch.nn.Module):
    def __init__(self, encoder=None, pretrained=False, freeze_conv=True):
        super().__init__()

        if encoder is None:
            encoder = torchvision.models.resnet18(pretrained=pretrained)
            encoder.conv1.weight.data = encoder.conv1.weight.data[:, :1]
            encoder.conv1.in_channels = 1
            encoder = torch.nn.Sequential(*list(encoder.children())[:-1])
        self.encoder = encoder
        self.freeze_conv = freeze_conv

        for param in self.encoder.parameters():
            param.requires_grad = not freeze_conv

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=512, out_features=256),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256, out_features=1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        if self.freeze_conv:
            self.encoder.eval()

        with torch.set_grad_enabled(not self.freeze_conv):
            x = self.encoder(x)
            x = torch.flatten(x, 1)
        return self.classifier(x)

class CovidClassifier50(torch.nn.Module):
    def __init__(self, encoder=None, pretrained=False, freeze_conv=True):
        super().__init__()

        if encoder is None:
            encoder = torchvision.models.resnet50(pretrained=pretrained)
            encoder.conv1.weight.data = encoder.conv1.weight.data[:, :1]
            encoder.conv1.in_channels = 1
            encoder = torch.nn.Sequential(*list(encoder.children())[:-1])
        self.encoder = encoder
        self.freeze_conv = freeze_conv

        for param in self.encoder.parameters():
            param.requires_grad = not freeze_conv

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=2048, out_features=512),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=512, out_features=1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        if self.freeze_conv:
            self.encoder.eval()

        with torch.set_grad_enabled(not self.freeze_conv):
            x = self.encoder(x)
            x = torch.flatten(x, 1)
        return self.classifier(x)


class LeNet1024(nn.Module):
    def __init__(self):
        super().__init__()

        self.scale=1
        self.ksize=3
        self.conv1 = nn.Conv2d(1, 1*self.scale, self.ksize, 1, bias=False)
        self.conv1_bn = nn.BatchNorm2d(1*self.scale)
        self.conv2 = nn.Conv2d(1*self.scale, 2*self.scale, self.ksize, 1, bias=False)
        self.conv2_bn = nn.BatchNorm2d(2*self.scale)
        self.conv3 = nn.Conv2d(2*self.scale, 4*self.scale, self.ksize, 1, bias=False)
        self.conv3_bn = nn.BatchNorm2d(4*self.scale)
        self.conv4 = nn.Conv2d(4*self.scale, 8*self.scale, self.ksize, 1, bias=False)
        self.conv4_bn = nn.BatchNorm2d(8*self.scale)
        self.conv5 = nn.Conv2d(8*self.scale, 16*self.scale, self.ksize, 1, bias=False)
        self.conv5_bn = nn.BatchNorm2d(16*self.scale)
        self.conv6 = nn.Conv2d(16*self.scale, 32*self.scale, self.ksize, 1, bias=False)
        self.conv6_bn = nn.BatchNorm2d(32*self.scale)
        self.conv7 = nn.Conv2d(32*self.scale, 64*self.scale, self.ksize, 1, bias=False)
        self.conv7_bn = nn.BatchNorm2d(64*self.scale)
        self.conv8 = nn.Conv2d(64*self.scale, 128*self.scale, self.ksize, 1, bias=False)
        self.conv8_bn = nn.BatchNorm2d(128*self.scale)
        self.fc2 = nn.Linear(2*2*128*self.scale, 1)
        self.sigm = torch.nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv5_bn(self.conv5(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv6_bn(self.conv6(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv7_bn(self.conv7(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv8_bn(self.conv8(x)))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1,2*2*128*self.scale)

        x = self.sigm(self.fc2(x))
        return x

class LeNet1024NoPooling(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale=4
        self.ksize=3
        self.conv1 = nn.Conv2d(1, 1*self.scale, self.ksize, 1, bias=False)
        self.conv1_bn = nn.BatchNorm2d(1*self.scale)

        self.conv2 = nn.Conv2d(1*self.scale, 2*self.scale, self.ksize, 1, bias=False)
        self.conv2_bn = nn.BatchNorm2d(2*self.scale)

        self.conv3 = nn.Conv2d(2*self.scale, 4*self.scale, self.ksize, 1, bias=False)
        self.conv3_bn = nn.BatchNorm2d(4*self.scale)

        self.fc2 = nn.Linear(1018*1018*(16), 1)
        self.sigm = torch.nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))

        x = x.view(-1,1018*1018*16)
        x = self.sigm(self.fc2(x))

        return x

class LeNet1024NoPoolingDeep(nn.Module):
    def __init__(self):
        super().__init__()

        self.scale=8
        self.ksize=3
        self.conv1 = nn.Conv2d(1, self.scale, self.ksize, 1, bias=False)
        self.conv1_bn = nn.BatchNorm2d(self.scale)
        self.conv2 = nn.Conv2d(self.scale, self.scale, self.ksize, stride=2, bias=False)
        self.conv2_bn = nn.BatchNorm2d(self.scale)
        self.conv3 = nn.Conv2d(self.scale, self.scale, self.ksize, stride=2, bias=False)
        self.conv3_bn = nn.BatchNorm2d(self.scale)
        self.conv4 = nn.Conv2d(self.scale, self.scale, self.ksize, stride=2, bias=False)
        self.conv4_bn = nn.BatchNorm2d(self.scale)

        self.conv5 = nn.Conv2d(self.scale, self.scale, self.ksize, stride=2, bias=False)
        self.conv5_bn = nn.BatchNorm2d(self.scale)
        self.conv6 = nn.Conv2d(self.scale, self.scale, self.ksize, stride=2, bias=False)
        self.conv6_bn = nn.BatchNorm2d(self.scale)
        self.conv7 = nn.Conv2d(self.scale, self.scale, self.ksize, stride=2, bias=False)
        self.conv7_bn = nn.BatchNorm2d(self.scale)
        self.conv8 = nn.Conv2d(self.scale, self.scale, self.ksize, stride=2, bias=False)
        self.conv8_bn = nn.BatchNorm2d(self.scale)

        self.fc2 = nn.Linear(self.scale*6*6, 1)
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x = F.relu(self.conv5_bn(self.conv5(x)))
        x = F.relu(self.conv6_bn(self.conv6(x)))
        x = F.relu(self.conv7_bn(self.conv7(x)))
        x = F.relu(self.conv8_bn(self.conv8(x)))

        x = x.view(-1,self.scale*6*6)
        x = self.sigm(self.fc2(x))
        return x
