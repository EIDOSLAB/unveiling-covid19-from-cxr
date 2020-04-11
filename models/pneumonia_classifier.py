import torch
import torchvision

class PneumoniaClassifierChest(torch.nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        encoder = torchvision.models.resnet18(pretrained=pretrained)
        encoder.conv1.weight.data = encoder.conv1.weight.data[:, :1]
        encoder.conv1.in_channels = 1
        encoder = torch.nn.Sequential(*list(encoder.children())[:-1])
        self.encoder = encoder

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=512, out_features=3),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

class PneumoniaClassifierChest50(torch.nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        encoder = torchvision.models.resnet50(pretrained=pretrained)
        encoder.conv1.weight.data = encoder.conv1.weight.data[:, :1]
        encoder.conv1.in_channels = 1
        encoder = torch.nn.Sequential(*list(encoder.children())[:-1])
        self.encoder = encoder

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=2048, out_features=3),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

class PneumoniaClassifierRSNA(torch.nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        encoder = torchvision.models.resnet18(pretrained=pretrained)
        encoder.conv1.weight.data = encoder.conv1.weight.data[:, :1]
        encoder.conv1.in_channels = 1
        encoder = torch.nn.Sequential(*list(encoder.children())[:-1])
        self.encoder = encoder

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=512, out_features=1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

class PneumoniaClassifierRSNA50(torch.nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        encoder = torchvision.models.resnet50(pretrained=pretrained)
        encoder.conv1.weight.data = encoder.conv1.weight.data[:, :1]
        encoder.conv1.in_channels = 1
        encoder = torch.nn.Sequential(*list(encoder.children())[:-1])
        self.encoder = encoder

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=2048, out_features=1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
