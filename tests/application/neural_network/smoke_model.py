"""Small inference model for testing parameter preparation without downloads."""

from torch import nn


class SmokeNet(nn.Module):
    def __init__(self):
        super(SmokeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(2, 3)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.flatten(1))
