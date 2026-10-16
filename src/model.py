import torch


class BasicBlock(torch.nn.Module):
    """ResNet basic block: two 3x3 conv-bn layers plus a shortcut connection."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

        # When the shape changes, a 1x1 conv matches the shortcut to the block output.
        self.shortcut = torch.nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class ConvNet(torch.nn.Module):
    """ResNet18-like network for Cifar100 (32x32 RGB images, 100 classes).

    Same layout as ResNet18 (a stem, then 4 stages of 2 basic blocks, then global
    average pooling and a linear classifier), with two changes for small images:
      - the stem is a single 3x3 conv with stride 1 and no max pooling;
      - half the channels of ResNet18 (32-64-128-256 instead of 64-128-256-512).
    """

    def __init__(self, num_classes=100, width=32):
        super().__init__()
        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(3, width, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(width),
            torch.nn.ReLU(),
        )
        self.layer1 = self._make_stage(width, width, stride=1)          # 32x32
        self.layer2 = self._make_stage(width, 2 * width, stride=2)      # 16x16
        self.layer3 = self._make_stage(2 * width, 4 * width, stride=2)  # 8x8
        self.layer4 = self._make_stage(4 * width, 8 * width, stride=2)  # 4x4
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(8 * width, num_classes)

        for module in self.modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    @staticmethod
    def _make_stage(in_channels, out_channels, stride):
        return torch.nn.Sequential(
            BasicBlock(in_channels, out_channels, stride),
            BasicBlock(out_channels, out_channels, 1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
