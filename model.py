"""
model.py — CNN architectures for the optimizer comparison study.

Two variants: a full CIFAR-10 model and a lightweight MNIST model for quick
sanity checks. Both share the same architectural philosophy so comparisons
stay meaningful across scales.

Outputs : used by train.py and loss_landscape.py
Usage   : imported as a module, not run directly
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Architecture constants ──────────────────────────────────────────────────

CIFAR_IN_CHANNELS  = 3
MNIST_IN_CHANNELS  = 1
NUM_CLASSES        = 10


# ── CIFAR-10 model ──────────────────────────────────────────────────────────

class ConvNet(nn.Module):
    """
    Small CNN for CIFAR-10. Two conv blocks followed by two FC layers.

    Deliberately kept shallow — the point is to study optimizer behavior,
    not squeeze out the last 2% accuracy with a ResNet.
    ~186K parameters, fast enough to train 6 times in a reasonable session.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout_rate: float = 0.3):
        super().__init__()

        # first block: 3→32 channels, 3x3 conv
        # BatchNorm here stabilizes training enough to make optimizer differences visible
        self.conv1 = nn.Conv2d(CIFAR_IN_CHANNELS, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)

        # second block: 32→64, then pool down to 8x8
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)

        # after two 2x2 pools: 32x32 → 8x8, so 64*8*8 = 4096
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── MNIST model (quick-test variant) ────────────────────────────────────────

class MiniConvNet(nn.Module):
    """
    Scaled-down version for MNIST. Same structural pattern, ~60K parameters.
    Useful for a 5-minute sanity run before committing to the full experiment.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        self.conv1 = nn.Conv2d(MNIST_IN_CHANNELS, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)

        # MNIST 28x28 → pool → 14x14 → pool → 7x7, so 32*7*7 = 1568
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Utility ──────────────────────────────────────────────────────────────────

def get_model(dataset: str = "cifar10", device: str = "cpu") -> nn.Module:
    """Return the appropriate model for the given dataset, moved to device."""
    if dataset == "mnist":
        model = MiniConvNet()
    else:
        model = ConvNet()

    model = model.to(device)
    return model


def estimate_flops(model: nn.Module, dataset: str = "cifar10") -> int:
    """
    Very rough FLOPs estimate — each multiply-add in a conv layer counts as 2 ops.
    Not a proper profiler output, but gives a ballpark for the paper.
    TODO: replace with torchprofile or fvcore if we need precise numbers
    """
    if dataset == "cifar10":
        # conv1: 32 * 32 * 32 * (3*3*3) * 2 = ~18.9M ops
        # conv2: 16 * 16 * 64 * (3*3*32) * 2 = ~75.5M ops  (approximate)
        # fc layers are small in comparison
        return 94_371_840
    else:
        # MiniConvNet on MNIST — roughly 5x smaller
        return 18_513_920


if __name__ == "__main__":
    # quick sanity check on model shapes
    cifar_model = get_model("cifar10")
    mnist_model = get_model("mnist")

    x_cifar = torch.randn(4, 3, 32, 32)
    x_mnist = torch.randn(4, 1, 28, 28)

    print(f"CIFAR-10 ConvNet output shape : {cifar_model(x_cifar).shape}")
    print(f"CIFAR-10 parameters           : {cifar_model.param_count():,}")
    print(f"Estimated FLOPs (CIFAR-10)    : {estimate_flops(cifar_model, 'cifar10'):,}")
    print()
    print(f"MNIST MiniConvNet output shape: {mnist_model(x_mnist).shape}")
    print(f"MNIST parameters              : {mnist_model.param_count():,}")
