from torchvision import transforms

from dataset.CIFAR10 import CIFAR10Pair
from torchvision.models import resnet18

model = resnet18()
print(model)