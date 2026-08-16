"""One-off helper: extract a single CIFAR-10 image as test_image.png."""
from torchvision import datasets

ds = datasets.CIFAR10(root="./data", train=False, download=False)
img, label = ds[0]  # first test image (a PIL image) and its label
img.save("test_image.png")
print(f"Saved test_image.png (true label: {ds.classes[label]})")
