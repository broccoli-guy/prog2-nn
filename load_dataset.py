import matplotlib.pyplot as plt
from torchvision import datasets
import torchvision.transforms.v2 as transformers
import torch

#データセット読み込み   
ds_train = datasets.FashionMNIST(
    root='dataset',
    train=True,
    download=True
)

print(f'dataset size: {len(ds_train)}')

image, target = ds_train[999]

print(type(image))
print(target)

fig, ax = plt.subplots()

ax.imshow(image,cmap='gray_r',vmin=0,vmax=255)
ax.set_title(target)
plt.show()

#pil -> torch tensor

image = transformers.functional.to_image(image)
image = transformers.functional.to_dtype(image,dtype=torch.float32,scale=True)
print(type(image))
print(image.shape, image.dtype)
print(image.min(),image.max())