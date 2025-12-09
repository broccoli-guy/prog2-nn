import matplotlib.pyplot as plt

import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms

import models

#モデルのインスタンス作成
model = models.MyModel()
print(model)

#データセットのロード
ds_train = datasets.FashionMNIST(
    root = 'data',
    train = True,
    download = True,
    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32,scale = True)])
)

#imageはPILではなくTensorに変換済み
image,target = ds_train[0]
print(type(image),image.shape,image.dtype)
#(1, H, W)から(1, 1, H, W)二次元を上げる
image = image.unsqueeze(dim=0)
print(image.shape)

#モデルに入れて結果(logits)を出す
model.eval()
with torch.no_grad():
    logits = model(image)

print(logits)

#ロジットをグラフにする
plt.bar(range(logits.shape[1]),logits[0])
plt.show()

#クラス確率のグラフ
probs = logits.softmax(dim=1)
plt.bar(range(probs.shape[1]),probs[0])
plt.ylim(0,1)
plt.show()

#演習のグラフ
probs = logits.softmax(dim=1)

fig,axs = plt.subplots(1,2,figsize=(10,6))

axs[0].imshow(image[0,0],cmap='gray_r',vmin=0,vmax=1) 
#(1,1,28,28)のうち[0,0, :,:]

axs[1].bar(range(probs.shape[1]),probs[0])
axs[1].set_ylim(0,1)

axs[0].set_title(f'class: {target} {datasets.FashionMNIST.classes[target]}')
axs[1].set_title(f'predicted class {probs[0].argmax()}')

plt.show()