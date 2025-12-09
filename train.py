import time
import matplotlib.pyplot as plt

import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms

import models

#データセットの前定義
ds_transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True)
])

#データセットの読み込み
ds_train = datasets.FashionMNIST(
    root='data',
    train=True,     #訓練用を指定
    download=True,
    transform=ds_transform
)
ds_test = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=ds_transform
)

#dataloader作成
batch_size = 64
dataloader_train = torch.utils.data.DataLoader(
    ds_train,
    batch_size=batch_size,
    shuffle=True
)
dataloader_test = torch.utils.data.DataLoader(
    ds_test,
    batch_size=batch_size
)

#バッチを取り出す実験

# for image_batch,label_batch in dataloader_test:
#     print(image_batch.shape)
#     print(label_batch.shape)
#     break

#モデルのインスタンスを作成
model = models.MyModel()

#精度を計算する
# train_acc = models.test_accuracy(model,dataloader_test)
# print(f'test accuracy{train_acc*100:.3f}%')

#損失関数の選択(誤差関数・ロス関数)
loss_fn = torch.nn.CrossEntropyLoss()

#最適化手法
learning_rate = 1e-3 #学習率
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)


# train_loss = models.train(model,dataloader_train,loss_fn,optimizer)
# print(f'training loss: {train_loss}')

n_epochs = 20
train_loss_log = []
train_acc_log = []
val_loss_log = []
val_acc_log = []

for epoch in range(n_epochs):
    print(f'epoch {epoch+1}/{n_epochs}')

    time_start = time.time()
    train_loss = models.train(model,dataloader_train,loss_fn,optimizer)
    time_end = time.time()

    print(f'    training loss: {train_loss} ({time_end-time_start:.3f}s)')
    train_loss_log.append(train_loss)

    time_start = time.time()
    val_loss = models.test(model,dataloader_test,loss_fn)
    time_end = time.time()
    print(f'    validation loss: {val_loss} ({time_end-time_start:.3f}s)')
    val_loss_log.append(val_loss)
    

    #精度を計算する

    time_start = time.time()
    train_acc = models.test_accuracy(model,dataloader_train)
    time_end = time.time()
    train_acc_log.append(train_acc)
    print(f'    training accuracy: {train_acc*100:.3f}% ({time_end-time_start:.3f}s)')

    time_start = time.time()
    val_acc = models.test_accuracy(model,dataloader_test)
    time_end = time.time()
    print(f'    validation accuracy: {val_acc*100:.3f}% ({time_end-time_start:.3f}s)')
    val_acc_log.append(val_acc)



#グラフ表示

# plt.plot(train_loss_log)
# plt.xlabel('epochs')
# plt.ylabel('loss')

# plt.xlim(left=1)
# plt.grid()
# plt.show()

# plt.plot(train_acc_log)
# plt.xlabel('epochs')
# plt.ylabel('accuracy')

# plt.xlim(left=1)
# plt.grid()
# plt.show()

fig,axs = plt.subplots(1,2,figsize=(10,6))

axs[0].plot(range(1, n_epochs+1),train_loss_log,label='train')
axs[0].plot(range(1, n_epochs+1),val_loss_log,label='validation')
axs[0].set_xlim(left=1)
axs[0].set_xlabel('epochs')
axs[0].set_ylabel('loss')

axs[1].plot(range(1, n_epochs+1),train_acc_log,label='train')
axs[1].plot(range(1, n_epochs+1),val_acc_log,label='validation')
axs[1].set_xlim(left=1)
axs[1].set_xlabel('epochs')
axs[1].set_ylabel('accuracy')

axs[0].set_xticks(range(1, n_epochs+1))
axs[1].set_xticks(range(1, n_epochs+1))

axs[0].grid()
axs[1].grid()

axs[0].legend()
axs[1].legend()

plt.show()