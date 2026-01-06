from torch import nn
import torch
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10)
        )

    def forward(self,x):
        x = self.flatten(x)
        logits = self.network(x)
        return logits
    
def test_accuracy(model,dataloader):
    #全てのミニバッチに対して推論をして、正解率を計算する
    n_corrects = 0 #正解した個数

    #モデルのデバイスを調べる
    device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        for image_batch,label_batch in dataloader:
            #バッチを、modelと同じデバイスに送る
            image_batch = image_batch.to(device)
            label_batch = label_batch.to(device)

            #モデルに入れて結果logitsを出す
            logits_batch = model(image_batch)
        
            predict_batch = logits_batch.argmax(dim=1)
            n_corrects += (label_batch == predict_batch).sum().item()
    
    #精度(正解率)を計算する
    accuracy = n_corrects / len(dataloader.dataset)

    return accuracy


def train(model,dataloader,loss_fn,optimizer):
    '''1 epochの学習を行う'''
    
     #モデルのデバイスを調べる
    device = next(model.parameters()).device

    model.train()
    for image_batch,label_batch in dataloader:
        
        #バッチを、modelと同じデバイスに送る
        image_batch = image_batch.to(device)
        label_batch = label_batch.to(device)

        #モデルにパッチを入れて計算
        logits_batch=model(image_batch)

        loss = loss_fn(logits_batch,label_batch)

        #最適化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()

def test(model,dataloader,loss_fn):
    '''1エポック分のロスを計算'''
    loss_total = 0.0 #ロスの合計

     #モデルのデバイスを調べる
    device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        for image_batch,label_batch in dataloader:
             #バッチを、modelと同じデバイスに送る
            image_batch = image_batch.to(device)
            label_batch = label_batch.to(device)
            
            #モデルにパッチを入れて、ロジット計算
            logits_batch=model(image_batch)

            #ロスを計算する(引数は、予測ロジットと正解ラベル)
            loss = loss_fn(logits_batch,label_batch)
            loss_total += loss.item()

           
    return loss_total / len(dataloader)
