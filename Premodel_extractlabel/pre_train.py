import numpy as np
import librosa
import torch
import numpy as np
import torch
from model.Alexnet import AlexNet
import torch.optim as optim
import os
from torch.utils.data import  DataLoader
from sklearn.model_selection import train_test_split
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from data.data_loader import compute_features_Data, UnifLabelSampler
from model.kmeans import Kmeans_model, stft_cluster_assign
import argparse
writer = SummaryWriter()



def compute_stft_features(dataloader, model, N, device):
    batch_size = 128
    model.eval()

    with torch.no_grad(): 
        for i, batch in enumerate(dataloader):
            inputs = batch["stft_input"].clone().detach().to(device)
            print(inputs.size())
            outputs, _ = model(inputs)
            outputs = outputs.cpu().numpy().astype('float32')

            if i == 0:
                features = np.zeros((N, outputs.shape[1]), dtype='float32')

            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, N)
            features[start_idx:end_idx] = outputs

            if i % 200 == 0:
                print('{0} / {1}\t'
                      .format(i, len(dataloader)))

    return features

    

def val(loader, model, crit, epoch, device):
    """Training of the CNN.
        Args:
            loader (torch.utils.data.DataLoader): Data loader
            model (nn.Module): CNN
            crit (torch.nn): loss
            opt (torch.optim.SGD): optimizer for every parameters with True
                                   requires_grad in model except top layer
            epoch (int)
    """

    # switch to train mode
    model.eval()
    losses = []
    with torch.no_grad():
        for i, (input_tensor, target) in enumerate(loader):
            input_tensor = input_tensor.to(device)
            target = target.to(device)

            _, output = model(input_tensor)
            loss = crit(output, target)

            losses.append(loss.item())
            writer.add_scalar("Loss/val", loss, i + epoch * len(loader))

            if i % 100 == 0:
                print(f"Epoch:{epoch}",
                    f"val_Loss:{loss}")


    return sum(losses)/len(loader)

def train(loader, model, crit, optimizer, epoch, device):
    """Training of the CNN.
        Args:
            loader (torch.utils.data.DataLoader): Data loader
            model (nn.Module): CNN
            crit (torch.nn): loss
            opt (torch.optim.SGD): optimizer for every parameters with True
                                   requires_grad in model except top layer
            epoch (int)
    """

    # switch to train mode
    model.train()

    optimizer_outlayer = optim.Adam(model.output_layer.parameters(),lr=1e-4)

    losses = []


    for i, (input_tensor, target) in enumerate(loader):

        optimizer_outlayer.zero_grad()
        optimizer.zero_grad()

        input_tensor = input_tensor.to(device)
        target = target.to(device)

        _, output = model(input_tensor)
        loss = crit(output, target)

        writer.add_scalar("Loss/train", loss, i + epoch * len(loader))
        losses.append(loss.item())

        loss.backward()
        optimizer_outlayer.step()
        optimizer.step()


        if i % 100 == 0:
            print(f"Epoch:{epoch}",
                  f"Loss:{loss}")


    return sum(losses)/len(loader)





def main(sr = 16000 , batch_size = 128, epochs = 500, workers = 4, pre_numclasses =527, train_ratio = 0.80):
    pre_paths = ["./Premodel_extractlabel/data/Audioset_data"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset =compute_features_Data(pre_paths, sr)
    train_dataset, val_dataset = train_test_split(dataset, train_size=train_ratio, random_state=42)
    train_stft_data = [torch.squeeze(dataset[idx]['stft_input']).numpy() for idx in range(len(dataset))]
    val_stft_data = [torch.squeeze(val_dataset[i]['stft_input']).numpy() for i in range(len(val_dataset))]

    statistic = dataset.statistic

    train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, num_workers=workers)
    deepcluster = Kmeans_model(pre_numclasses)


    model_stft = AlexNet(pre_numclasses)
    fd_stft = model_stft.classifier[-1].out_features
    model_stft.output_layer = None
    model_stft.features = torch.nn.DataParallel(model_stft.features)
    model_stft.to(device)


    optimizer_stft = optim.Adam(model_stft.parameters(), lr=1e-4)

    criterion = nn.CrossEntropyLoss()


    stft_loss = []
    stft_val_loss = []

    for epoch in range(epochs):
        """
        train stftmodel
        """

        # remove head
        model_stft.output_layer = None
        features = compute_stft_features(train_loader, model_stft, len(dataset), device)
        val_features = compute_stft_features(val_loader, model_stft, len(val_dataset), device)
        clustering_loss = deepcluster.cluster(features, val_features, True)
        train_stft_dataset = stft_cluster_assign(deepcluster.images_lists,
                                                  train_stft_data)
        
        sampler_stft = UnifLabelSampler(int(len(dataset)),
                                deepcluster.images_lists)
        train_stft_dataloader = DataLoader(
            train_stft_dataset,
            batch_size=batch_size,
            num_workers=workers,
            sampler=sampler_stft,
            pin_memory=True,
        )


        model_stft.output_layer = nn.Linear(fd_stft, len(deepcluster.images_lists))
        model_stft.output_layer.weight.data.normal_(0, 0.01)
        model_stft.output_layer.bias.data.zero_()
        model_stft.output_layer.to(device)

        loss = train(train_stft_dataloader, model_stft, criterion, optimizer_stft, epoch, device)

        stft_loss.append(loss)

        print(f"stftmodel_loss:{loss}")

        """
        val stftmodel
        """

        val_stft_dataset = stft_cluster_assign(deepcluster.val_lists, val_stft_data, statistic)
        
        val_sampler_stft = UnifLabelSampler(int(len(val_stft_dataset)), deepcluster.val_lists)
        
        val_stft_dataloader = DataLoader(
            val_stft_dataset,
            batch_size=batch_size,
            num_workers=workers,
            sampler=None,
            pin_memory=True,
        )

        val_loss = val(val_stft_dataloader, model_stft, criterion, epoch, device)

        stft_val_loss.append(val_loss)

        print(f"stftmodel_val_loss:{val_loss}")
        writer.add_scalar("total_loss/val", val_loss, epoch)

        print("------------------------------------------------------------------------------------------------------------------------")

        """
        save model
        """
        if (epoch + 1) % 10 == 0:
            PATH = os.path.join('Premodel_extractlabel/model/pre_model_checkpoints/', 'pretrain_model_' + str(epoch+1) + '.pth')
            torch.save({
                'epoch': epoch +1 ,
                'model_state_dict': model_stft.state_dict(),
                'optimizer_state_dict': optimizer_stft.state_dict(),
                'loss': loss,  
            }, PATH)

if __name__ == '__main__':
    main()


