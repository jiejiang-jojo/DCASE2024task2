import torch
import numpy as np
import torch
from model.Alexnet import AlexNet
import torch.nn as nn
import os
from torch.utils.data import  DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from model.loss import ContrastiveLoss
from data.data_loader import finetuning_Data
import argparse
writer = SummaryWriter()



def parse_args():
    parser = argparse.ArgumentParser(description='finetuning model')
    parser.add_argument('--pretrain_model_checkpoint', type=str, required=True, help='pretrain model checkpoint path')
    return parser.parse_args()
              

def train_classification(dataloader,model, contrastiveloss, criterion, optimizer,device,epoch):

    losses = []
    model.train()

    for i, (x1_data,x2_data,x1_label,judge_label) in enumerate(dataloader):

        optimizer.zero_grad()

        input_x1_data = x1_data.to(device)
        input_x2_data = x2_data.to(device)
        target = x1_label.to(device)
        judge_label = judge_label.to(device)

        x1_output,class_output  = model(input_x1_data)
        x2_output, _ = model(input_x2_data)

        loss = 0.01 * contrastiveloss(x1_output, x2_output, judge_label) +  criterion(class_output, target)

        writer.add_scalar("Loss/train", loss, i + epoch * len(dataloader))

        losses.append(loss.item())

        loss.backward()
        optimizer.step()


        if i % 200 == 0:
            print('{0} / {1}\t'
                    .format(i, len(dataloader)))

    return sum(losses)/len(dataloader)


def val_classification(dataloader,model, contrastiveloss, criterion, device,epoch):

    losses = []
    model.eval()

    with torch.no_grad():
        for i, (x1_data,x2_data,x1_label,judge_label) in enumerate(dataloader):


            input_x1_data = x1_data.to(device)
            input_x2_data = x2_data.to(device)
            target = x1_label.to(device)
            judge_label = judge_label.to(device)

            x1_output,class_output  = model(input_x1_data)
            x2_output, _ = model(input_x2_data)

            loss = 0.01 * contrastiveloss(x1_output, x2_output, judge_label) +  criterion(class_output, target)

            losses.append(loss.item())

            writer.add_scalar("Loss/val", loss, i + epoch * len(dataloader))

            if i % 200 == 0:
                print('{0} / {1}\t'
                        .format(i, len(dataloader)))

    return sum(losses)/len(dataloader)





def main(arg, sr = 16000, batch_size = 128, epochs = 60, workers = 4, train_ratio = 0.80):
    pretrain_model_checkpoint = arg.pretrain_model_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dicts = ['dev_data', 'eval_data']
    dev_machine_types = ['bearing', 'ToyCar', 'fan', 'valve']
    add_machine_types = ['3DPrinter', 'HairDryer', 'RoboticArm', 'Scanner', 'ToyCircuit']
    finetuning_paths = []
    for dict in dicts:
        if dict == 'dev_data':
            finetuning_paths.extend([f"Premodel_extractlabel/data/{dict}/{machine_type}/train" for machine_type in  dev_machine_types ])
        elif dict == 'add_data':
            finetuning_paths.extend([f"Premodel_extractlabel/data/{dict}/{machine_type}/train" for machine_type in  add_machine_types ])


    dataset = finetuning_Data(finetuning_paths, sr)

    num_classes = dataset.k

    model_stft = AlexNet(num_classes)
    fd_stft = model_stft.classifier[-1].out_features
    model_stft.output_layer = None
    model_stft.features = torch.nn.DataParallel(model_stft.features)
    model_stft.to(device)

    checkpoint_stft_path = f"{pretrain_model_checkpoint}.pth"
    checkpoint_stft = torch.load(f"/home/jiejiang/Multimodel_deep_clustering/model_checkpoints/{checkpoint_stft_path}")
    keys_to_delete = [key for key in checkpoint_stft['model_state_dict'] if 'output_layer' in key]
    for key in keys_to_delete:
        del checkpoint_stft['model_state_dict'][key]
    model_stft.load_state_dict(checkpoint_stft['model_state_dict'])

    if not os.path.exists(f'Premodel_extractlabel/model/fine_model_checkpoints/{pretrain_model_checkpoint}'):
        os.makedirs(f'Premodel_extractlabel/model/fine_model_checkpoints/{pretrain_model_checkpoint}')

    model_stft.output_layer = nn.Linear(fd_stft, num_classes)
    model_stft.output_layer.weight.data.normal_(0, 0.01)
    model_stft.output_layer.bias.data.zero_()
    model_stft.output_layer.to(device)


    criterion = nn.CrossEntropyLoss()
    contrastiveloss = ContrastiveLoss()

    train_class_dataset, val_class_dataset = train_test_split(dataset, train_size=train_ratio, random_state=42)
    train_class_loader = DataLoader(train_class_dataset, batch_size=batch_size, num_workers=workers)
    val_class_loader = DataLoader(val_class_dataset, batch_size=batch_size, num_workers=workers)

    optimizer = torch.optim.Adam([
    {'params': model_stft.classifier.parameters()},
    {'params': model_stft.output_layer.parameters()}
    ], lr=1e-4)

    stft_class_loss = []
    val_class_loss = []

    for epoch in range(epochs):

        """
        train model
        """

        class_loss = train_classification(train_class_loader, model_stft, contrastiveloss, criterion, optimizer, device,epoch)
        print(f"Epoch{epoch + 1} stft class loss:", class_loss)
        stft_class_loss.append(class_loss)

        """
        train model
        """

        val_loss = val_classification(val_class_loader, model_stft, contrastiveloss, criterion, device,epoch)
        writer.add_scalar("total_loss/val", val_loss, epoch)
        print(f"Epoch{epoch + 1}val class loss:", val_loss)
        val_class_loss.append(val_loss)

        print("------------------------------------------------------------------------------------------------------------------------")

        """
        save model
        """
        if (epoch + 1) % 5 == 0:
            PATH = os.path.join(f'Premodel_extractlabel/model/fine_model_checkpoints/{pretrain_model_checkpoint}', 'finetuning_model_' + str(epoch+1) + '.pth')
            torch.save({
                'epoch': epoch +1 ,
                'model_state_dict': model_stft.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': class_loss,  
            }, PATH)



    print("stft class loss",stft_class_loss )
    print("val class loss",val_class_loss )




if __name__ == '__main__':
    arg = parse_args()
    main(arg)