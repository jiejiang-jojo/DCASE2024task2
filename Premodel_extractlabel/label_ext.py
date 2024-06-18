import numpy as np
import time
import faiss
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import torch
from model.Alexnet import AlexNet
import os
# from DataLoad import StartData, UnifLabelSampler
from tqdm import tqdm
import pandas as pd
from data.data_loader import compute_features_Data
from model.kmeans import Kmeans_model
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='extract label files')
    parser.add_argument('--pretrain_model_checkpoint', type=str, required=True, help='pretrain model checkpoint path')
    parser.add_argument('--finetuning_model_checkpoint', type=str, required=True, help='finetuning_model_checkpoint')
    parser.add_argument('--num_classes', type=int, required=True, help='classes number')
    return parser.parse_args()
    
def file_label(images_lists,file_lists):

    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))
    label = []
    for j, idx in enumerate(image_indexes):
        wav = file_lists[idx]
        pseudolabel = pseudolabels[j]
        label.append([wav, pseudolabel])
    return label
    


def compute_stft_features(dataloader, model, N, device):
    batch_size = 128
    model.eval()

    with torch.no_grad(): 
        for i, batch in enumerate(dataloader):
            inputs = batch["stft_input"].clone().detach().to(device)
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


def main(arg, sr = 16000, batch_size = 128, workers = 4, num_classes = 26):
    pretrain_model_checkpoint = arg.pretrain_model_checkpoint
    finetuning_model_checkpoint = arg.finetuning_model_checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dicts = ['dev_data', 'eval_data']
    dev_machine_types = ['slider', 'gearbox', 'ToyTrain']
    add_machine_types = ['AirCompressor', 'BrushlessMotor', 'HoveringDrone', 'ToothBrush']
    ext_paths = []
    for dict in dicts:
        if dict == 'dev_data':
            ext_paths.extend([f"Premodel_extractlabel/data/{dict}/{machine_type}/train" for machine_type in  dev_machine_types ])
        elif dict == 'add_data':
            ext_paths.extend([f"Premodel_extractlabel/data/{dict}/{machine_type}/train" for machine_type in  add_machine_types ])

    if not os.path.exists(f'Premodel_extractlabel/extract_labels/{pretrain_model_checkpoint}/{finetuning_model_checkpoint}'):
        os.makedirs(f'Premodel_extractlabel/extract_labels/{pretrain_model_checkpoint}/{finetuning_model_checkpoint}')


    for dev_path in ext_paths:
        category =dev_path.split('/')[-2]
        print(category)
        dataset = compute_features_Data([dev_path], sr)

        data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers)
        deepcluster = Kmeans_model(num_classes)


        model_stft = AlexNet(num_classes)
        model_stft.output_layer = None
        model_stft.features = torch.nn.DataParallel(model_stft.features)
        model_stft.to(device)



        checkpoint_path = f"{finetuning_model_checkpoint}.pth"
        checkpoint = torch.load(f"/home/jiejiang/Multimodel_deep_clustering/model_checkpoints/{checkpoint_path}")
        keys_to_delete = [key for key in checkpoint['model_state_dict'] if 'output_layer' in key]
        for key in keys_to_delete:
            del checkpoint['model_state_dict'][key]
        model_stft.load_state_dict(checkpoint['model_state_dict'])


        features = compute_stft_features(data_loader, model_stft, len(dataset), device)

        clustering_loss = deepcluster.cluster_ectractlabels(features, True)

        header = ['file_name', 'pse_label']
        keamslabel = file_label(deepcluster.images_lists,dataset.files_list)
        csv_file = f"Premodel_extractlabel/extract_labels/{pretrain_model_checkpoint}/{finetuning_model_checkpoint}/{category}_labels_{num_classes}classes.csv"
        df = pd.DataFrame(keamslabel, columns=header)
        df.to_csv(csv_file, index=False)
        print(f"{category} machine label extract completely")
        


if __name__ == '__main__':
    arg = parse_args()
    main(arg)
    








    


    







