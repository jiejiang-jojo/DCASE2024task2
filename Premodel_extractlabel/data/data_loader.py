import numpy as np
import librosa
import librosa.display
import os
from tqdm import tqdm
from torch.utils.data import Dataset
import torch
from torch.utils.data.sampler import Sampler
from sklearn.preprocessing import LabelEncoder
import random



def adjust_size(audio, new_size):
    if len(audio) > new_size:
        audio = audio[:new_size]

    if len(audio) < new_size:
        pad_length = new_size - len(audio)
        audio = np.pad(audio, (0, pad_length), mode='constant')
        audio= audio[:new_size]
    return audio

def wav_load(sr, dev_paths):
    wav = []
    files_list= []
    for dev_path in dev_paths:
        wave_fname_list = os.listdir(dev_path)
        for i, fname in enumerate(tqdm(wave_fname_list)):
            audio_path = os.path.join(dev_path, fname)
            try:
                # duration = math.floor(librosa.get_duration(filename=audio_path))
                audio, _ = librosa.load(audio_path, sr=sr) #duration=duration
                audio = adjust_size(audio, 160000)  # 将音频重复并裁剪到288000
                wav.append(audio)
                files_list.append(fname)
            except ValueError as e:
                print("Error loading audio:", e)
                continue
    wav_array = np.array(wav)  
    statistic = {"mean": np.mean(wav_array, axis=0), "std": np.std(wav_array, axis=0)}
    return wav, statistic, files_list

def fine_tuning_wav_load(sr, dev_paths):
    wav = []
    files_list= []
    wav_ids= []
    wav_domains= []
    wav_atts= []
    for dev_path in dev_paths:
        category = dev_path.split('/')[-2]
        wave_fname_list = os.listdir(dev_path)
        print(category)
        for i, fname in enumerate(tqdm(wave_fname_list)):
            audio_path = os.path.join(dev_path, fname)
            try:
                audio, _ = librosa.load(audio_path, sr=sr) #duration=duration
                audio = adjust_size(audio, 160000)  # 将音频重复并裁剪到288000
                wav.append(audio)
                files_list.append(fname)
                wav_ids.append(category + '_' + fname.split('_')[1])
                wav_domains.append(fname.split('_')[2])
                wav_atts.append('_'.join(fname.split('.wav')[0].split('_')[6:]))
            except ValueError as e:
                print("Error loading audio:", e)
                continue
    wav_array = np.array(wav)  
    statistic = {"mean": np.mean(wav_array, axis=0), "std": np.std(wav_array, axis=0)}
    wav_truelabels = np.array(['###'.join([wav_ids[k], wav_atts[k], wav_domains[k]]) for k in np.arange(len(wav_ids))]) 
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(wav_truelabels)
    return wav, statistic, files_list, labels, len(label_encoder.classes_)


class compute_features_Data(Dataset):
    def __init__(self, file_path, sr):
        self.file_path = file_path
        self.data, self.statistic, self.files_list= wav_load(sr, self.file_path)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        x = self.data[idx]
        x = (x - self.statistic["mean"]) / self.statistic["std"]
        wav_x = np.array([x]).astype(np.float32)
        D = librosa.stft(x)
        D_amp = np.abs(D)
        stft_rgb = np.stack([D_amp] * 3, axis=0)
        stft_rgb = torch.tensor(stft_rgb.astype(np.float32))

        sample = {
            'stft_input': torch.Tensor(stft_rgb),
            'wav_input': torch.Tensor(wav_x)
        }
        return sample

class UnifLabelSampler(Sampler):
    """Samples elements uniformely accross pseudolabels.
        Args:
            N (int): size of returned iterator.
            images_lists: dict of key (target), value (list of data with this target)
    """

    def __init__(self, N, images_lists):
        self.N = N
        self.images_lists = images_lists
        self.indexes = self.generate_indexes_epoch()

    def generate_indexes_epoch(self):
        nmb_non_empty_clusters = 0
        for i in range(len(self.images_lists)):
            if len(self.images_lists[i]) != 0:
                nmb_non_empty_clusters += 1

        size_per_pseudolabel = int(self.N / nmb_non_empty_clusters) + 1
        res = np.array([])

        for i in range(len(self.images_lists)):
            # skip empty clusters
            if len(self.images_lists[i]) == 0:
                continue
            indexes = np.random.choice(
                self.images_lists[i],
                size_per_pseudolabel,
                replace=(len(self.images_lists[i]) <= size_per_pseudolabel)
            )
            res = np.concatenate((res, indexes))

        np.random.shuffle(res)
        res = list(res.astype('int'))
        if len(res) >= self.N:
            return res[:self.N]
        res += res[: (self.N - len(res))]
        return res

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return len(self.indexes)
    

class Pseudolabel_Dataset(Dataset):
    def __init__(self, image_indexes, pseudolabels, dataset):
        self.imgs = self.make_dataset(image_indexes, pseudolabels, dataset)

    def make_dataset(self, image_indexes, pseudolabels, dataset):

        images = []
        for j, idx in enumerate(image_indexes):
            path = dataset[idx]
            pseudolabel = pseudolabels[j]
            images.append((path, pseudolabel))
        return images
    
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        stft_rgb, pseudolabel = self.imgs[index]

        return stft_rgb, pseudolabel
    

class finetuning_Data(Dataset):
    def __init__(self, file_path, sr):
        self.file_path = file_path
        self.data, self.statistic, self.files_list, self.wav_truelabels, self.k= fine_tuning_wav_load(sr, self.file_path)
        self.datasets = self.make_dataset()

    def stft_audio(self):
        stft_data = []
        for i in range(len(self.data)):
            x = self.data[i]
            x = (x - self.statistic["mean"]) / self.statistic["std"]
            D = librosa.stft(x)
            D_amp = np.abs(D)
            stft_rgb = np.stack([D_amp] * 3, axis=0)
            stft_rgb = torch.tensor(stft_rgb.astype(np.float32))
            stft_data.append(stft_rgb)
        return stft_data


    def make_dataset(self):
        audios = []
        data = self.stft_audio()
        random_idx = [i for i in range(len(self.data))]
        random.shuffle(random_idx)
        for i in range(len(self.data)):
            x1_data = data[i]
            x1_label = self.wav_truelabels[i]
            x2_data = data[random_idx[i]]
            x2_label = self.wav_truelabels[random_idx[i]]

            judge_label = int(x1_label == x2_label) 

            audios.append((x1_data,x2_data,x1_label,judge_label))
        
        return audios
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        x1_data,x2_data,x1_label,judge_label = self.datasets[index]

        return x1_data,x2_data,x1_label,judge_label