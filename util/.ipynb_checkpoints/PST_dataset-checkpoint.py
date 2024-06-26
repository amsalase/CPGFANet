import os, torch
from torch.utils.data.dataset import Dataset
import numpy as np
import PIL

class PST_dataset(Dataset):

    def __init__(self, data_dir, split, input_h=480, input_w=640 ,transform=[]):
        super(PST_dataset, self).__init__()

        assert split in ['train', 'val', 'test_a', 'test'], \
            'split must be "train"|"val"|"test"'  # test_day, test_night

        with open(os.path.join(data_dir, split+'.txt'), 'r') as f:
            self.names = [name.strip() for name in f.readlines()]

        self.data_dir  = data_dir
        self.split     = split
        self.input_h   = input_h
        self.input_w   = input_w
        self.transform = transform
        self.n_data    = len(self.names)

    def read_image(self, name, folder):
        file_path = os.path.join(self.data_dir, '%s/%s' % (folder, name))
        image = np.asarray(PIL.Image.open(file_path))
        return image

    def __getitem__(self, index):
        name  = self.names[index]
        image = self.read_image(name, 'rgb')
        thermal = self.read_image(name, 'thermal')
        label = self.read_image(name, 'labels')
        image = np.asarray(PIL.Image.fromarray(image).resize((self.input_w, self.input_h)), dtype=np.float32).transpose((2,0,1))
        thermal = np.asarray(PIL.Image.fromarray(thermal).resize((self.input_w, self.input_h)), dtype=np.float32)[np.newaxis, :]
        label = np.asarray(PIL.Image.fromarray(label).resize((self.input_w, self.input_h), resample=PIL.Image.NEAREST), dtype=np.int64)
        return torch.tensor(image), torch.tensor(thermal), torch.tensor(label), name

    def __len__(self):
        return self.n_data
