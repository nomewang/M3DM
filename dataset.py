import os
from PIL import Image
from torchvision import transforms
import glob
from torch.utils.data import Dataset
from utils.mvtec3d_util import *
from torch.utils.data import DataLoader
import numpy as np

def eyecandies_classes():
    return [
        'CandyCane',
        'ChocolateCookie',
        'ChocolatePraline',
        'Confetto',
        'GummyBear',
        'HazelnutTruffle',
        'LicoriceSandwich',
        'Lollipop',
        'Marshmallow',
        'PeppermintCandy',   
    ]

def mvtec3d_classes():
    return [
        "bagel",
        "cable_gland",
        "carrot",
        "cookie",
        "dowel",
        "foam",
        "peach",
        "potato",
        "rope",
        "tire",
    ]

RGB_SIZE = 224

class BaseAnomalyDetectionDataset(Dataset):

    def __init__(self, split, class_name, img_size, dataset_path='datasets/eyecandies_preprocessed'):
        self.IMAGENET_MEAN = [0.485, 0.456, 0.406]
        self.IMAGENET_STD = [0.229, 0.224, 0.225]
        self.cls = class_name
        self.size = img_size
        self.img_path = os.path.join(dataset_path, self.cls, split)
        self.rgb_transform = transforms.Compose(
            [transforms.Resize((RGB_SIZE, RGB_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
             transforms.ToTensor(),
             transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)])

class PreTrainTensorDataset(Dataset):
    def __init__(self, root_path):
        super().__init__()
        self.root_path = root_path
        self.tensor_paths = os.listdir(self.root_path)


    def __len__(self):
        return len(self.tensor_paths)

    def __getitem__(self, idx):
        tensor_path = self.tensor_paths[idx]

        tensor = torch.load(os.path.join(self.root_path, tensor_path))

        label = 0

        return tensor, label

class TrainDataset(BaseAnomalyDetectionDataset):
    def __init__(self, class_name, img_size, dataset_path='datasets/eyecandies_preprocessed'):
        super().__init__(split="train", class_name=class_name, img_size=img_size, dataset_path=dataset_path)
        self.img_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        img_tot_paths = []
        tot_labels = []
        rgb_paths = glob.glob(os.path.join(self.img_path, 'good', 'rgb') + "/*.png")
        tiff_paths = glob.glob(os.path.join(self.img_path, 'good', 'xyz') + "/*.tiff")
        rgb_paths.sort()
        tiff_paths.sort()
        sample_paths = list(zip(rgb_paths, tiff_paths))
        img_tot_paths.extend(sample_paths)
        tot_labels.extend([0] * len(sample_paths))
        return img_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx], self.labels[idx]
        rgb_path = img_path[0]
        tiff_path = img_path[1]
        img = Image.open(rgb_path).convert('RGB')

        img = self.rgb_transform(img)
        organized_pc = read_tiff_organized_pc(tiff_path)
        
        depth_map_3channel = np.repeat(organized_pc_to_depth_map(organized_pc)[:, :, np.newaxis], 3, axis=2)
        resized_depth_map_3channel = resize_organized_pc(depth_map_3channel)
        resized_organized_pc = resize_organized_pc(organized_pc, target_height=self.size, target_width=self.size)
        resized_organized_pc = resized_organized_pc.clone().detach().float()

        return (img, resized_organized_pc, resized_depth_map_3channel), label


class TestDataset(BaseAnomalyDetectionDataset):
    def __init__(self, class_name, img_size, dataset_path='datasets/eyecandies_preprocessed'):
        super().__init__(split="test", class_name=class_name, img_size=img_size, dataset_path=dataset_path)
        self.gt_transform = transforms.Compose([
            transforms.Resize((RGB_SIZE, RGB_SIZE), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()])
        self.img_paths, self.gt_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                rgb_paths = glob.glob(os.path.join(self.img_path, defect_type, 'rgb') + "/*.png")
                tiff_paths = glob.glob(os.path.join(self.img_path, defect_type, 'xyz') + "/*.tiff")
                rgb_paths.sort()
                tiff_paths.sort()
                sample_paths = list(zip(rgb_paths, tiff_paths))
                img_tot_paths.extend(sample_paths)
                gt_tot_paths.extend([0] * len(sample_paths))
                tot_labels.extend([0] * len(sample_paths))
            else:
                rgb_paths = glob.glob(os.path.join(self.img_path, defect_type, 'rgb') + "/*.png")
                tiff_paths = glob.glob(os.path.join(self.img_path, defect_type, 'xyz') + "/*.tiff")
                gt_paths = glob.glob(os.path.join(self.img_path, defect_type, 'gt') + "/*.png")
                rgb_paths.sort()
                tiff_paths.sort()
                gt_paths.sort()
                sample_paths = list(zip(rgb_paths, tiff_paths))

                img_tot_paths.extend(sample_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(sample_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label = self.img_paths[idx], self.gt_paths[idx], self.labels[idx]
        rgb_path = img_path[0]
        tiff_path = img_path[1]
        img_original = Image.open(rgb_path).convert('RGB')
        img = self.rgb_transform(img_original)

        organized_pc = read_tiff_organized_pc(tiff_path)
        depth_map_3channel = np.repeat(organized_pc_to_depth_map(organized_pc)[:, :, np.newaxis], 3, axis=2)
        resized_depth_map_3channel = resize_organized_pc(depth_map_3channel)
        resized_organized_pc = resize_organized_pc(organized_pc, target_height=self.size, target_width=self.size)
        resized_organized_pc = resized_organized_pc.clone().detach().float()
        

        

        if gt == 0:
            gt = torch.zeros(
                [1, resized_depth_map_3channel.size()[-2], resized_depth_map_3channel.size()[-2]])
        else:
            gt = Image.open(gt).convert('L')
            gt = self.gt_transform(gt)
            gt = torch.where(gt > 0.5, 1., .0)

        return (img, resized_organized_pc, resized_depth_map_3channel), gt[:1], label, rgb_path


def get_data_loader(split, class_name, img_size, args):
    if split in ['train']:
        dataset = TrainDataset(class_name=class_name, img_size=img_size, dataset_path=args.dataset_path)
    elif split in ['test']:
        dataset = TestDataset(class_name=class_name, img_size=img_size, dataset_path=args.dataset_path)

    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False,
                             pin_memory=True)
    return data_loader
