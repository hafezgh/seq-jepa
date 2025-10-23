
from torch.utils.data import Dataset
import torch
from torch.nn.functional import grid_sample
import torchvision
from PIL import Image
import numpy as np
from scipy.spatial.transform import Rotation as R
import os

import torch
from torchvision import transforms

from torch.utils.data import Dataset
from augmentations import *
import copy
from utils import generate_cropping_grid
from pathlib import Path

from torchvision import datasets, transforms


############################# 3DIEBench dataset #############################
### Download 3DIEBench images from https://github.com/facebookresearch/SIE/tree/main/data
### Download OOD images from HF TODO
class Dataset_3DIEBench_MultipleViews(Dataset):
    def __init__(self, dataset_root, img_file, labels_file, num_views, latent_type, experience="quat", size_dataset=-1, transform=None, is_eval=False):
        self.dataset_root = dataset_root
        self.num_views = num_views
        self.latent_type = latent_type
        self.is_eval = is_eval
        self.samples = np.load(img_file)
        self.labels = np.load(labels_file)
        if size_dataset > 0:
            self.samples = self.samples[:size_dataset]
            self.labels = self.labels[:size_dataset]
        assert len(self.samples) == len(self.labels)
        self.transform = transform
        self.to_tensor = torchvision.transforms.ToTensor()
        self.experience = experience    

    def get_img(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
            if self.transform:
                img = self.transform(img)
        return img

    def __getitem__(self, i):
        label = self.labels[i]
        dataset_root = Path(self.dataset_root)
        # Latent vector creation
        views = np.random.choice(50, self.num_views, replace=False)
        imgs = [self.get_img(dataset_root / self.samples[i][1:] / f"image_{views[j]}.jpg") for j in range(self.num_views)]
        stacked_imgs = torch.stack(imgs, dim=0)

        # Process latents
        latents = []
        all_latents = []
        all_angles = []
        all_colors = []
        for j in range(self.num_views):
            latent = np.load(dataset_root / self.samples[i][1:] / f"latent_{views[j]}.npy").astype(np.float32)
            angles = latent[:3]
            all_angles.append(angles)
            color = latent[[3,6]]
            all_colors.append(color)
            
            other_params = latent[3:]
            rot = R.from_euler("xyz", angles)
            
            if self.experience == "quat":
                angles = rot.as_quat().astype(np.float32)
            else:
                angles = rot.as_euler("xyz").astype(np.float32)
            
            if self.latent_type == "rot":
                latent_total = angles
            elif self.latent_type == "color":
                latent_total = color
            elif self.latent_type == "rotcolor":
                latent_total = np.concatenate((angles, color))
            elif self.latent_type == "all":
                latent_total = np.concatenate((angles, other_params))
            else:
                raise ValueError("Invalid latent type")
            all_latents.append(np.concatenate((angles, other_params)))
            latents.append(latent_total)

        stacked_latents = torch.FloatTensor(np.stack(latents, axis=0))
        # Calculate relative latents
        relative_latents = []
        for j in range(self.num_views - 1):
            # Calculate relative rotation
            rot_1 = R.from_euler("xyz",all_angles[j])
            rot_2 = R.from_euler("xyz",all_angles[j+1])
            rot_1_to_2 = rot_1.inv()*rot_2
            if self.experience == "quat":
                relative_rotation = rot_1_to_2.as_quat().astype(np.float32)
            else:
                relative_rotation = rot_1_to_2.as_euler("xyz").astype(np.float32)

            relative_color = all_colors[j + 1] - all_colors[j]
            relative_other_params = all_latents[j + 1][3:] - all_latents[j][3:]

            # Concatenate relative components based on latent type
            if self.latent_type == "rot":
                relative_latent_total = relative_rotation
            elif self.latent_type == "color":
                relative_latent_total = relative_color
            elif self.latent_type == "rotcolor":
                relative_latent_total = np.concatenate((relative_rotation, relative_color))
            elif self.latent_type == "all":
                relative_latent_total = np.concatenate((relative_rotation, relative_other_params))
            else:
                raise ValueError("Invalid latent type")

            relative_latents.append(relative_latent_total)

        stacked_rel_latents = torch.FloatTensor(np.stack(relative_latents, axis=0))
        return stacked_imgs, stacked_latents, stacked_rel_latents, label

    def __len__(self):
        return len(self.samples)



############################# CIFAR100 dataset #############################
class CIFAR100_aug(torch.utils.data.Dataset):
    def __init__(self, root, split, num_augs, aug=True, no_blur=False):
        if split == "train":
            self.dataset = torchvision.datasets.CIFAR100(root=root, train=True, download=False, transform=None)
        else:
            self.dataset = torchvision.datasets.CIFAR100(root=root, train=False, download=False, transform=None)
        norm_params = {"mean": [0.5071, 0.4867, 0.4408], "std": [0.2675, 0.2565, 0.2761],}
        self.num_params = 0
        self.aug = aug
        self.no_blur = no_blur
        self.orig_img_trans = torchvision.transforms.Compose([
                torchvision.transforms.Resize((32, 32)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(norm_params["mean"], norm_params["std"]),
            ])
        if self.aug == True:
            if self.no_blur:
                print("CIFAR100: With blur")
                self.trans = ParamCompose([
                    ParamRandomResizedCrop(pflip=0.5, size=(32, 32), scale=(0.2, 1.0), ratio=(3./4., 4./3.), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
                    ParamColorJitter(pjitter=0.8, pgray=0.2, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
                    ParamSolarize(threshold=128.0, p=0.0),
                ], [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(aug_["mean"], aug_["std"]),
                ])
                self.pred_trans = ParamCompose([
                    ParamRandomResizedCrop(pflip=0.5, size=(32, 32), scale=(0.2, 1.0), ratio=(3./4., 4./3.), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
                    ParamColorJitter(pjitter=0.8, pgray=0.2, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
                    ParamSolarize(threshold=128.0, p=0.2),
                ], [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(aug_["mean"], aug_["std"]),
                ])
            else:
                print("CIFAR100: With blur")
                self.trans = ParamCompose([
                    ParamRandomResizedCrop(pflip=0.5, size=(32, 32), scale=(0.2, 1.0), ratio=(3./4., 4./3.), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
                    ParamColorJitter(pjitter=0.8, pgray=0.2, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
                    ParamGaussianBlur(pblur=1.0, kernel_size=tuple(2*round((x - 10)/20) + 1 for x in (32, 32)), sigma=(0.1, 2.0)),
                    ParamSolarize(threshold=128.0, p=0.0),
                ], [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(aug_["mean"], aug_["std"]),
                ])
                self.pred_trans = ParamCompose([
                    ParamRandomResizedCrop(pflip=0.5, size=(32, 32), scale=(0.2, 1.0), ratio=(3./4., 4./3.), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
                    ParamColorJitter(pjitter=0.8, pgray=0.2, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
                    ParamGaussianBlur(pblur=0.1, kernel_size=tuple(2*round((x - 10)/20) + 1 for x in (32, 32)), sigma=(0.1, 2.0)),
                    ParamSolarize(threshold=128.0, p=0.2),
                ], [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(aug_["mean"], aug_["std"]),
                ])
            self.orig_img_trans = torchvision.transforms.Compose([
                torchvision.transforms.Resize((32, 32)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(aug_["mean"], aug_["std"]),
            ])
            self.num_params = self.trans.nb_params
            
        self.aug = aug
        self.num_classes = 100
        self.num_augs = num_augs
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.aug:
            augmented_images = []
            augmented_params = []
            orig_img = self.orig_img_trans(img)
            ### obs sequence
            for _ in range(self.num_augs-1):
                aug_img, aug_params = self.trans(img)
                augmented_images.append(aug_img)
                augmented_params.append(aug_params)
            ### pred
            aug_img, aug_params = self.pred_trans(img)
            augmented_images.append(aug_img)
            augmented_params.append(aug_params)
            
            augmented_images = torch.stack(augmented_images, dim=0)
            augmented_params = torch.stack(augmented_params, dim=0)
            return augmented_images, augmented_params, label, orig_img
        else:
            trans_img = self.orig_img_trans(img)
            return trans_img, label


############################# TinyImageNet dataset #############################
### Download from https://www.kaggle.com/datasets/nikhilshingadiya/tinyimagenet200
class TinyImageNet_aug(Dataset):
    def __init__(self, root_dir, split, num_augs, aug=True):
        """
        TinyImageNetPath
        ├── test
        │   └── images
        │       ├── test_0.JPEG
        │       ├── t...
        │       └── ...
        ├── train
        │   ├── n01443537
        │   │   ├── images
        │   │   │   ├── n01443537_0.JPEG
        │   │   │   ├── n...
        │   │   │   └── ...
        │   │   └── n01443537_boxes.txt
        │   ├── n01629819
        │   │   ├── images
        │   │   │   ├── n01629819_0.JPEG
        │   │   │   ├── n...
        │   │   │   └── ...
        │   │   └── n01629819_boxes.txt
        │   ├── n...
        │   │   ├── images
        │   │   │   ├── ...
        │   │   │   └── ...
        ├── val
        │   ├── images
        │   │   ├── val_0.JPEG
        │   │   ├── v...
        │   │   └── ...
        │   └── val_annotations.txt
        ├── wnids.txt
        └── words.txt
        Args:
            root_dir (string): Directory with all the images.
            split (string): One of 'train', 'val', or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = os.path.join(root_dir, split)
        self.split = split
        self.num_augs = num_augs
        self.images = []
        self.labels = []
        self.wnids_path = os.path.join(root_dir, "wnids.txt")
        self.label_to_idx = self.load_wnids()
        self.mean = [0.4802, 0.4481, 0.3975]
        self.std=[0.2302, 0.2265, 0.2262]
        self.num_classes = 200
        self.aug = aug
        self.num_params = 0
        if self.split == 'train':
            for class_dir in os.listdir(self.root_dir):
                class_path = os.path.join(self.root_dir, class_dir, 'images')
                for img_name in os.listdir(class_path):
                    self.images.append(os.path.join(class_path, img_name))
                    self.labels.append(self.label_to_idx[class_dir])
        elif self.split == 'val':
            with open(os.path.join(root_dir, 'val', 'val_annotations.txt'), 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    img_name = parts[0]
                    img_label = parts[1]
                    self.images.append(os.path.join(root_dir, 'val', 'images', img_name))
                    self.labels.append(self.label_to_idx[img_label])
        elif self.split == 'test':
            self.images = [os.path.join(self.root_dir, 'images', img_name) for img_name in os.listdir(os.path.join(self.root_dir, 'images'))]

        self.orig_img_trans = torchvision.transforms.Compose([
                torchvision.transforms.Resize((64, 64)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(self.mean, self.std),
            ])
        if self.aug == True:
            self.trans = ParamCompose([
                ParamRandomResizedCrop(pflip=0.5, size=(64, 64), scale=(0.2, 1.0), ratio=(3./4., 4./3.), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
                ParamColorJitter(pjitter=0.8, pgray=0.2, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
                ParamGaussianBlur(pblur=1.0, kernel_size=tuple(2*round((x - 10)/20) + 1 for x in (64, 64)), sigma=(0.1, 2.0)),
                ParamSolarize(threshold=128.0, p=0.0),
            ], [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(self.mean, self.std),
            ])
            self.pred_trans = ParamCompose([
                ParamRandomResizedCrop(pflip=0.5, size=(64, 64), scale=(0.2, 1.0), ratio=(3./4., 4./3.), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
                ParamColorJitter(pjitter=0.8, pgray=0.2, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
                ParamGaussianBlur(pblur=0.1, kernel_size=tuple(2*round((x - 10)/20) + 1 for x in (64, 64)), sigma=(0.1, 2.0)),
                ParamSolarize(threshold=128.0, p=0.2),
            ], [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(self.mean, self.std),
            ])
            self.orig_img_trans = torchvision.transforms.Compose([
                torchvision.transforms.Resize((64, 64)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(self.mean, self.std),
            ])
            self.num_params = self.trans.nb_params
    def load_wnids(self):
        label_to_idx = {}
        with open(self.wnids_path, 'r') as file:
            for idx, line in enumerate(file):
                wnid = line.strip()
                label_to_idx[wnid] = idx
        return label_to_idx

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        if self.split == "test":
            label = -1
        else:
            label = self.labels[idx]
        img = Image.open(img_name).convert('RGB')
        if self.aug:
            augmented_images = []
            augmented_params = []
            orig_img = self.orig_img_trans(img)
            ### obs sequence
            for _ in range(self.num_augs):
                aug_img, aug_params = self.trans(img)
                augmented_images.append(aug_img)
                augmented_params.append(aug_params)
            ### pred            
            augmented_images = torch.stack(augmented_images, dim=0)
            augmented_params = torch.stack(augmented_params, dim=0)
            return augmented_images, augmented_params, label, orig_img
        else:
            trans_img = self.orig_img_trans(img)
            return trans_img, label
        

############################# STL10 SalMap dataset #############################
### Download STL10 images from https://cs.stanford.edu/~acoates/stl10/
### Download STL10 saliency maps from hugging face TODO
class STL10_SalMap(Dataset):
    def __init__(self, data_path, sal_path, split, num_patches, use_sal, ior):
        """
            data_path (string): Path to the directory containing the dataset files.
            sal_path (string): Path to the directory containing the saliency maps.
            split (string): One of {'train', 'test', 'unlabeled'} to specify the dataset split.
            num_patches (int): Number of patches to sample from the image.
            use_sal (bool): Whether to use saliency map for sampling patch centers.
            ior (bool): Whether to use inhibition of return when sampling patch centers.
        """
        self.dataset = torchvision.datasets.STL10(root=data_path, split=split, download=False)
        self.sal_path = os.path.join(sal_path, split)
        if split == 'train':
            self.sal_path = os.path.join(self.sal_path, 'saliency_train.npy')
        elif split == 'test':
            self.sal_path = os.path.join(self.sal_path, 'saliency_test.npy')
        elif split == 'unlabeled':
            self.sal_path = os.path.join(self.sal_path, 'saliency_unlabeled.npy')
        else:
            raise ValueError("Invalid split")
        self.saliency_maps = np.load(self.sal_path)
        assert len(self.dataset) == len(self.saliency_maps), "Mismatch between dataset size and saliency maps"
        self.use_sal = use_sal
        self.ior = ior
        self.num_patches = num_patches
        
        norm_params = {"mean": [0.4467, 0.4398, 0.4066], "std": [0.2241, 0.2215, 0.2239],}
        self.transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize(mean=norm_params["mean"], std=norm_params["std"]),])
        self.full_img_trans = transforms.Compose([transforms.Resize((96, 96)), transforms.ToTensor(), transforms.Normalize(mean=norm_params["mean"], std=norm_params["std"]),])
        self.split = split

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        sal_map = self.saliency_maps[idx]    # Shape: (1, 96, 96)
        
        sal_map = torch.from_numpy(sal_map).float()
        if len(sal_map.shape) == 2:
            sal_map = sal_map.unsqueeze(0)
        
        sac_positions = None
        patches = None

        rgb_image = image
        image_size = 96
        fovea_size = 32
        sac_positions = []
        inh_radius = fovea_size//2
        trans_hflip = transforms.functional.hflip
        trans_rot = transforms.functional.rotate
        img_full = transforms.ToTensor()(rgb_image)
        img_full = img_full.unsqueeze(0)
        probs_inh = copy.deepcopy(sal_map)
        probs_inh = probs_inh.unsqueeze(0)
        for _ in range(self.num_patches):
            if self.use_sal:
                probs_flat = probs_inh.view(probs_inh.size(0), -1)
                sac_position = torch.multinomial(probs_flat, num_samples=1)
                sac_positions.append(sac_position)
            else:
                probs_flat = probs_inh.view(probs_inh.size(0), -1)
                num_elements = probs_flat.size(1)  # Number of elements in each row of probs_flat
                uniform_value = 1.0 / num_elements
                uniform_dist_tensor = torch.full(probs_flat.size(), uniform_value)
                sac_position = torch.multinomial(uniform_dist_tensor, num_samples=1)
                sac_positions.append(sac_position)
            # Calculate row and column from sac_position
            row = sac_position // image_size
            col = sac_position % image_size
            # Create inhibition mask
            if self.ior:
                rows = torch.arange(probs_inh.size(2)).view(1, 1, -1, 1)
                cols = torch.arange(probs_inh.size(3)).view(1, 1, 1, -1)
                row_mask = (rows >= (row - inh_radius).unsqueeze(2)) & (rows <= (row + inh_radius).unsqueeze(2))
                col_mask = (cols >= (col - inh_radius).unsqueeze(2)) & (cols <= (col + inh_radius).unsqueeze(2))
                inh_mask = row_mask & col_mask
                inh_mask = inh_mask.permute(1, 0, 2, 3) 
                probs_inh = probs_inh * (1 - inh_mask.float()) + 1e-16
                probs_inh = probs_inh / probs_inh.sum(dim=(2, 3), keepdim=True)
        # Concatenate and shuffle the order of saccade positions
        sac_positions = torch.cat(sac_positions, dim=1)

        ### Get patches
        rows = sac_positions // image_size
        cols = sac_positions % image_size
        centers = torch.stack((rows, cols), dim=2)
        
        grid_centers = centers.reshape(-1, 2)
        grid = generate_cropping_grid(image_size, fovea_size, grid_centers, "cpu")
        repeated_batch = img_full.unsqueeze(1).repeat(1, self.num_patches, 1, 1, 1)
        repeated_batch = repeated_batch.reshape(-1, 3, image_size, image_size)
        foveated_x = grid_sample(repeated_batch, grid, mode='bilinear', padding_mode='zeros')
        foveated_x = trans_rot(foveated_x,-90)
        foveated_x = trans_hflip(foveated_x)
        foveated_x = foveated_x.reshape((-1,3,fovea_size,fovea_size))
        foveated_x_ = foveated_x.reshape((self.num_patches, 3, fovea_size, fovea_size))
        
        sac_pos_norm = centers / image_size
        sac_pos_norm_reshaped = sac_pos_norm.reshape((self.num_patches, 2))
        to_pil = transforms.ToPILImage()
        patches = []
        for img_tensor in foveated_x_:
            # Convert tensor to PIL Image
            img_pil = to_pil(img_tensor)
            patch = self.transform(img_pil)
            patches.append(patch)
        patches = torch.stack(patches, dim=0)
        rgb_data = self.full_img_trans(rgb_image)
        sac_positions = sac_pos_norm_reshaped

        return rgb_data, sal_map, label, patches, sac_positions

        
############################# ImageNet1k SalMap dataset #############################
### Download ImageNet1k images from https://huggingface.co/datasets/ILSVRC/imagenet-1k
### Download ImageNet1k saliency maps from hugging face TODO
class Imagenet1k_Sal(Dataset):
    def __init__(self, data_path_img, data_path_sal, split, full_img_transform, full_img_size=224, patch_size=64,
                 num_patches=5, use_sal=True, ior=True):
        """
        Args:
            data_path_img (string): Path to the directory containing ImageNet images.
            data_path_sal (string): Path to the directory containing the saliency maps.
            split (string): 'train' or 'val'.
            num_patches (int): Number of patches (saccades) to sample.
            use_sal (bool): Whether to use saliency map for sampling patch centers.
            ior (bool): Whether to use inhibition of return when sampling patch centers.
            use_sal (bool): If True, sample saccade positions using the saliency map distribution.
            ior (bool): If True, apply inhibition-of-return when sampling.
            full_img_size (int): Size (height and width) of the full image after full-image transform.
            patch_size (int): Size (height and width) of the extracted patches.
        """
        self.data_path_img = os.path.join(data_path_img, split)
        self.data_path_sal = os.path.join(data_path_sal, split)
        self.split = split
        self.num_patches = num_patches
        self.use_sal = use_sal
        self.ior = ior
        self.full_img_size = full_img_size
        self.patch_size = patch_size
        self.full_img_trans = transform
        
        # Build file list and labels from the folder structure.
        self.class_folders = [d for d in os.listdir(self.data_path_img)
                              if os.path.isdir(os.path.join(self.data_path_img, d))]
        self.file_list = []
        self.labels = []
        for class_idx, class_folder in enumerate(self.class_folders):
            img_folder = os.path.join(self.data_path_img, class_folder)
            sal_folder = os.path.join(self.data_path_sal, class_folder)
            for file in os.listdir(img_folder):
                if file.lower().endswith(('.jpeg', '.jpg', '.png')):
                    img_id = file.split('.')[0]
                    sal_file = f"{img_id}_sal.npy"
                    if os.path.exists(os.path.join(sal_folder, sal_file)):
                        self.file_list.append((os.path.join(class_folder, file),
                                               os.path.join(class_folder, sal_file)))
                        self.labels.append(class_idx)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Load the image and corresponding saliency map.
        img_rel_path, sal_rel_path = self.file_list[idx]
        img_path = os.path.join(self.data_path_img, img_rel_path)
        sal_path = os.path.join(self.data_path_sal, sal_rel_path)
        
        with Image.open(img_path) as img:
            rgb_image = img.convert('RGB')
        sal_map = np.load(sal_path)
        sal_map = torch.from_numpy(sal_map).float()
        if sal_map.ndim == 2:
            sal_map = sal_map.unsqueeze(0)  # ensure shape [1, H, W]
        label = self.labels[idx]
        
        # Apply the full image transform to get a normalized full image for patch extraction.
        rgb_data = self.full_img_trans(rgb_image)  # shape [3, full_img_size, full_img_size]
        
        image_size = self.full_img_size  # e.g., 224
        patch_size = self.patch_size       # e.g., 32 or 84
        inh_radius = patch_size // 2

        img_full = rgb_data.unsqueeze(0)  # [1, 3, image_size, image_size]
        
        # Copy the saliency map (assumed to be of shape [1, image_size, image_size]) for sampling.
        probs_inh = copy.deepcopy(sal_map)  # shape [1, image_size, image_size]
        probs_inh = probs_inh.unsqueeze(0)  # shape [1, 1, image_size, image_size]
        
        sac_positions = []
        # Sample num_patches saccade positions.
        for _ in range(self.num_patches):
            if self.use_sal:
                probs_flat = probs_inh.view(probs_inh.size(0), -1)
                sac_position = torch.multinomial(probs_flat, num_samples=1)
                sac_positions.append(sac_position)
            else:
                probs_flat = probs_inh.view(probs_inh.size(0), -1)
                num_elements = probs_flat.size(1)
                uniform_value = 1.0 / num_elements
                uniform_dist_tensor = torch.full(probs_flat.size(), uniform_value)
                sac_position = torch.multinomial(uniform_dist_tensor, num_samples=1)
                sac_positions.append(sac_position)
            # Calculate row and column from the flattened index.
            row = sac_position // image_size
            col = sac_position % image_size
            # If inhibition-of-return (IOR) is enabled, update the distribution.
            if self.ior:
                rows = torch.arange(probs_inh.size(2)).view(1, 1, -1, 1)
                cols = torch.arange(probs_inh.size(3)).view(1, 1, 1, -1)
                row_val = row.float().unsqueeze(2).unsqueeze(3)
                col_val = col.float().unsqueeze(2).unsqueeze(3)
                row_mask = (rows.float() >= (row_val - inh_radius)) & (rows.float() <= (row_val + inh_radius))
                col_mask = (cols.float() >= (col_val - inh_radius)) & (cols.float() <= (col_val + inh_radius))
                inh_mask = row_mask & col_mask
                inh_mask = inh_mask.float()
                probs_inh = probs_inh * (1 - inh_mask) + 1e-16
                probs_inh = probs_inh / probs_inh.sum(dim=(2, 3), keepdim=True)
        
        # Concatenate and optionally shuffle saccade positions.
        sac_positions = torch.cat(sac_positions, dim=1)  # shape [1, num_patches]

        
        # Compute row and column coordinates.
        rows = sac_positions // image_size
        cols = sac_positions % image_size
        centers = torch.stack((rows, cols), dim=2)  # shape [1, num_patches, 2]
        sac_pos_norm = centers.float() / image_size  # normalized to [0, 1]
        sac_pos_norm = sac_pos_norm.reshape((self.num_patches, 2))
        
        # Generate cropping grid using centers.
        grid_centers = centers.reshape(-1, 2)
        # generate_cropping_grid should create a grid for each center given the image and patch sizes.
        grid = generate_cropping_grid(image_size, patch_size, grid_centers, device="cpu")
        
        # Extract patches using grid_sample.
        repeated_batch = img_full.unsqueeze(1).repeat(1, self.num_patches, 1, 1, 1)
        repeated_batch = repeated_batch.reshape(-1, 3, image_size, image_size)
        foveated_x = grid_sample(repeated_batch, grid, mode='bilinear', padding_mode='zeros')
        foveated_x = transforms.functional.rotate(foveated_x, -90)
        foveated_x = transforms.functional.hflip(foveated_x)
        foveated_x = foveated_x.reshape((-1, 3, patch_size, patch_size))
        foveated_x = foveated_x.reshape((self.num_patches, 3, patch_size, patch_size))

        patches = [patch_tensor for patch_tensor in foveated_x]
        patches = torch.stack(patches, dim=0)
        
        return rgb_data, sal_map, label, patches, sac_pos_norm
    
