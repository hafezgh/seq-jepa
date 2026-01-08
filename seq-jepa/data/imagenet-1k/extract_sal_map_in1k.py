import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from scipy.special import logsumexp
from scipy.ndimage import zoom
import deepgaze_pytorch
import sys
from PIL import Image
import time  

class ImageNetDatasetWithNames(Dataset):
    def __init__(self, root_dir, split, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        if split == 'train':
            self.image_dir = os.path.join(root_dir, 'train')
        elif split == 'val':
            self.image_dir = os.path.join(root_dir, 'val')
        else:
            raise ValueError("Split must be either 'train' or 'val'")

        self.image_list = []
        for subdir in os.listdir(self.image_dir):
            subdir_path = os.path.join(self.image_dir, subdir)
            if os.path.isdir(subdir_path):
                self.image_list.extend([
                    os.path.join(subdir, f)
                    for f in os.listdir(subdir_path)
                    if f.lower().endswith(('.jpeg', '.jpg', '.png'))
                ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_list[idx])
        image = Image.open(img_path).convert('RGB')
        original_size = image.size 
        if self.transform:
            image = self.transform(image)
        return image, self.image_list[idx], original_size

def process_imagenet(train_output_dir, val_output_dir):
    overall_start = time.time()
    total_images = len(train_set) 

    # Process the training set
    with torch.no_grad():
        for idx, (image, batch_name, original_size) in enumerate(
            tqdm(train_loader, desc="Processing Training Images", unit="img")
        ):
            image = image.to(DEVICE)


            # load precomputed centerbias log density (from MIT1003) over a 1024x1024 image
            # alternatively, you can use a uniform centerbias via `centerbias_template = np.zeros((1024, 1024))`.
            centerbias_template = np.load('/home/hafezgh/seq-jepa-dev/data/centerbias_mit1003.npy')
            # rescale to match image size
            centerbias = zoom(centerbias_template, (image.shape[-2]/centerbias_template.shape[0], image.shape[-1]/centerbias_template.shape[1]), order=0, mode='nearest')
            # renormalize log density
            centerbias -= logsumexp(centerbias)

            centerbias_tensor = torch.tensor([centerbias]).to(DEVICE)

            log_density_prediction = model(image, centerbias_tensor)
            # Convert from log density to probability map
            prob_map = torch.exp(log_density_prediction)
            prob_map = prob_map / prob_map.sum(dim=(-2, -1), keepdim=True)
            # Convert to numpy (assume single-channel saliency) and save as float32
            prob_map_np = prob_map.cpu().numpy().astype(np.float32)[0, 0]

            # Reconstruct output folder and filename (append "-sal" before extension)
            subdir = os.path.dirname(batch_name[0])
            out_dir = os.path.join(train_output_dir, subdir)
            os.makedirs(out_dir, exist_ok=True)
            base, ext = os.path.splitext(os.path.basename(batch_name[0]))
            sal_filename = base + '-sal.npy'
            np.save(os.path.join(out_dir, sal_filename), prob_map_np)

        print(f"Processed and saved {len(train_set)} training images to {train_output_dir}")

    # Process the validation set
    with torch.no_grad():
        total_val_images = len(val_set)
        overall_start_val = time.time()
        for idx, (image, batch_name, original_size) in enumerate(
            tqdm(val_loader, desc="Processing Validation Images", unit="img")
        ):
            image = image.to(DEVICE)

            # load precomputed centerbias log density (from MIT1003) over a 1024x1024 image
            centerbias_template = np.load('/home/hafezgh/seq-jepa-dev/data/centerbias_mit1003.npy')
            # rescale to match image size
            centerbias = zoom(centerbias_template, (image.shape[-2]/centerbias_template.shape[0], image.shape[-1]/centerbias_template.shape[1]), order=0, mode='nearest')
            # renormalize log density
            centerbias -= logsumexp(centerbias)

            centerbias_tensor = torch.tensor([centerbias]).to(DEVICE)

            log_density_prediction = model(image, centerbias_tensor)
            # Convert from log density to probability map
            prob_map = torch.exp(log_density_prediction)
            prob_map = prob_map / prob_map.sum(dim=(-2, -1), keepdim=True)
            # Convert to numpy (assume single-channel saliency) and save as float32
            prob_map_np = prob_map.cpu().numpy().astype(np.float32)[0, 0]

            subdir = os.path.dirname(batch_name[0])
            out_dir = os.path.join(val_output_dir, subdir)
            os.makedirs(out_dir, exist_ok=True)
            base, ext = os.path.splitext(os.path.basename(batch_name[0]))
            sal_filename = base + '-sal.npy'
            np.save(os.path.join(out_dir, sal_filename), prob_map_np)

        print(f"Processed and saved {len(val_set)} validation images to {val_output_dir}")


if __name__ == '__main__':
    DEVICE = 'cuda'
    # Use batch_size=1 to accommodate variable native image sizes.
    batch_size = 1

    sys.path.append('/scratch/users/hafezgh/imagenet')

    model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(DEVICE)
    model.eval()

    # Load the center bias template; we recompute its scale per image.
    centerbias_template = np.load("/home/hafezgh/seq-jepa-dev/data/centerbias_mit1003.npy")

    train_output_dir = "/scratch/users/hafezgh/imagenet_sal_native_ratio/train"
    val_output_dir = "/scratch/users/hafezgh/imagenet_sal_native_ratio/val"

    # Use a minimal transform to preserve native resolution and aspect ratio.
    transform_native = transforms.ToTensor()

    train_set = ImageNetDatasetWithNames('/scratch/datasets/imagenet2012/', "train", transform=transform_native)
    val_set = ImageNetDatasetWithNames('/scratch/datasets/imagenet2012/', "val", transform=transform_native)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)

    process_imagenet(train_output_dir, val_output_dir)
