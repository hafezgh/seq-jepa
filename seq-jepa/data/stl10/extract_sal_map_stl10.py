import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from scipy.special import logsumexp
from scipy.ndimage import zoom
import torch
import deepgaze_pytorch
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Extract saliency maps from STL10 dataset')
    parser.add_argument('--data_root', type=str, default='DEFAULT',
                        help='Root directory of STL10 dataset')
    parser.add_argument('--output_root', type=str, default='salmap-stl10',
                        help='Root directory for output saliency maps')
    parser.add_argument('--centerbias_path', type=str, default='centerbias_mit1003.npy',
                        help='Path to centerbias template')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for processing')
    return parser.parse_args()

def process_images(unlabeled_output_dir, train_output_dir, test_output_dir):
    with torch.no_grad():
        saliency_maps_unlabeled = []
        for batch_idx, ((images_512, images_96), labels) in enumerate(unlabeled_loader):
            images_512 = images_512.to('cuda')

            # Extract saliency maps
            log_density_prediction = model(images_512, centerbias_tensor)

            # Resize saliency maps to 96x96
            saliency_maps_96 = torch.nn.functional.interpolate(
                log_density_prediction, size=(96, 96), mode='bilinear', align_corners=False
            )

            # Convert log densities to probabilities
            prob_maps = torch.exp(saliency_maps_96)
            prob_maps = prob_maps / prob_maps.sum(dim=(-2, -1), keepdim=True)

            # Move to CPU and convert to numpy
            prob_maps_np = prob_maps.cpu().numpy()
            saliency_maps_unlabeled.append(prob_maps_np)

            print(f"Unlabeled Set Batch: {batch_idx+1}/{len(unlabeled_loader)}")
            
        # Stack all saliency maps
        saliency_maps_unlabeled = np.concatenate(saliency_maps_unlabeled, axis=0)
        # Save saliency maps
        np.save(os.path.join(unlabeled_output_dir, 'saliency_unlabeled.npy'), saliency_maps_unlabeled)
        print(f"Processed and saved {len(unlabeled_set)} unlabeled saliency maps to {unlabeled_output_dir}")

    with torch.no_grad():
        saliency_maps_train = []
        for batch_idx, ((images_512, images_96), labels) in enumerate(train_loader):
            images_512 = images_512.to('cuda')

            # Extract saliency maps
            log_density_prediction = model(images_512, centerbias_tensor)

            # Resize saliency maps to 96x96
            saliency_maps_96 = torch.nn.functional.interpolate(
                log_density_prediction, size=(96, 96), mode='bilinear', align_corners=False
            )

            # Convert log densities to probabilities
            prob_maps = torch.exp(saliency_maps_96)
            prob_maps = prob_maps / prob_maps.sum(dim=(-2, -1), keepdim=True)

            # Move to CPU and convert to numpy
            prob_maps_np = prob_maps.cpu().numpy()
            saliency_maps_train.append(prob_maps_np)

            print(f"Train Set Batch: {batch_idx+1}/{len(train_loader)}")
            
        # Stack all saliency maps
        saliency_maps_train = np.concatenate(saliency_maps_train, axis=0)
        # Save saliency maps
        np.save(os.path.join(train_output_dir, 'saliency_train.npy'), saliency_maps_train)
        print(f"Processed and saved {len(train_loader.dataset)} train saliency maps to {train_output_dir}")

    with torch.no_grad():
        saliency_maps_test = []
        for batch_idx, ((images_512, images_96), labels) in enumerate(test_loader):
            images_512 = images_512.to('cuda')

            # Extract saliency maps
            log_density_prediction = model(images_512, centerbias_tensor)

            # Resize saliency maps to 96x96
            saliency_maps_96 = torch.nn.functional.interpolate(
                log_density_prediction, size=(96, 96), mode='bilinear', align_corners=False
            )

            # Convert log densities to probabilities
            prob_maps = torch.exp(saliency_maps_96)
            prob_maps = prob_maps / prob_maps.sum(dim=(-2, -1), keepdim=True)

            # Move to CPU and convert to numpy
            prob_maps_np = prob_maps.cpu().numpy()
            saliency_maps_test.append(prob_maps_np)

            print(f"Test Set Batch: {batch_idx+1}/{len(test_loader)}")

        # Stack all saliency maps 
        saliency_maps_test = np.concatenate(saliency_maps_test, axis=0)
        # Save saliency maps 
        np.save(os.path.join(test_output_dir, 'saliency_test.npy'), saliency_maps_test)
        print(f"Processed and saved {len(test_loader.dataset)} test saliency maps to {test_output_dir}")

if __name__ == '__main__':
    args = parse_args()
    DEVICE = 'cuda'
    batch_size = args.batch_size

    model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(DEVICE)
    model.eval()

    # Handle centerbias path relative to script location
    if args.centerbias_path == 'centerbias_mit1003.npy':
        script_dir = os.path.dirname(os.path.abspath(__file__))
        centerbias_path = os.path.join(script_dir, 'centerbias_mit1003.npy')
    else:
        centerbias_path = args.centerbias_path
   
    # Load and process centerbias template
    centerbias_template = np.load(centerbias_path)
    # rescale to match image size
    centerbias = zoom(centerbias_template, (512/centerbias_template.shape[0], 512/centerbias_template.shape[1]), order=0, mode='nearest')
    # renormalize log density
    centerbias -= logsumexp(centerbias)
    centerbias_tensor = torch.tensor(centerbias).unsqueeze(0).repeat(batch_size, 1, 1).to(DEVICE)
    
    # Create output directories
    unlabeled_output_dir = os.path.join(args.output_root, "unlabeled")
    train_output_dir = os.path.join(args.output_root, "train")
    test_output_dir = os.path.join(args.output_root, "test")
    
    os.makedirs(unlabeled_output_dir, exist_ok=True)
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)

    transform_512 = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
        ])
    transform_96 = transforms.Compose([
        transforms.Resize(96),
        transforms.CenterCrop(96),
        transforms.ToTensor(),
    ])

    unlabeled_set = torchvision.datasets.STL10(root=args.data_root, split="unlabeled", transform=lambda x: (transform_512(x), transform_96(x)))
    train_set = torchvision.datasets.STL10(root=args.data_root, split="train", transform=lambda x: (transform_512(x), transform_96(x)))
    test_set = torchvision.datasets.STL10(root=args.data_root, split="test", transform=lambda x: (transform_512(x), transform_96(x)))
    unlabeled_loader = DataLoader(unlabeled_set, batch_size=batch_size, shuffle=False, num_workers=2)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    process_images(unlabeled_output_dir, train_output_dir, test_output_dir)

