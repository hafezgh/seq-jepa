import torch
import numpy as np
import os
import random


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def generate_cropping_grid(input_size, crop_size, centers, device):
    # Create normalized grid
    ys = torch.linspace(-1, 1, crop_size, device=device)
    xs = torch.linspace(-1, 1, crop_size, device=device)
    grid_x, grid_y = torch.meshgrid(xs, ys)

    # Convert centers to normalized coordinates
    centers = torch.tensor(centers, device=device).float()
    centers = (centers * 2) / torch.tensor([input_size, input_size], device=device) - 1

    # Adjust grid using broadcasting
    grid_x = grid_x.unsqueeze(0) * (crop_size / input_size) + centers[:, 0].unsqueeze(-1).unsqueeze(-1)
    grid_y = grid_y.unsqueeze(0) * (crop_size / input_size) + centers[:, 1].unsqueeze(-1).unsqueeze(-1)

    grid = torch.stack([grid_x, grid_y], -1)
    return grid


def save_checkpoint(args, epoch, model, min_loss, optimizer, logger, distributed, tag=''):
    if distributed:
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
        
    save_state = {'model': model_state,
                'optimizer': optimizer.state_dict(),
                'min_loss': min_loss,
                'epoch': epoch,
                'args': args}

    save_path = os.path.join(args.output, f'ckpt_epoch_{epoch}_{tag}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved!")


def seed_worker(worker_id):
    np.random.seed(worker_id)
    random.seed(worker_id)
