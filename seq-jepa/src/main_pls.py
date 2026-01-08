import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist

import datasets as ds
from models import *

from torch.optim.lr_scheduler import CosineAnnealingLR

import argparse
import os
import torch.optim as optim
import random

from engine import *

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import wandb

from utils import seed_worker


def seeding(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    return seed

def parse_option():
    parser = argparse.ArgumentParser('Arguments for training')
    parser.add_argument('--method', type=str, default='seqjepa', choices=['seqjepa', 'conv-ijepa'])
    parser.add_argument('--shuffle-saccades', action='store_true', default=False)
    parser.add_argument('--aug-patches', action='store_true', default=False)
    parser.add_argument('--distributed', action='store_true', default=False)
    #### Dataset
    parser.add_argument('--img-size', type=int, default=96)
    parser.add_argument('--data-path-img', type=str, default='DEFAULT')
    parser.add_argument('--data-path-sal', type=str, default='DEFAULT')
    parser.add_argument('--dataset', type=str, default='stl10')
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--pos-dim', type=int, default=128)
    parser.add_argument('--ior', type=int, default=1) ### 0 for no ior, 1 for ior
    parser.add_argument('--use-sal', type=int, default=1) ### 0 for no sal, 1 for sal
    parser.add_argument("--cifar-resnet", action='store_true', default=False) ### for STL-10 patches (32x32)
    parser.add_argument('--num-classes', type=int, default=10) ### stl10: 10, imagenet: 1000
    parser.add_argument('--n-channels', type=int, default=3)
    parser.add_argument('--num-saccades', type=int, default=5)
    parser.add_argument('--fovea-size', type=int, default=32) ### stl10: 32, imagenet: 64 or 84
    
    #### Optimizer
    parser.add_argument('--lr', type=float, default=0.0004)
    parser.add_argument('--scheduler', action='store_true', default=False)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--initial-lr', type=float, default=1e-5)
    parser.add_argument('--weight-decay', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--optimizer', type=str, default='AdamW')
    
    #### EMA 
    
    parser.add_argument('--ema', action='store_true', default=False)
    parser.add_argument('--ema-decay', type=float, default=0.996)
    
    #### Model architecture
    parser.add_argument('--pred-hidden', type=int, default=1024)  ### hidden dimension for the predictor
    parser.add_argument('--num-heads', type=int, default=4)  ### number of heads for the transformer
    parser.add_argument('--num-enc-layers', type=int, default=3)  ### number of layers for the transformer
    parser.add_argument('--act-cond', type=int, default=1) ### 0 for no action conditioning, 1 for action conditioning
    parser.add_argument('--learn-act-emb', type=int, default=1) #### To learn an action embeddings or not
    parser.add_argument('--act-latentdim', type=int, default=2)
    parser.add_argument('--act-projdim', type=int, default=128)
   
    ### Miscellaneous
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument('--offline-wandb', action='store_true', default=False)
    parser.add_argument('--run-id', type=str, default='use_default')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-folder', type=str, default='DEFAULT')
    parser.add_argument('--is-eval', action='store_true', default=False)
    parser.add_argument('--load-path', type=str, default='DEFAULT')
    parser.add_argument('--save-freq', type=int, default=100) ### save ckpt every n epochs

    args = parser.parse_args()
    return args

    
def main(args):
    ### setup
    if args.distributed:
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            global_rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
        elif 'SLURM_PROCID' in os.environ and 'SLURM_NTASKS' in os.environ:
            global_rank = int(os.environ['SLURM_PROCID'])
            world_size = int(os.environ['SLURM_NTASKS'])
            os.environ.setdefault('MASTER_ADDR', os.environ.get('SLURM_NODELIST', '127.0.0.1').split(',')[0])
            os.environ.setdefault('MASTER_PORT', '12345')
            os.environ.setdefault('RANK', str(global_rank))
            os.environ.setdefault('WORLD_SIZE', str(world_size))
        else:
            raise RuntimeError("Distributed execution requested but RANK/WORLD_SIZE not configured")
        dist.init_process_group(backend='nccl', init_method='env://')
        gpu = global_rank % torch.cuda.device_count()
        torch.cuda.set_device(gpu)
        device = torch.device("cuda", gpu)
        args.world_size = world_size
        args.effective_batch_size = args.batch_size * args.world_size
        print(f"Distributed training on {args.world_size} GPUs with effective batch size {args.effective_batch_size}")
        print("cuda.device_count:", torch.cuda.device_count(), "rank:", global_rank, "device", device)
    else:
        global_rank = 0
        gpu = 0
        torch.cuda.set_device(gpu)
        device = torch.device("cuda", gpu)
        torch.cuda.empty_cache()
        args.world_size = 1
        args.effective_batch_size = args.batch_size
        print("Single GPU training On GPU:", torch.cuda.current_device())
        
    seed = args.seed
    seeding(seed)

    run_id = f"seq-jepa-pls-{args.dataset}"

    ### load ckpt if exists
    load_path = args.load_path if args.load_path != 'DEFAULT' else None
    load_dict = torch.load(load_path,map_location=f"cuda:{gpu}") if load_path is not None else None
    
    output_folder = os.path.join(args.output_folder, run_id)
            
    if not os.path.exists(output_folder) and global_rank == 0:
        os.makedirs(output_folder, exist_ok=True)
        
    ### load datasets
    if args.dataset == 'stl10':
        unlabeled_dataset = ds.STL10_SalMap(args.data_path_img, args.data_path_sal, 'unlabeled', args.num_saccades,
                                            args.use_sal, args.ior)
        train_dataset = ds.STL10_SalMap(args.data_path_img, args.data_path_sal, 'train', args.num_saccades,
                                            args.use_sal, args.ior)
        test_dataset = ds.STL10_SalMap(args.data_path_img, args.data_path_sal, 'test', args.num_saccades,
                                            args.use_sal, args.ior)
    elif args.dataset == 'imagenet':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),  # Random crop with scaling and aspect ratio variation.
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
        transform_val = transforms.Compose([
            transforms.Resize(256),           # Resize the shorter side to 256 pixels.
            transforms.CenterCrop(224),       # Then center crop to 224x224.
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])

        train_dataset = ds.Imagenet1k_Sal(args.data_path_img, args.data_path_sal, "train",
                                        full_img_transform=transform_train, full_img_size=args.img_size, 
                                        patch_size=args.fovea_size, num_patches=args.num_saccades)
        test_dataset = ds.Imagenet1k_Sal(args.data_path_img, args.data_path_sal, "val",
                                        full_img_transform=transform_val, full_img_size=args.img_size, 
                                        patch_size=args.fovea_size, num_patches=args.num_saccades)
        unlabeled_dataset = train_dataset
    else:
        raise ValueError("Dataset should be one of: stl10, imagenet.")

    
    g = torch.Generator()
    g.manual_seed(args.seed)

    if args.distributed:
        unlabeled_sampler = DistributedSampler(unlabeled_dataset, shuffle=True)
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
        unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=args.batch_size, num_workers=args.num_workers, sampler=unlabeled_sampler, worker_init_fn=seed_worker, generator=g)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, sampler=train_sampler, worker_init_fn=seed_worker, generator=g)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, sampler=test_sampler, worker_init_fn=seed_worker, generator=g)
    else:
        unlabeled_sampler = None
        unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, worker_init_fn=seed_worker, generator=g)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, worker_init_fn=seed_worker, generator=g)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, worker_init_fn=seed_worker, generator=g)
    
    
    ### model and optimizer
    kwargs = {"num_heads": args.num_heads, "n_channels": args.n_channels, "act_cond": args.act_cond, "pred_hidden": args.pred_hidden,
              "num_enc_layers": args.num_enc_layers, "num_classes": args.num_classes, "pos_dim": args.pos_dim, "backbone": args.backbone, "ema_decay": args.ema_decay,
              "act_projdim": args.act_projdim, "act_latentdim": args.act_latentdim, "learn_act_emb": args.learn_act_emb, "cifar_resnet": args.cifar_resnet}
    
    
    methods = {"seqjepa": SeqJEPA_PLS, "conv-ijepa": Conv_IJEPA}
    model = methods[args.method](args.fovea_size, args.img_size, args.ema, **kwargs) 
    
    if load_path is not None:
        model.load_state_dict(load_dict['model'])
        print("Model loaded!")
        
    if args.is_eval:
        model.add_probes()
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
    else:
        model.online_res_probe = nn.Sequential(nn.Linear(model.res_out_dim, args.num_classes)).to(device)
        if args.method == 'seqjepa':
            model.online_agg_probe = nn.Sequential(nn.Linear(model.emb_dim, args.num_classes)).to(device)
        model.online_equiprobe = nn.Sequential(nn.Linear(model.res_out_dim*2, args.act_latentdim)).to(device)
        model = model.to(device)
        param_weights = []
        param_biases = []
        for name, param in model.named_parameters():
            if param.ndim == 1: # bias and precision
                param_biases.append(param)
            else: 
                param_weights.append(param)
        parameters = [{'params': param_weights, 'weight_decay': args.weight_decay}, 
                    {'params': param_biases, 'weight_decay': args.weight_decay}]
        if args.optimizer == 'AdamW':
            parameters[1]['weight_decay'] = 0.0
            optimizer = optim.AdamW(parameters, lr=args.lr)
        elif args.optimizer == 'Adam':
            optimizer = optim.Adam(parameters, lr=args.lr)
        elif args.optimizer == 'SGD':
            optimizer = optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        else:
            raise ValueError(f"Optimizer {args.optimizer} not supported.")
        learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of learnable parameters: {learnable_params}")
    
    if args.scheduler and args.is_eval == False:
        eta_min = args.initial_lr
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=eta_min)
        print("Scheduler added!")
    if load_path is not None and args.is_eval == False:
        optimizer.load_state_dict(load_dict['optimizer'])
        print("Optimizer loaded!")
        
    ### wandb logger
    config_dict = vars(args)
    print(config_dict)
    id_ = "no-id"
    
    if global_rank == 0:
        if args.wandb == True:
            if args.offline_wandb:
                os.environ['WANDB_MODE'] = 'offline'
                os.environ["WANDB__SERVICE_WAIT"] = "300"
            else:
                os.environ["WANDB__SERVICE_WAIT"] = "300"
                wandb.login()
            id_ = wandb.util.generate_id()
            run_id = f"wandbid-{id_}_" + run_id
            wandb_logger = wandb.init(name=run_id, id=run_id, config=config_dict)
            print("Wandb initialized!")
 
    ### training and eval
    if load_path is not None and args.is_eval == False:
        ep_tr = int(load_dict['epoch'])
        min_loss = load_dict['min_loss']
        print("Resuming training from epoch:", ep_tr)
    else:
        ep_tr = 0
        min_loss = 1e9
    epochs = args.epochs
    if args.is_eval == False:
        print("Training...")
        for epoch in range(ep_tr, epochs):
            if args.distributed:
                unlabeled_sampler.set_epoch(epoch)
            if args.warmup > 0 and args.is_eval == False:
                if epoch < args.warmup:
                    initial_lr = args.initial_lr
                    lr = initial_lr + (args.lr - initial_lr) * (epoch / args.warmup)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                else:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = args.lr
            result = train_one_epoch_pls(model, unlabeled_loader, optimizer, args.num_saccades, args.img_size,
                                            args.fovea_size, device, args.ema, args.ema_decay, epoch, epochs, conv_jepa=args.method == 'conv-ijepa',
                                            train_loader=train_loader, test_loader=test_loader, dataset=args.dataset)
            if args.scheduler:
                if args.warmup and epoch < args.warmup:
                    pass
                else:
                    lr_scheduler.step()

            epoch_loss = result["ep_loss"]
            ep_time = result["ep_time"]
            min_loss = min(min_loss, epoch_loss)
            
            if global_rank == 0:
                online_linacc = result["online_linacc_test"]
                online_r2 = result["online_r2_test"]
                
                print("Epoch {}/{}, Loss: {:.6f}, min_loss: {:.6f}, ep_time:{:.2f}, online_linacc: {:.4f}, online_r2: {:.4f}".format(epoch+1, args.epochs, epoch_loss, min_loss, ep_time, online_linacc, online_r2))
                if args.wandb == True:
                    log_data = {"ep_loss": epoch_loss, "ep_time": ep_time}
                    log_data["online_linacc_test"] = online_linacc
                    log_data["online_r2_test"] = online_r2
                    log_data["online_r2_train"] = result["online_r2_train"]
                    log_data["online_linacc_train"] = result["online_linacc_train"]
                    log_data["online_r2_loss_test"] = result["online_r2_loss_test"]
                    log_data["online_r2_loss_train"] = result["online_r2_loss_train"]
                    log_data["online_linloss_train"] = result["online_linloss_train"]
                    log_data["online_linloss_test"] = result["online_linloss_test"]

                    log_data["online_linacc"] = online_linacc
                    log_data["online_r2"] = online_r2
                    wandb_logger.log(log_data, step=epoch)
                    print("Wandb logged!")
                if (epoch+1) % args.save_freq == 0:
                    exclude_keys = ['online_res_probe', 'online_agg_probe', 'online_equiprobe']
                    model_state = {k: v for k, v in model.state_dict().items() if k not in exclude_keys}
                    save_state = {'model': model_state, 'optimizer': optimizer.state_dict(), 'min_loss': min_loss, 'epoch': epoch+1, 'run_id': run_id}
                    if args.scheduler:
                        save_state['lr_scheduler'] = lr_scheduler.state_dict()
                    save_path = os.path.join(output_folder, f'ckpt_wandb-{id_}_epoch_{epoch+1}.pth')
                    torch.save(save_state, save_path)
    else:
        for epoch in range(ep_tr, epochs):
            print("Evaluating... training eval head...")
            results = val_all_one_epoch_pls(
                model, device, train_loader, test_loader, args.img_size, args.fovea_size,
                optimizer
            )

            if global_rank == 0:
                formatted_results = ", ".join(
                    f"{key}: {value:.6f}" if isinstance(value, (int, float)) else f"{key}: {value}"
                    for key, value in results.items()
                )
                print(f"Epoch {epoch+1}/{args.epochs}, {formatted_results}")
                if args.wandb:
                    log_data = {}
                    for key, value in results.items():
                        if isinstance(value, list):
                            # Log each element separately in Wandb
                            for j, v in enumerate(value):
                                log_data[f"{key}_{j}"] = v
                        else:
                            log_data[key] = value
                    
                    wandb_logger.log(log_data, step=epoch)
    print("Done!")



if __name__ == '__main__':
    args = parse_option()
    main(args)