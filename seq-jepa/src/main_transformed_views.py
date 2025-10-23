import numpy as np
import torch
import torch.nn as nn
import datasets as ds
import models
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
import torch.optim as optim
import os
import random
from engine import *
from augmentations import *
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
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
    parser.add_argument('--seed', type=int, default=42)
    #### Dataset
    parser.add_argument('--dataset', type=str, default='3diebench')
    parser.add_argument('--n-channels', type=int, default=3)
    parser.add_argument('--data-root', type=str, default='DEFAULT')
    parser.add_argument('--latent-type',  type=str, default="rot")
    #### Optimizer
    parser.add_argument('--optimizer', type=str, default='AdamW')
    parser.add_argument('--lr', type=float, default=0.0004)
    parser.add_argument('--scheduler', action='store_true', default=False)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--weight-decay', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--num-workers', type=int, default=8)
  
    #### Action conditioning
    parser.add_argument('--act-cond', type=int, default=1)
    parser.add_argument('--learn-act-emb', type=int, default=1) #### To learn an action embeddings or not
    parser.add_argument('--act-latentdim', type=int, default=4)
    parser.add_argument('--act-projdim', type=int, default=128)
    #### EMA 
    parser.add_argument('--ema', action='store_true', default=False)
    parser.add_argument('--ema-decay', type=float, default=0.996)
    #### Model architecture
    parser.add_argument('--pred-hidden', type=int, default=1024)    #### hidden dimension for the predictor
    parser.add_argument("--cifar-resnet", action='store_true', default=False) #### To use the CIFAR variant of the ResNet backbone
    parser.add_argument("--no-blur", action='store_true', default=False)  #### To use blur augmentation or not
    parser.add_argument("--model", type=str, default='seqjepa') #### Model to use
    parser.add_argument('--seq-len', type=int, default=3) #### Number of views in the sequence
    
    ## Eval
    parser.add_argument('--is-eval', action='store_true', default=False)
    parser.add_argument('--eval-type', type=str, default='rotcolor')

    ### Misc
    parser.add_argument('--output-folder', type=str, default='def', help='path to output folder')
    parser.add_argument('--load-path', type=str, default='DEFAULT')
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument('--offline-wandb', action='store_true', default=False)
    parser.add_argument('--save-freq', type=int, default=100)
    parser.add_argument('--gpu-id', type=int, default=0)    

    args = parser.parse_args()
    return args

    
def main(args):
    ### Setup
    global_rank = 0
    gpu = 0
    if args.gpu_id >= 0:
        gpu = args.gpu_id
    device = torch.device("cuda", gpu)
    torch.cuda.set_device(gpu)
    torch.cuda.empty_cache()
    print("Single GPU training On GPU:", torch.cuda.current_device())

    seed = args.seed
    seeding(seed)

    run_id = f"seq-jepa-transforms_{args.dataset}"

    load_path = args.load_path if args.load_path != 'DEFAULT' else None
    load_dict = torch.load(load_path,map_location=f"cuda:{gpu}") if load_path is not None else None
    output_folder = os.path.join(args.output_folder, run_id)

    if not os.path.exists(output_folder) and global_rank == 0:
        os.makedirs(output_folder, exist_ok=True)
 
    ### Load datasets
    if args.dataset == '3diebench':
        aug = {"mean":[0.5016, 0.5037, 0.5060], "std":[0.1030, 0.0999, 0.0969]}
        trans = transforms.Compose([transforms.Resize((args.img_size, args.img_size)), transforms.ToTensor(), transforms.Normalize(mean=aug["mean"], std=aug["std"]),])
        
        data_root = args.data_root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)  # Go up one level from src to project root
        idx_file_root = os.path.join(project_root, 'data', '3DIEBench')
        img_file_train = os.path.join(idx_file_root, 'train_images.npy')
        labels_file_train = os.path.join(idx_file_root, 'train_labels.npy')
        img_file_val = os.path.join(idx_file_root, 'val_images.npy')
        labels_file_val = os.path.join(idx_file_root, 'val_labels.npy')

        if args.is_eval:
            dataset_latent_type = "rotcolor"
        else:
            dataset_latent_type = args.latent_type
        train_dataset = ds.Dataset_3DIEBench_MultipleViews(data_root, img_file_train, labels_file_train, experience="quat", size_dataset=-1, transform=trans,
                                                num_views=args.seq_len+1, latent_type=dataset_latent_type, is_eval=args.is_eval)
        val_dataset = ds.Dataset_3DIEBench_MultipleViews(data_root, img_file_val, labels_file_val, experience="quat", size_dataset=-1, transform=trans,
                                            num_views=args.seq_len+1, latent_type=dataset_latent_type, is_eval=args.is_eval)
        num_classes = 55
    elif args.dataset == 'cifar100':
        dataset_root = args.data_root
        train_dataset = ds.CIFAR100_aug(dataset_root, "train", args.seq_len+1, aug=True, no_blur=args.no_blur)
        val_dataset = ds.CIFAR100_aug(dataset_root, "test", args.seq_len+1, aug=True, no_blur=args.no_blur)
        num_classes = 100
    elif args.dataset == 'tinyimagenet':
        dataset_root = args.data_root
        train_dataset = ds.TinyImageNet_aug(dataset_root, "train", args.seq_len+1, aug=True)
        val_dataset = ds.TinyImageNet_aug(dataset_root, "val", args.seq_len+1, aug=True)
        num_classes = 200
    else:
        raise ValueError("Dataset should be one of: 3diebench, cifar100, tinyimagenet.")

    num_workers = args.num_workers
    batch_size = args.batch_size
    
    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True,  worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True, worker_init_fn=seed_worker, generator=g)

    ### Model and optimizer
    possible_latent_types = {"rot": 4, "rotcolor": 6, "all": 8, "aug": train_dataset.num_params, "crop": 4, "blur": 1, "colorjitter": 4}
    act_latentdim = possible_latent_types[args.latent_type]
    
    kwargs = {"num_heads": args.num_heads, "n_channels": args.n_channels, "ema_decay": args.ema_decay,
              "num_enc_layers": args.num_enc_layers,"num_classes": num_classes, "act_cond": args.act_cond, "pred_hidden": args.pred_hidden,
              "backbone": args.backbone, "act_projdim": args.act_projdim, "act_latentdim": act_latentdim, "cifar_resnet": args.cifar_resnet,
              "learn_act_emb": args.learn_act_emb}
    
    model = models.SeqJEPA_Transforms(args.img_size, args.ema, **kwargs)    

    if args.is_eval == False:
        model.online_linprobe = nn.Sequential(nn.Linear(emb_dim, num_classes))
        model.online_equiprobe = nn.Sequential(
                    nn.Linear(model.res_out_dim*2,1024),
                    nn.ReLU(),
                    nn.Linear(1024,1024),
                    nn.ReLU(),
                    nn.Linear(1024, act_latentdim),)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
    else:
        model.add_probes(args.latent_type)
        model = model.to(device)
        param_weights = []
        param_biases = []
        for name, param in model.named_parameters():
            if param.ndim == 1: # bias and precision
                param_biases.append(param)
            else: 
                param_weights.append(param)
        parameters = [{'params': param_weights, 'weight_decay': args.weight_decay}, 
                    {'params': param_biases, 'weight_decay': 0.0}]
        optimizer = optim.AdamW(parameters, lr=args.lr)
        learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of learnable parameters: {learnable_params}")

    if load_path is not None:
        model.load_state_dict(load_dict['model'])
        if args.is_eval == False:
            optimizer.load_state_dict(load_dict['optimizer'])
            print("Optimizer loaded!")
        print("Model loaded!")
        
    if args.scheduler and args.is_eval == False:
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
        print("Scheduler added!")
        if load_path is not None and "lr_scheduler" in load_dict.keys():
            lr_scheduler.load_state_dict(load_dict['lr_scheduler'])
            print("Scheduler loaded!")

    ### Wandb logger
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
            
            
    ### Training and eval
    min_loss = 1e9
    if load_path is not None and args.is_eval == False:
        ep_tr = int(load_dict['epoch'])
        min_loss = load_dict['min_loss']
        print("Resuming training from epoch:", ep_tr) 
    else:
        ep_tr = 0
    
    epochs = args.epochs
    if args.is_eval == False:
        print("Training...")
        for epoch in range(ep_tr, epochs):
            if args.warmup > 0 and args.is_eval == False:
                if epoch < args.warmup:
                    initial_lr = 1e-5
                    lr = initial_lr + (args.lr - initial_lr) * (epoch / args.warmup)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                else:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = args.lr

            if args.dataset == '3diebench':
                result = train_one_epoch_3diebench(model, train_loader, optimizer,
                                                    device, val_loader,
                                                    args.ema, args.ema_decay, epoch, epochs)
            else:
                result = train_one_epoch_aug(model, optimizer, device,
                                                train_loader, val_loader, args.ema, args.ema_decay, epoch, epochs, args.latent_type, args.eval_type)
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
                    log_data = {"ep_loss": epoch_loss, "ep_time": ep_time,}
                    log_data["online_linacc_test"] = online_linacc
                    log_data["online_r2_test"] = online_r2
                    log_data["online_r2_train"] = result["online_r2_train"]
                    log_data["online_linacc_train"] = result["online_linacc_train"]
                    log_data["online_r2_loss_test"] = result["online_r2_loss_test"]
                    log_data["online_r2_loss_train"] = result["online_r2_loss_train"]
                    log_data["online_linloss_train"] = result["online_linloss_train"]
                    log_data["online_linloss_test"] = result["online_linloss_test"]

                    wandb_logger.log(log_data, step=epoch)
                if (epoch+1) % args.save_freq == 0:
                    model_state = model.state_dict()
                    save_state = {'model': model_state, 'optimizer': optimizer.state_dict(), 'min_loss': min_loss, 'epoch': epoch+1, 'run_id': run_id}
                            
                    if args.scheduler:
                        save_state['lr_scheduler'] = lr_scheduler.state_dict()
                    save_path = os.path.join(output_folder, f'ckpt_wandb-{id_}_epoch_{epoch+1}.pth')
                    torch.save(save_state, save_path)
    else:
        for epoch in range(ep_tr, epochs):
            print("Evaluating... training eval head...")
            if args.dataset == "3diebench":
                results = val_all_one_epoch_3diebench(
                    model, optimizer, device, train_loader, val_loader, args.img_size,
                    args.latent_type
                )
            else:
                results = val_all_one_epoch_aug(
                    model, optimizer, device, train_loader, val_loader, args.img_size,
                    optimizer, args.latent_type
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

