import torchvision.transforms as transforms
import torch
import time

import math

from scipy.spatial.transform import Rotation as R
from utils import *
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score


############
def train_one_epoch_aug(model, optimizer, device, train_loader, val_loader,
                        ema, ema_tau_base, current_epoch, num_epochs, latent_type):
    """
    Train one epoch for augmentation datasets (CIFAR100, TinyImageNet).
    Uses latent_type for both conditioning and R2 eval.
    """
    def get_latent_slice(params, lt):
        if lt == "crop":
            return params[:, :, :4]
        elif lt == "blur":
            return params[:, :, 8].unsqueeze(2)
        elif lt == "colorjitter":
            return params[:, :, 4:8]
        return params

    # Detect model type for proper input formatting
    is_seq_model = hasattr(model, 'transformer_encoder')
    is_traj_model = hasattr(model, 'alpha')  # VICReg_Traj has alpha param

    mse = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    model.train()
    prev_time = time.time()
    epoch_loss = 0.
    tot_samples = 0.
    train_targets, train_preds = [], []
    train_equi_preds, train_equi_targets = [], []
    running_lin_loss_train, running_equi_loss_train = 0., 0.

    for augmented_images, augmented_params, label, orig_img in tqdm(train_loader, dynamic_ncols=True):
        batch = augmented_images.to(device)
        labels = label.to(device)
        aug_params = get_latent_slice(augmented_params, latent_type).to(device)
        aug_params = torch.nn.functional.normalize(aug_params, p=2, dim=1)
        latent_size = aug_params.shape[-1]
        rel_latents = (aug_params[:, 1] - aug_params[:, 0]).reshape(-1, latent_size)

        optimizer.zero_grad()
        if is_seq_model:
            model_out = model(batch[:, :-1], batch[:, -1], aug_params[:, :-1], aug_params[:, -1])
        elif is_traj_model:
            model_out = model(batch[:, :3])
        else:
            model_out = model(batch[:, 0], batch[:, 1], rel_latents)
        
        if len(model_out) == 4:
            loss, agg_out, z1, z2 = model_out
        else:
            loss, z1, z2 = model_out
            agg_out = z1  # Use z1 for probing when no aggregator

        # Online linear probe (detached)
        online_linout = model.online_linprobe(agg_out.detach())
        online_linloss = criterion(online_linout, labels)
        loss = loss + online_linloss
        running_lin_loss_train += online_linloss.item()
        train_preds.extend(torch.argmax(online_linout, dim=1).cpu().numpy())
        train_targets.extend(labels.cpu().numpy())

        # Online equivariance probe (detached)
        embs_concat = torch.cat((z1.detach(), z2.detach()), dim=1)
        pred_rel = model.online_equiprobe(embs_concat)
        online_equiloss = mse(pred_rel, rel_latents)
        loss = loss + online_equiloss
        running_equi_loss_train += online_equiloss.item()
        train_equi_preds.append(pred_rel.detach().cpu().numpy())
        train_equi_targets.append(rel_latents.cpu().numpy())

        loss.backward()
        optimizer.step()

        tot_samples += labels.size(0)
        epoch_loss += loss.item()

        if ema and hasattr(model, 'update_moving_average'):
            model.update_moving_average()
            model.ema_decay = 1 - (1 - ema_tau_base) * ((math.cos(math.pi * current_epoch / num_epochs) + 1) / 2)

    train_acc = accuracy_score(train_targets, train_preds)
    online_r2_train = r2_score(np.concatenate(train_equi_targets), np.concatenate(train_equi_preds))

    # Validation
    model.eval()
    test_targets, test_preds = [], []
    test_equi_preds, test_equi_targets = [], []
    running_lin_loss_test, running_equi_loss_test = 0., 0.

    with torch.no_grad():
        for augmented_images, augmented_params, label, orig_img in tqdm(val_loader, desc="Val", dynamic_ncols=True):
            batch = augmented_images.to(device)
            labels = label.to(device)
            aug_params = get_latent_slice(augmented_params, latent_type).to(device)
            aug_params = torch.nn.functional.normalize(aug_params, p=2, dim=1)
            latent_size = aug_params.shape[-1]
            rel_latents = (aug_params[:, 1] - aug_params[:, 0]).reshape(-1, latent_size)

            if is_seq_model:
                model_out = model(batch[:, :-1], batch[:, -1], aug_params[:, :-1], aug_params[:, -1])
            elif is_traj_model:
                model_out = model(batch[:, :3])
            else:
                model_out = model(batch[:, 0], batch[:, 1], rel_latents)
            
            if len(model_out) == 4:
                _, agg_out, z1, z2 = model_out
            else:
                _, z1, z2 = model_out
                agg_out = z1

            online_linout = model.online_linprobe(agg_out)
            running_lin_loss_test += criterion(online_linout, labels).item()
            test_preds.extend(torch.argmax(online_linout, dim=1).cpu().numpy())
            test_targets.extend(labels.cpu().numpy())

            embs_concat = torch.cat((z1, z2), dim=1)
            pred_rel = model.online_equiprobe(embs_concat)
            running_equi_loss_test += mse(pred_rel, rel_latents).item()
            test_equi_preds.append(pred_rel.cpu().numpy())
            test_equi_targets.append(rel_latents.cpu().numpy())

    test_acc = accuracy_score(test_targets, test_preds)
    online_r2_test = r2_score(np.concatenate(test_equi_targets), np.concatenate(test_equi_preds))

    result = {
        "online_linacc_train": train_acc * 100.0,
        "online_linacc_test": test_acc * 100.0,
        "online_r2_train": online_r2_train,
        "online_r2_test": online_r2_test,
        "online_linloss_train": running_lin_loss_train / len(train_loader),
        "online_linloss_test": running_lin_loss_test / len(val_loader),
        "online_r2_loss_train": running_equi_loss_train / len(train_loader),
        "online_r2_loss_test": running_equi_loss_test / len(val_loader),
        "ep_loss": epoch_loss / tot_samples,
        "ep_time": time.time() - prev_time,
    }
    return result


def train_one_epoch_3diebench(model, train_loader, optimizer, device, test_loader,
                              ema, ema_tau_base, current_epoch, num_epochs):
    """Train one epoch for 3DIEBench. R2 eval uses same latent as conditioning (latent_type)."""
    # Detect model type for proper input formatting
    is_seq_model = hasattr(model, 'transformer_encoder')
    is_traj_model = hasattr(model, 'alpha')  # VICReg_Traj has alpha param
    
    mse = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    model.train()
    prev_time = time.time()
    epoch_loss = 0.
    tot_samples = 0.
    train_targets, train_preds = [], []
    train_equi_preds, train_equi_targets = [], []
    running_lin_loss_train, running_equi_loss_train = 0., 0.

    for batch, act_latents, rel_latents, labels in tqdm(train_loader, dynamic_ncols=True):
        batch = batch.to(device)
        labels = labels.to(device)
        act_latents = act_latents.to(device)
        rel_latents = rel_latents.to(device)

        optimizer.zero_grad()
        if is_seq_model:
            model_out = model(batch[:,:-1], batch[:,-1], act_latents[:,:-1], act_latents[:,-1], rel_latents=rel_latents)
        elif is_traj_model:
            # VICReg_Traj expects (B, 3, C, H, W) - 3 sequential views
            model_out = model(batch[:, :3])
        else:
            model_out = model(batch[:, 0], batch[:, 1], rel_latents[:, 0])
        
        if len(model_out) == 4:
            loss, agg_out, emb1, emb2 = model_out
        else:
            loss, emb1, emb2 = model_out
            agg_out = emb1

        # Online linear probe (detached)
        online_linout = model.online_linprobe(agg_out.detach())
        online_linloss = criterion(online_linout, labels)
        loss = loss + online_linloss
        running_lin_loss_train += online_linloss.item()
        train_preds.extend(torch.argmax(online_linout, dim=1).cpu().numpy())
        train_targets.extend(labels.cpu().numpy())

        # Online equivariance probe (detached)
        embs_concat = torch.cat((emb1.detach(), emb2.detach()), dim=1)
        pred_rel = model.online_equiprobe(embs_concat)
        target_rel = rel_latents[:, 0]
        online_equiloss = mse(pred_rel, target_rel)
        loss = loss + online_equiloss
        running_equi_loss_train += online_equiloss.item()
        train_equi_preds.append(pred_rel.detach().cpu().numpy())
        train_equi_targets.append(target_rel.cpu().numpy())

        loss.backward()
        optimizer.step()

        tot_samples += labels.size(0)
        epoch_loss += loss.item()

        if ema and hasattr(model, 'update_moving_average'):
            model.update_moving_average()
            model.ema_decay = 1 - (1 - ema_tau_base) * ((math.cos(math.pi * current_epoch / num_epochs) + 1) / 2)

    train_acc = accuracy_score(train_targets, train_preds)
    online_r2_train = r2_score(np.concatenate(train_equi_targets), np.concatenate(train_equi_preds))

    # Validation
    model.eval()
    test_targets, test_preds = [], []
    test_equi_preds, test_equi_targets = [], []
    running_lin_loss_test, running_equi_loss_test = 0., 0.

    with torch.no_grad():
        for batch, act_latents, rel_latents, labels in tqdm(test_loader, desc="Val", dynamic_ncols=True):
            batch = batch.to(device)
            labels = labels.to(device)
            act_latents = act_latents.to(device)
            rel_latents = rel_latents.to(device)

            if is_seq_model:
                model_out = model(batch[:,:-1], batch[:,-1], act_latents[:,:-1], act_latents[:,-1], rel_latents=rel_latents)
            elif is_traj_model:
                model_out = model(batch[:, :3])
            else:
                model_out = model(batch[:, 0], batch[:, 1], rel_latents[:, 0])
            
            if len(model_out) == 4:
                _, agg_out, emb1, emb2 = model_out
            else:
                _, emb1, emb2 = model_out
                agg_out = emb1

            online_linout = model.online_linprobe(agg_out)
            running_lin_loss_test += criterion(online_linout, labels).item()
            test_preds.extend(torch.argmax(online_linout, dim=1).cpu().numpy())
            test_targets.extend(labels.cpu().numpy())

            embs_concat = torch.cat((emb1, emb2), dim=1)
            pred_rel = model.online_equiprobe(embs_concat)
            target_rel = rel_latents[:, 0]
            running_equi_loss_test += mse(pred_rel, target_rel).item()
            test_equi_preds.append(pred_rel.cpu().numpy())
            test_equi_targets.append(target_rel.cpu().numpy())

    test_acc = accuracy_score(test_targets, test_preds)
    online_r2_test = r2_score(np.concatenate(test_equi_targets), np.concatenate(test_equi_preds))

    result = {
        "online_linacc_train": train_acc * 100.0,
        "online_linacc_test": test_acc * 100.0,
        "online_r2_train": online_r2_train,
        "online_r2_test": online_r2_test,
        "online_linloss_train": running_lin_loss_train / len(train_loader),
        "online_linloss_test": running_lin_loss_test / len(test_loader),
        "online_r2_loss_train": running_equi_loss_train / len(train_loader),
        "online_r2_loss_test": running_equi_loss_test / len(test_loader),
        "ep_loss": epoch_loss / tot_samples,
        "ep_time": time.time() - prev_time,
    }
    return result

def train_one_epoch_pls(model, data_loader, optimizer, num_saccades, image_size, fovea_size,
                        device, ema, ema_tau_base, current_epoch, num_epochs, conv_jepa=False,
                        train_loader=None, test_loader=None, dataset="stl10"):
    prev_time = time.time()
    epoch_loss = 0.
    tot_samples = 0
    CELoss = nn.CrossEntropyLoss()
    MSELoss = nn.MSELoss()
    has_agg_probe = hasattr(model, 'online_agg_probe') and not conv_jepa

    model.train()
    for _, _, labels, patches, sac_pos in tqdm(data_loader, desc="SSL Train", dynamic_ncols=True):
        patches, sac_pos, labels = patches.to(device), sac_pos.to(device), labels.to(device)
        foveated_x_obs, foveated_x_last = patches[:, :-1], patches[:, -1]

        optimizer.zero_grad()
        loss, agg_out, z1, z2 = model(foveated_x_obs, foveated_x_last, sac_pos)

        if dataset == "imagenet":
            z1_d, z2_d = z1.detach(), z2.detach()
            loss = loss + CELoss(model.online_res_probe(z1_d), labels)
            if has_agg_probe:
                loss = loss + CELoss(model.online_agg_probe(agg_out.detach()), labels)
            rel_pos = sac_pos[:, 1] - sac_pos[:, 0]
            loss = loss + MSELoss(model.online_equiprobe(torch.cat((z1_d, z2_d), dim=1)), rel_pos)

        loss.backward()
        optimizer.step()

        tot_samples += patches.size(0)
        epoch_loss += loss.item()

        if ema and hasattr(model, 'update_moving_average'):
            model.update_moving_average()
            model.ema_decay = 1 - (1 - ema_tau_base) * ((math.cos(math.pi * current_epoch / num_epochs) + 1) / 2)

    ### stl10 unlabelled set does not have labels, so we need to train the probes on the labelled train set
    if dataset == "stl10":
        model.eval()  # Freeze BN stats; probes (Linear) unaffected
        train_preds, train_targets = [], []
        train_equi_preds, train_equi_targets = [], []
        running_lin_loss, running_equi_loss = 0., 0.

        for _, _, labels, patches, sac_pos in tqdm(train_loader, desc="Probe Train", dynamic_ncols=True):
            patches, sac_pos, labels = patches.to(device), sac_pos.to(device), labels.to(device)

            with torch.no_grad():
                _, _, z1, z2 = model(patches[:, :-1], patches[:, -1], sac_pos)

            optimizer.zero_grad()
            res_out = model.online_res_probe(z1)
            rel_pos = sac_pos[:, 1] - sac_pos[:, 0]
            equi_out = model.online_equiprobe(torch.cat((z1, z2), dim=1))
            probe_loss = CELoss(res_out, labels) + MSELoss(equi_out, rel_pos)
            probe_loss.backward()
            optimizer.step()

            running_lin_loss += CELoss(res_out, labels).item()
            running_equi_loss += MSELoss(equi_out, rel_pos).item()
            train_preds.extend(torch.argmax(res_out, dim=1).cpu().numpy())
            train_targets.extend(labels.cpu().numpy())
            train_equi_preds.extend(equi_out.detach().cpu().numpy())
            train_equi_targets.extend(rel_pos.cpu().numpy())

        online_linacc_train = accuracy_score(train_targets, train_preds) * 100.0
        online_r2_train = r2_score(train_equi_targets, train_equi_preds)
        online_linloss_train = running_lin_loss / len(train_loader)
        online_r2_loss_train = running_equi_loss / len(train_loader)
    else:
        online_linacc_train, online_r2_train = 0., 0.
        online_linloss_train, online_r2_loss_train = 0., 0.

    ### eval on test set
    model.eval()
    test_preds, test_targets = [], []
    test_equi_preds, test_equi_targets = [], []
    running_lin_loss_test, running_equi_loss_test = 0., 0.

    with torch.no_grad():
        for _, _, labels, patches, sac_pos in tqdm(test_loader, desc="Eval", dynamic_ncols=True):
            patches, sac_pos, labels = patches.to(device), sac_pos.to(device), labels.to(device)
            _, _, z1, z2 = model(patches[:, :-1], patches[:, -1], sac_pos)

            res_out = model.online_res_probe(z1)
            rel_pos = sac_pos[:, 1] - sac_pos[:, 0]
            equi_out = model.online_equiprobe(torch.cat((z1, z2), dim=1))

            running_lin_loss_test += CELoss(res_out, labels).item()
            running_equi_loss_test += MSELoss(equi_out, rel_pos).item()
            test_preds.extend(torch.argmax(res_out, dim=1).cpu().numpy())
            test_targets.extend(labels.cpu().numpy())
            test_equi_preds.extend(equi_out.cpu().numpy())
            test_equi_targets.extend(rel_pos.cpu().numpy())

    result = {
        "ep_loss": epoch_loss / tot_samples,
        "ep_time": time.time() - prev_time,
        "online_linacc_train": online_linacc_train,
        "online_linacc_test": accuracy_score(test_targets, test_preds) * 100.0,
        "online_r2_train": online_r2_train,
        "online_r2_test": r2_score(test_equi_targets, test_equi_preds),
        "online_linloss_train": online_linloss_train,
        "online_linloss_test": running_lin_loss_test / len(test_loader),
        "online_r2_loss_train": online_r2_loss_train,
        "online_r2_loss_test": running_equi_loss_test / len(test_loader),
    }
    return result
    
    

################################ EVALUATION FUNCTIONS ################################
def val_all_one_epoch_3diebench(model, optimizer, device, train_loader, val_loader, img_size, latent_type):
    mse = nn.MSELoss()
    cross_entropy = nn.CrossEntropyLoss()
    has_rot = hasattr(model, 'rot_regressor')
    has_color = hasattr(model, 'color_regressor')

    # Training loop
    model.train()
    for batch, act_latents, rel_latents, labels in tqdm(train_loader, desc="Train probes", dynamic_ncols=True):
        batch = batch.to(device)
        labels = labels.to(device)
        rel_latents = rel_latents.to(device)

        optimizer.zero_grad()
        loss = 0.

        enc_inputs = batch[:, :2].reshape(-1, 3, img_size, img_size)
        reg_features = model.encoder(enc_inputs).detach()
        reg_features = reg_features.reshape(-1, model.res_out_dim * 2)
        target_rel = rel_latents[:, 0]

        if has_rot and has_color:
            loss += mse(model.rot_regressor(reg_features), target_rel[:, :4])
            loss += mse(model.color_regressor(reg_features), target_rel[:, 4:])
        elif has_rot:
            loss += mse(model.rot_regressor(reg_features), target_rel)
        elif has_color:
            loss += mse(model.color_regressor(reg_features), target_rel)

        loss += cross_entropy(model.res_classifier(reg_features[:, :model.res_out_dim]), labels)

        loss.backward()
        optimizer.step()

    # Eval loop
    model.eval()
    all_preds, all_targets = [], []
    test_preds_res, test_targets = [], []

    with torch.no_grad():
        for batch, act_latents, rel_latents, labels in tqdm(val_loader, desc="Eval probes", dynamic_ncols=True):
            batch = batch.to(device)
            labels = labels.to(device)
            rel_latents = rel_latents.to(device)

            enc_inputs = batch[:, :2].reshape(-1, 3, img_size, img_size)
            reg_features = model.encoder(enc_inputs)
            reg_features = reg_features.reshape(-1, model.res_out_dim * 2)
            target_rel = rel_latents[:, 0]

            if has_rot and has_color:
                pred_rot = model.rot_regressor(reg_features)
                pred_color = model.color_regressor(reg_features)
                pred = torch.cat([pred_rot, pred_color], dim=1)
            elif has_rot:
                pred = model.rot_regressor(reg_features)
            elif has_color:
                pred = model.color_regressor(reg_features)
            else:
                pred = torch.zeros_like(target_rel)

            all_preds.append(pred.cpu().numpy())
            all_targets.append(target_rel.cpu().numpy())

            res_out = model.res_classifier(reg_features[:, :model.res_out_dim])
            test_preds_res.extend(torch.argmax(res_out, dim=1).cpu().numpy())
            test_targets.extend(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    results = {
        "R2": r2_score(all_targets, all_preds),
        "test_acc_res": accuracy_score(test_targets, test_preds_res) * 100.0,
    }
    if has_rot and has_color:
        results["R2_rot"] = r2_score(all_targets[:, :4], all_preds[:, :4])
        results["R2_color"] = r2_score(all_targets[:, 4:], all_preds[:, 4:])

    return results


#########


def val_all_one_epoch_aug(model, optimizer, device, train_loader, val_loader, img_size, latent_type):
    def get_latent_slice(params, lt):
        if lt == "crop":
            return params[:, :4]
        elif lt == "blur":
            return params[:, 8].unsqueeze(1)
        elif lt == "colorjitter":
            return params[:, 4:8]
        return params

    mse = nn.MSELoss()
    cross_entropy = nn.CrossEntropyLoss()
    has_crop = hasattr(model, 'crop_regressor')
    has_blur = hasattr(model, 'blur_regressor')
    has_jitter = hasattr(model, 'jitter_regressor')

    # Training loop
    model.train()
    for augmented_images, augmented_params, labels, orig_img in tqdm(train_loader, desc="Train probes", dynamic_ncols=True):
        batch = augmented_images.to(device)
        labels = labels.to(device)
        orig_img = orig_img.to(device)
        augmented_params = augmented_params.to(device)
        augmented_params = torch.nn.functional.normalize(augmented_params, p=2, dim=1)
        rel_latents = augmented_params[:, 1] - augmented_params[:, 0]

        optimizer.zero_grad()
        loss = 0.

        enc_inputs = batch[:, :2].reshape(-1, 3, img_size, img_size)
        reg_features = model.encoder(enc_inputs).detach()
        reg_features = reg_features.reshape(-1, model.res_out_dim * 2)

        if has_crop and has_blur and has_jitter:
            loss += mse(model.crop_regressor(reg_features), rel_latents[:, :4])
            loss += mse(model.blur_regressor(reg_features), rel_latents[:, 8].unsqueeze(1))
            loss += mse(model.jitter_regressor(reg_features), rel_latents[:, 4:8])
        elif has_crop:
            loss += mse(model.crop_regressor(reg_features), get_latent_slice(rel_latents, latent_type))
        elif has_blur:
            loss += mse(model.blur_regressor(reg_features), get_latent_slice(rel_latents, latent_type))
        elif has_jitter:
            loss += mse(model.jitter_regressor(reg_features), get_latent_slice(rel_latents, latent_type))

        res_features = model.encoder(orig_img).detach()
        loss += cross_entropy(model.res_classifier(res_features), labels)

        loss.backward()
        optimizer.step()

    # Eval loop
    model.eval()
    all_preds, all_targets = [], []
    test_preds_res, test_targets = [], []

    with torch.no_grad():
        for augmented_images, augmented_params, labels, orig_img in tqdm(val_loader, desc="Eval probes", dynamic_ncols=True):
            batch = augmented_images.to(device)
            labels = labels.to(device)
            orig_img = orig_img.to(device)
            augmented_params = augmented_params.to(device)
            augmented_params = torch.nn.functional.normalize(augmented_params, p=2, dim=1)
            rel_latents = augmented_params[:, 1] - augmented_params[:, 0]
            target_rel = get_latent_slice(rel_latents, latent_type)

            enc_inputs = batch[:, :2].reshape(-1, 3, img_size, img_size)
            reg_features = model.encoder(enc_inputs)
            reg_features = reg_features.reshape(-1, model.res_out_dim * 2)

            if has_crop and has_blur and has_jitter:
                pred_crop = model.crop_regressor(reg_features)
                pred_blur = model.blur_regressor(reg_features)
                pred_jitter = model.jitter_regressor(reg_features)
                pred = torch.cat([pred_crop, pred_jitter, pred_blur], dim=1)  # [crop(4), jitter(4), blur(1)]
                target = torch.cat([rel_latents[:, :4], rel_latents[:, 4:8], rel_latents[:, 8].unsqueeze(1)], dim=1)
            elif has_crop:
                pred = model.crop_regressor(reg_features)
                target = target_rel
            elif has_blur:
                pred = model.blur_regressor(reg_features)
                target = target_rel
            elif has_jitter:
                pred = model.jitter_regressor(reg_features)
                target = target_rel
            else:
                pred = torch.zeros_like(target_rel)
                target = target_rel

            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.cpu().numpy())

            res_features = model.encoder(orig_img)
            res_out = model.res_classifier(res_features)
            test_preds_res.extend(torch.argmax(res_out, dim=1).cpu().numpy())
            test_targets.extend(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    results = {
        "R2": r2_score(all_targets, all_preds),
        "test_acc_res": accuracy_score(test_targets, test_preds_res) * 100.0,
    }
    if has_crop and has_blur and has_jitter:
        results["R2_crop"] = r2_score(all_targets[:, :4], all_preds[:, :4])
        results["R2_jitter"] = r2_score(all_targets[:, 4:8], all_preds[:, 4:8])
        results["R2_blur"] = r2_score(all_targets[:, 8:], all_preds[:, 8:])

    return results

############
def val_all_one_epoch_pls(model, device, train_loader, test_loader, img_size, fovea_size, optimizer, conv_jepa=False):
    """
    Train and evaluate probes for SeqJEPA_PLS / Conv_IJEPA in eval mode.
    Probes: pos_regressor, res_classifier, agg_classifier (seqjepa only)
    """
    cross_entropy = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    res_out_dim = model.res_out_dim
    has_agg = hasattr(model, 'agg_classifier') and model.agg_classifier is not None

    # Training loop
    model.train()
    train_targets, train_preds_res, train_preds_agg = [], [], []
    train_equi_preds, train_equi_targets = [], []
    running_train_loss_res, running_train_loss_agg, running_train_loss_equi = 0., 0., 0.

    for batch, probs, labels, patches, sac_pos in tqdm(train_loader, desc="Training probes"):
        patches = patches.to(device)
        labels = labels.to(device)
        sac_pos = sac_pos.to(device)

        optimizer.zero_grad()

        # Get encodings
        foveated_x_obs = patches[:, :-1]
        foveated_x_last = patches[:, -1]
        
        if conv_jepa:
            _, _, z1, z2 = model(foveated_x_obs, foveated_x_last, sac_pos)
        else:
            _, agg_out, z1, z2 = model(foveated_x_obs, foveated_x_last, sac_pos)

        loss = 0.

        # Res classifier
        res_out = model.res_classifier(z1.detach())
        loss_res = cross_entropy(res_out, labels)
        loss += loss_res
        running_train_loss_res += loss_res.item()
        train_preds_res.extend(torch.argmax(res_out, dim=1).cpu().numpy())
        train_targets.extend(labels.cpu().numpy())

        # Agg classifier (seqjepa only)
        if has_agg and not conv_jepa:
            agg_out_cls = model.agg_classifier(agg_out.detach())
            loss_agg = cross_entropy(agg_out_cls, labels)
            loss += loss_agg
            running_train_loss_agg += loss_agg.item()
            train_preds_agg.extend(torch.argmax(agg_out_cls, dim=1).cpu().numpy())

        # Pos regressor (equivariance)
        enc_input = patches[:, :2].reshape(-1, 3, fovea_size, fovea_size)
        reg_features = model.encoder(enc_input).detach()
        reg_features = reg_features.reshape(-1, res_out_dim * 2)
        sac_pos_1 = sac_pos[:, 0]
        sac_pos_2 = sac_pos[:, 1]
        rel_pos = sac_pos_2 - sac_pos_1
        outputs_reg = model.pos_regressor(reg_features)
        loss_equi = mse(outputs_reg, rel_pos)
        loss += loss_equi
        running_train_loss_equi += loss_equi.item()
        train_equi_preds.append(outputs_reg.detach().cpu().numpy())
        train_equi_targets.append(rel_pos.cpu().numpy())

        loss.backward()
        optimizer.step()

    train_acc_res = accuracy_score(train_targets, train_preds_res) * 100.0
    train_r2 = r2_score(np.concatenate(train_equi_targets), np.concatenate(train_equi_preds))
    train_acc_agg = accuracy_score(train_targets, train_preds_agg) * 100.0 if has_agg and not conv_jepa else 0.

    # Eval loop
    model.eval()
    test_targets, test_preds_res, test_preds_agg = [], [], []
    test_equi_preds, test_equi_targets = [], []
    running_test_loss_res, running_test_loss_agg, running_test_loss_equi = 0., 0., 0.

    with torch.no_grad():
        for batch, probs, labels, patches, sac_pos in tqdm(test_loader, desc="Evaluating probes"):
            patches = patches.to(device)
            labels = labels.to(device)
            sac_pos = sac_pos.to(device)

            foveated_x_obs = patches[:, :-1]
            foveated_x_last = patches[:, -1]

            if conv_jepa:
                _, _, z1, z2 = model(foveated_x_obs, foveated_x_last, sac_pos)
            else:
                _, agg_out, z1, z2 = model(foveated_x_obs, foveated_x_last, sac_pos)

            # Res classifier
            res_out = model.res_classifier(z1)
            loss_res = cross_entropy(res_out, labels)
            running_test_loss_res += loss_res.item()
            test_preds_res.extend(torch.argmax(res_out, dim=1).cpu().numpy())
            test_targets.extend(labels.cpu().numpy())

            # Agg classifier
            if has_agg and not conv_jepa:
                agg_out_cls = model.agg_classifier(agg_out)
                loss_agg = cross_entropy(agg_out_cls, labels)
                running_test_loss_agg += loss_agg.item()
                test_preds_agg.extend(torch.argmax(agg_out_cls, dim=1).cpu().numpy())

            # Pos regressor
            enc_input = patches[:, :2].reshape(-1, 3, fovea_size, fovea_size)
            reg_features = model.encoder(enc_input)
            reg_features = reg_features.reshape(-1, res_out_dim * 2)
            sac_pos_1 = sac_pos[:, 0]
            sac_pos_2 = sac_pos[:, 1]
            rel_pos = sac_pos_2 - sac_pos_1
            outputs_reg = model.pos_regressor(reg_features)
            loss_equi = mse(outputs_reg, rel_pos)
            running_test_loss_equi += loss_equi.item()
            test_equi_preds.append(outputs_reg.cpu().numpy())
            test_equi_targets.append(rel_pos.cpu().numpy())

    test_acc_res = accuracy_score(test_targets, test_preds_res) * 100.0
    test_r2 = r2_score(np.concatenate(test_equi_targets), np.concatenate(test_equi_preds))
    test_acc_agg = accuracy_score(test_targets, test_preds_agg) * 100.0 if has_agg and not conv_jepa else 0.

    results = {
        "train_acc_res": train_acc_res,
        "train_acc_agg": train_acc_agg,
        "train_r2": train_r2,
        "train_loss_res": running_train_loss_res / len(train_loader),
        "train_loss_agg": running_train_loss_agg / len(train_loader) if has_agg else 0.,
        "train_loss_equi": running_train_loss_equi / len(train_loader),
        "test_acc_res": test_acc_res,
        "test_acc_agg": test_acc_agg,
        "test_r2": test_r2,
        "test_loss_res": running_test_loss_res / len(test_loader),
        "test_loss_agg": running_test_loss_agg / len(test_loader) if has_agg else 0.,
        "test_loss_equi": running_test_loss_equi / len(test_loader),
    }
    return results
