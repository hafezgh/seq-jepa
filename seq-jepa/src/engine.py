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



############ ONLINE PROBE TRAINING ####################

############
def online_inv_probe_train_one_epoch_stl10_pls(model, device, fovea_size, num_saccades,
                                               img_size, train_loader, val_loader, external_linprobe, optimizer_lin, res_out_eval=False):
    
    train_loss = 0.0
    train_targets = []
    train_predictions = []
    criterion = nn.CrossEntropyLoss()
    model.eval()
    external_linprobe.train()
    
    for i, (batch, probs, labels, patches, sac_pos, _) in enumerate(tqdm(train_loader, disable=False, dynamic_ncols=True)):
        batch = batch.to(device)
        #down_rgb = down_rgb.to(device)
        probs = probs.to(device)
        labels = labels.to(device)
        patches = patches.to(device)
        sac_pos = sac_pos.to(device)
        loss = 0.
        optimizer_lin.zero_grad()
        foveated_x_obs = patches[:,:-1]
        foveated_x_last = patches[:,-1]

        action_latents = sac_pos 
        optimizer_lin.zero_grad()
        if res_out_eval:
            agg_out = model.encoder(patches[:,0])
        else:
            _, agg_out, _, _, _ = model(foveated_x_obs, foveated_x_last, action_latents)
        
        output = external_linprobe(agg_out)
        
        loss = criterion(output, labels)
        train_loss += loss.item()
        #loss = AllReduce.apply(loss)
        loss.backward()
        optimizer_lin.step()

        preds = torch.argmax(output, dim=1)
        train_predictions.extend(preds.cpu().numpy())
        train_targets.extend(labels.cpu().numpy())
        
    train_acc = accuracy_score(train_targets, train_predictions)

    test_loss = 0.0
    test_targets = []
    test_predictions = []
    external_linprobe.eval()
    with torch.no_grad():
        for i, (batch, probs, labels, patches, sac_pos, _) in enumerate(tqdm(val_loader, disable=False, dynamic_ncols=True)):
            batch = batch.to(device)
            #down_rgb = down_rgb.to(device)
            probs = probs.to(device)
            labels = labels.to(device)
            patches = patches.to(device)
            sac_pos = sac_pos.to(device)
            foveated_x_obs = patches[:,:-1]
            foveated_x_last = patches[:,-1]
            action_latents = sac_pos
            if res_out_eval:
                agg_out = model.encoder(patches[:,0])
            else:
                _, agg_out, _, _, _ = model(foveated_x_obs, foveated_x_last, action_latents)
            output = external_linprobe(agg_out)
            
            loss = criterion(output, labels)
            test_loss += loss.item()

            preds = torch.argmax(output, dim=1)
            test_predictions.extend(preds.cpu().numpy())
            test_targets.extend(labels.cpu().numpy())
            
    test_acc = accuracy_score(test_targets, test_predictions)

    result = {
        'train_loss': train_loss / len(train_loader),
        'train_acc': train_acc,
        'test_acc': test_acc,
        'test_loss': test_loss / len(val_loader),
    }
    
    return result["test_acc"], result["test_loss"], result["train_acc"], result["train_loss"]


###########
def online_equi_probe_train_one_epoch_stl10_pls(model, device, train_loader, val_loader, fovea_size,
                                                external_equiprobe, optimizer_equi, img_size=96):

    mse = nn.MSELoss()
    running_loss_reg = 0.
    num_saccades = 2
    model.eval()
    res_out_dim = model.res_out_dim
    external_equiprobe.train()
    all_reg_outputs_train = []
    all_acts_train = []
    for i, (batch, probs, labels, patches, sac_positions, _) in enumerate(tqdm(train_loader, disable=False, dynamic_ncols=True)):
        batch = batch.to(device)
        probs = probs.to(device)
        labels = labels.to(device)
        patches = patches.to(device)
        sac_positions = sac_positions.to(device)
        sac_pos_norm1 = sac_positions[:,0]
        sac_pos_norm2 = sac_positions[:,1]
        rel_pos = sac_pos_norm2 - sac_pos_norm1
        
        ### regression (equivariance)
        enc_input = patches[:,:2].reshape(-1, 3, fovea_size, fovea_size)
        reg_features = model.encoder(enc_input)
        reg_features = reg_features.reshape(-1, res_out_dim*2)
        outputs_reg = external_equiprobe(reg_features)
        optimizer_equi.zero_grad()
        loss_reg = mse(outputs_reg, rel_pos)
        #loss = AllReduce.apply(loss)
        loss_reg.backward()
        optimizer_equi.step()
        running_loss_reg += loss_reg.item()
        all_acts_train.append(rel_pos.detach().cpu().numpy())
        all_reg_outputs_train.append(outputs_reg.detach().cpu().numpy())
    r2_train = r2_score(np.concatenate(all_acts_train, axis=0), np.concatenate(all_reg_outputs_train, axis=0))
    all_reg_outputs = []
    all_acts = []
    running_val_reg_loss = 0.
    external_equiprobe.eval()
    with torch.no_grad():
        for i, (batch, probs, labels, patches, sac_positions, _) in enumerate(tqdm(val_loader, disable=False, dynamic_ncols=True)):
            batch = batch.to(device)
            probs = probs.to(device)
            labels = labels.to(device)
            patches = patches.to(device)
            sac_positions = sac_positions.to(device)
            sac_pos_norm1 = sac_positions[:,0]
            sac_pos_norm2 = sac_positions[:,1]
            rel_pos = sac_pos_norm2 - sac_pos_norm1
            
            ### regression (equivariance)
            enc_input = patches[:,:2].reshape(-1, 3, fovea_size, fovea_size)
            reg_features = model.encoder(enc_input)
            reg_features = reg_features.reshape(-1, res_out_dim*2)
            outputs_reg = external_equiprobe(reg_features)
            #loss = AllReduce.apply(loss)
            val_loss = mse(outputs_reg, rel_pos)
            all_acts.append(rel_pos.cpu().numpy())
            all_reg_outputs.append(outputs_reg.cpu().numpy())
            running_val_reg_loss += val_loss.item()
            
    all_reg_outputs = np.concatenate(all_reg_outputs, axis=0)
    all_acts = np.concatenate(all_acts, axis=0)
    r2 = r2_score(all_acts, all_reg_outputs)
    
    results = {"R2": r2, "R2_train":r2_train, "val_reg_loss": running_val_reg_loss/len(val_loader), "train_reg_loss": running_loss_reg/len(train_loader)}
    return results["R2"], results["val_reg_loss"], results["R2_train"], results["train_reg_loss"]

############
def val_online_linprobe_3diebench(model, device, test_loader, online_linprobe, two_emb, sym=False):
    print("Validating online linear probe...")
    test_predictions = []
    test_targets = []
    criterion = nn.CrossEntropyLoss()
    running_val_loss = 0.
    with torch.no_grad():
        for i, (batch, act_latents, rel_latents, labels) in enumerate(tqdm(test_loader, disable=False, dynamic_ncols=True)):
            batch = batch.to(device)
            labels = labels.to(device)
            act_latents = act_latents.to(device)
            rel_latents = rel_latents.to(device)

            if sym:
                _, _, _, agg_out, _, _, _ = model(batch[:,:-1], batch[:,-1], act_latents[:,:-1], act_latents[:,-1], rel_latents=rel_latents)
            else:
                if two_emb:
                    agg_out = model.encoder(batch[:,0])
                else:
                    _, agg_out, _, _, _ = model(batch[:,:-1], batch[:,-1], act_latents[:,:-1], act_latents[:,-1], rel_latents=rel_latents)
            
            output = online_linprobe(agg_out)
            linloss = criterion(output, labels)
            running_val_loss += linloss.item()
            preds = torch.argmax(output, dim=1)
            test_predictions.extend(preds.cpu().numpy())
            test_targets.extend(labels.cpu().numpy())
            
    test_acc = accuracy_score(test_targets, test_predictions)
    loss_ = running_val_loss/len(test_loader)
    return test_acc, loss_

############

def val_online_equiprobe_3diebench(model, device, test_loader, online_equiprobe, img_size=128):
    print("Validating online equivaraince probe...")
    running_val_reg_loss = 0.
    mse = nn.MSELoss()
    res_out_dim = model.res_out_dim
    all_acts = []
    all_reg_outputs = []
    with torch.no_grad():
        for i, (batch, act_latents, rel_latents, labels) in enumerate(tqdm(test_loader, disable=False, dynamic_ncols=True)):
            batch = batch[:,[0,1]].to(device)
            labels = labels.to(device)
            act_latents = act_latents.to(device)
            rel_latents = rel_latents.to(device)
            
            enc_inputs = batch.reshape(-1, 3, img_size, img_size)
            reg_features = model.encoder(enc_inputs)
            reg_features = reg_features.reshape(-1, res_out_dim*2)
            outputs_reg = online_equiprobe(reg_features)
            targets = rel_latents[:,0]
            val_loss = mse(outputs_reg, targets)
            all_acts.extend(targets.cpu().numpy())
            all_reg_outputs.extend(outputs_reg.cpu().numpy())
            running_val_reg_loss += val_loss.item()
            
    r2 = r2_score(all_acts, all_reg_outputs)
    loss_ = running_val_reg_loss/len(test_loader)
    return r2, loss_

############

def val_online_linprobe_aug(model, device, two_emb, val_loader, external_linprobe, aug_type, img_size=224):
    test_predictions = []
    test_targets = []
    model.eval()
    external_linprobe.eval()
    criterion = nn.CrossEntropyLoss()
    running_val_loss = 0.
    with torch.no_grad():
        for i, (augmented_images, augmented_params, label, orig_images) in enumerate(tqdm(val_loader, disable=False, dynamic_ncols=True)):
            orig_images = orig_images.to(device)
            augmented_images = augmented_images.to(device)
            if aug_type == "crop":
                augmented_params = augmented_params[:,:,:4].to(device)
            elif aug_type == "blur":
                augmented_params = augmented_params[:,:,8].unsqueeze(2).to(device)
            elif aug_type == "colorjitter":
                augmented_params = augmented_params[:,:,4:8].to(device)
            else:
                augmented_params = augmented_params.to(device)
            augmented_params = torch.nn.functional.normalize(augmented_params, p=2, dim=1)
            labels = label.to(device)
            if two_emb:
                x = model.encoder(orig_images)
            else:
                _, x, _, _, _ = model(augmented_images[:,:-1], augmented_images[:,-1], augmented_params[:,:-1], augmented_params[:,-1])
            out = external_linprobe(x)
            preds = torch.argmax(out, dim=1)
            test_predictions.extend(preds.cpu().numpy())
            test_targets.extend(labels.cpu().numpy())
            linloss = criterion(out, labels)
            running_val_loss += linloss.item()
            #break
        test_acc = accuracy_score(test_targets, test_predictions)
        loss_ = running_val_loss/len(val_loader)
    return test_acc, loss_

############

def val_online_equiprobe_aug(model, device, val_loader, external_equiprobe, aug_type, img_size=224):
    all_reg_outputs = []
    all_acts = []
    mse = nn.MSELoss()
    model.eval()
    running_val_reg_loss = 0.
    external_equiprobe.eval()
    with torch.no_grad():
        for i, (augmented_images, augmented_params, label, orig_img) in enumerate(tqdm(val_loader, disable=False, dynamic_ncols=True)):
            batch = augmented_images[:,[0,1]].to(device)
            if aug_type == "crop":
                augmented_params = augmented_params[:,:,:4].to(device)
            elif aug_type == "blur":
                augmented_params = augmented_params[:,:,8].unsqueeze(2).to(device)
            elif aug_type == "colorjitter":
                augmented_params = augmented_params[:,:,4:8].to(device)
            else:
                augmented_params = augmented_params.to(device)
            augmented_params = torch.nn.functional.normalize(augmented_params, p=2, dim=1)
            latent_size = augmented_params.shape[-1]
            rel_params = augmented_params[:,1] - augmented_params[:,0]
            rel_params = rel_params.reshape(-1, latent_size)
            rel_latents = rel_params
            ### regression (equivariance)
            enc_inputs = batch.reshape(-1, 3, img_size, img_size)
            reg_features = model.encoder(enc_inputs)
            res_out_dim = model.res_out_dim
            reg_features = reg_features.reshape(-1, res_out_dim*2)
            outputs_reg = external_equiprobe(reg_features)
            val_loss = mse(outputs_reg, rel_latents)
            all_acts.append(rel_latents.cpu().numpy())
            all_reg_outputs.append(outputs_reg.cpu().numpy())
            running_val_reg_loss += val_loss.item()
            #break
            
    all_reg_outputs = np.concatenate(all_reg_outputs, axis=0)
    all_acts = np.concatenate(all_acts, axis=0)
    r2 = r2_score(all_acts, all_reg_outputs)
    
    results = {"R2": r2, "val_reg_loss": running_val_reg_loss/len(val_loader)}
    return r2, results["val_reg_loss"]


############
def train_one_epoch_aug(model, optimizer, device,
                              train_loader, val_loader, ema, two_emb, external_linprobe, optimizer_linprobe,
                              external_equiprobe, optimizer_equiprobe, ema_tau_base, current_epoch, num_epochs, aug_type, eval_type):

    model.train()
    prev_time = time.time()
    epoch_loss = 0.
    result = dict()
    tot_samples = 0.
    train_targets = []
    train_preds = []
    train_equi_preds = []
    train_equi_targets = []
    running_equi_loss_train = 0.
    running_lin_loss_train = 0.
    for i, (augmented_images, augmented_params, label, orig_img) in enumerate(tqdm(train_loader, disable=False, dynamic_ncols=True)):
        batch = augmented_images.to(device)
        labels = label.to(device)
        if aug_type == "crop":
            aug_params = augmented_params[:,:,:4].to(device)
        elif aug_type == "blur":
            aug_params = augmented_params[:,:,8].unsqueeze(2).to(device)
        elif aug_type == "colorjitter":
            aug_params = augmented_params[:,:,4:8].to(device)
        else:
            aug_params = augmented_params.to(device)
        aug_params = torch.nn.functional.normalize(aug_params, p=2, dim=1)
        latent_size = aug_params.shape[-1]
        #### the first two embs
        rel_params = aug_params[:,1] - aug_params[:,0]
        rel_params = rel_params.reshape(-1, latent_size)
        rel_latents = rel_params
        tot_loss = 0.
        loss = 0.
        optimizer.zero_grad()
        if two_emb:
            loss, z1, z2 = model(batch[:,0], batch[:,1], rel_latent=rel_latents, abs_latent=aug_params[:,1])
            agg_out = z1
        else: 
            loss, agg_out, z1, z2, _ = model(batch[:,:-1], batch[:,-1], aug_params[:,:-1], aug_params[:,-1])

        loss.backward()
        tot_loss += loss.item()

        optimizer.step()
        
        # MSE loss
        #criterion = nn.MSELoss(reduction='mean')
        tot_samples += labels.size(0)
        epoch_loss += tot_loss
        
        #### online linear probe
        agg_out_d = agg_out.detach()
        online_linout = external_linprobe(agg_out_d)
        criterion = nn.CrossEntropyLoss()
        optimizer_linprobe.zero_grad()
        online_linloss = criterion(online_linout, labels)
        #online_linloss = AllReduce.apply(online_linloss)
        online_linloss.backward()
        optimizer_linprobe.step()
        train_pred = torch.argmax(online_linout, dim=1)
        train_preds.extend(train_pred.detach().cpu().numpy())
        train_targets.extend(labels.cpu().numpy())
        running_lin_loss_train += online_linloss.item()
        
        #### online equivariance probe
        z1 = z1.detach()
        z2 = z2.detach()
        embs_concat = torch.cat((z1, z2), dim=1)
        pred_rel_latents = external_equiprobe(embs_concat)
        mse = nn.MSELoss()
        if eval_type == "crop":
            eval_act = augmented_params[:,:,:4].to(device)
        elif eval_type == "blur":
            eval_act = augmented_params[:,:,8].unsqueeze(2).to(device)
        elif eval_type == "colorjitter":
            eval_act = augmented_params[:,:,4:8].to(device)
        else:
            eval_act = augmented_params.to(device)
        eval_act = torch.nn.functional.normalize(eval_act, p=2, dim=1)
        eval_latent_size = eval_act.shape[-1]
        target_rel = eval_act[:,1] - eval_act[:,0]
        target_rel = target_rel.reshape(-1, eval_latent_size)
        online_equiloss = mse(pred_rel_latents, target_rel)
        optimizer_equiprobe.zero_grad()
        online_equiloss.backward()
        optimizer_equiprobe.step()
        train_equi_targets.extend(target_rel.cpu().numpy())
        train_equi_preds.extend(pred_rel_latents.detach().cpu().numpy())
        running_equi_loss_train += online_equiloss.item()

        if ema:
            model.update_moving_average()
            model.ema_decay = 1 - (1 - ema_tau_base) * ((math.cos(math.pi * current_epoch / num_epochs) + 1) / 2) ### cosine decay
            ##model.ema_decay = ema_tau_base + (1 - ema_tau_base) * (current_epoch / num_epochs) ### linear decay
        #break
    
    train_acc = accuracy_score(train_targets, train_preds)
    online_r2_train = r2_score(train_equi_targets, train_equi_preds)
    ##online_linacc = 0.0
    online_linacc, lin_loss_test = val_online_linprobe_aug(model, device, two_emb, val_loader, external_linprobe, aug_type, img_size=orig_img.size(-1))
    
    online_equi_r2 = 0.0
    r2_loss_test = 0.0
    online_equi_r2, r2_loss_test = val_online_equiprobe_aug(model, device, val_loader, external_equiprobe, eval_type, img_size=orig_img.size(-1))

    result["online_linacc_train"] = train_acc*100.0
    result["online_linacc_test"] = online_linacc*100.0
    result["online_r2_train"] = online_r2_train
    result["online_r2_test"] = online_equi_r2
    result["online_linloss_train"] = running_lin_loss_train/len(train_loader)
    result["online_r2_loss_train"] = running_equi_loss_train/len(train_loader)
    result["online_r2_loss_test"] = r2_loss_test
    result["online_linloss_test"] = lin_loss_test

    fin_time = time.time()
    tot_time = fin_time-prev_time
    
    epoch_loss = epoch_loss/tot_samples
    result["ep_loss"] = epoch_loss
    result["ep_time"] = tot_time

    return model, optimizer, result


def train_one_epoch_3diebench(model, train_loader, optimizer, device,
                              test_loader, ema, two_emb,
                              external_linprobe, optimizer_linprobe,
                              external_equiprobe, optimizer_equiprobe, ema_tau_base, current_epoch, num_epochs, ):
    mse = nn.MSELoss()
    model.train()
    prev_time = time.time()
    epoch_loss = 0.
    result = dict()
    tot_samples = 0.
    train_targets = []
    train_preds = []
    train_equi_preds = []
    train_equi_targets = []
    running_lin_loss_train = 0.
    running_equi_loss_train = 0.
    for i, (batch, act_latents, rel_latents, labels) in enumerate(tqdm(train_loader, disable=False, dynamic_ncols=True)):
        batch = batch.to(device)
        labels = labels.to(device)
        act_latents = act_latents.to(device)
        rel_latents = rel_latents.to(device)
        tot_loss = 0.
        loss = 0.
        optimizer.zero_grad()
        if two_emb:
            loss, emb1, emb2 = model(batch[:,0], batch[:,1], rel_latent=rel_latents[:,0], abs_latent=act_latents[:,1])
            agg_out = emb1
        else: 
            loss, agg_out, emb1, emb2, _ = model(batch[:,:-1], batch[:,-1], act_latents[:,:-1], act_latents[:,-1], rel_latents=rel_latents)
        
        loss.backward()
        tot_loss += loss.item()

        optimizer.step()
    
        tot_samples += labels.size(0)
        epoch_loss += tot_loss
        #### online linear probe
        agg_out_d = agg_out.detach()
        online_linout = external_linprobe(agg_out_d)
        criterion = nn.CrossEntropyLoss()
        optimizer_linprobe.zero_grad()
        online_linloss = criterion(online_linout, labels)
        #online_linloss = AllReduce.apply(online_linloss)
        online_linloss.backward()
        optimizer_linprobe.step()
        train_pred = torch.argmax(online_linout, dim=1)
        train_preds.extend(train_pred.detach().cpu().numpy())
        train_targets.extend(labels.cpu().numpy())
        running_lin_loss_train += online_linloss.item()
        
        #### online equivariance probe

        emb1 = emb1.detach()
        emb2 = emb2.detach()
        embs_concat = torch.cat((emb1, emb2), dim=1)
        pred_rel_latents = external_equiprobe(embs_concat)
        mse = nn.MSELoss()
        target_rel = rel_latents[:,0]
        online_equiloss = mse(pred_rel_latents, target_rel)
        optimizer_equiprobe.zero_grad()
        online_equiloss.backward()
        optimizer_equiprobe.step()
        train_equi_targets.extend(target_rel.cpu().numpy())
        train_equi_preds.extend(pred_rel_latents.detach().cpu().numpy())
        running_equi_loss_train += online_equiloss.item()
        
        if ema:
            model.update_moving_average()
            model.ema_decay = 1 - (1 - ema_tau_base) * ((math.cos(math.pi * current_epoch / num_epochs) + 1) / 2) ### cosine decay
            ##model.ema_decay = ema_tau_base + (1 - ema_tau_base) * (current_epoch / num_epochs) ### linear decay
    
    train_acc = accuracy_score(train_targets, train_preds)
    online_r2_train = r2_score(train_equi_targets, train_equi_preds)
   
    online_linacc, lin_loss_test = val_online_linprobe_3diebench(model, device, test_loader, external_linprobe, two_emb)
    online_r2 = 0.0
    r2_loss_test = 0.0

    online_r2, r2_loss_test = val_online_equiprobe_3diebench(model, device, test_loader, external_equiprobe)

    
    result["online_linacc_train"] = train_acc*100.0
    result["online_linacc_test"] = online_linacc*100.0
    result["online_r2_train"] = online_r2_train
    result["online_r2_test"] = online_r2
    result["online_linloss_train"] = running_lin_loss_train/len(train_loader)
    result["online_r2_loss_train"] = running_equi_loss_train/len(train_loader)
    result["online_r2_loss_test"] = r2_loss_test
    result["online_linloss_test"] = lin_loss_test

    
    fin_time = time.time()
    tot_time = fin_time-prev_time
    
    epoch_loss = epoch_loss/tot_samples
    result["ep_loss"] = epoch_loss
    result["ep_time"] = tot_time

    return model, optimizer, result
model, unlabeled_loader, optimizer, args.num_saccades, args.img_size,
                                            args.fovea_size, device, args.ema, args.ema_decay, epoch, epochs,
                                            train_loader=train_loader, test_loader=test_loader, dataset=args.dataset

def train_one_epoch_pls(model, data_loader, optimizer, num_saccades, image_size, fovea_size,
                        device, ema, ema_tau_base, current_epoch, num_epochs, conv_jepa=False, train_loader=None,
                        test_loader=None, dataset="stl10"):
    model.train()
    prev_time = time.time()
    epoch_loss = 0.
    result = dict()
    tot_samples = 0.
    res_out_eval = True if conv_jepa else False
    
    CELoss = nn.CrossEntropyLoss()
    MSELoss = nn.MSELoss()
    
    for i, (batch, probs, labels, patches, sac_pos) in enumerate(tqdm(data_loader, disable=False, dynamic_ncols=True)):
        batch = batch.to(device)
        #down_rgb = down_rgb.to(device)
        probs = probs.to(device)
        labels = labels.to(device)
        patches = patches.to(device)
        sac_pos = sac_pos.to(device)
        
        tot_loss = 0.
        loss = 0.
        optimizer.zero_grad()
        foveated_x_obs = patches[:,:-1].to(device)
        foveated_x_last = patches[:,-1].to(device)
        action_latents = sac_pos
        loss, agg_out, z1, z2 = model(foveated_x_obs, foveated_x_last, action_latents)
        
        if dataset == "imagenet":
            z1, z2 = z1.detach(), z2.detach()
            if conv_jepa == False:
                agg_out = agg_out.detach()
            online_linloss = CELoss(model.online_linprobe(agg_out), labels)
            loss += online_linloss

        loss.backward()
        tot_loss += loss.item()
        #torch.cuda.empty_cache()
        optimizer.step()

        tot_samples += labels.size(0)
        epoch_loss += tot_loss
        if ema:
            model.update_moving_average()
            model.ema_decay = 1 - (1 - ema_tau_base) * ((math.cos(math.pi * current_epoch / num_epochs) + 1) / 2) ### cosine decay
            ##model.ema_decay = ema_tau_base + (1 - ema_tau_base) * (current_epoch / num_epochs) ### linear decay
    
    if dataset == "stl10":
        ### train loader should be used for stl10 to train the probes because unlabeled dataset does not have labels
        for i, (batch, probs, labels, patches, sac_pos) in enumerate(tqdm(train_loader, disable=False, dynamic_ncols=True)):
            batch = batch.to(device)
            probs = probs.to(device)
            labels = labels.to(device)
            patches = patches.to(device)
            sac_pos = sac_pos.to(device)
            loss, agg_out, z1, z2 = model(foveated_x_obs, foveated_x_last, action_latents)
        
        
    result["online_linacc_train"] = online_linacc_train*100.0
    result["online_linacc_test"] = online_linacc_test*100.0
    result["online_r2_train"] = online_r2_train
    result["online_r2_test"] = online_equi_r2
    result["online_linloss_train"] = online_linloss_train
    result["online_r2_loss_train"] = r2_loss_train
    result["online_r2_loss_test"] = r2_loss_test
    result["online_linloss_test"] = online_linloss_test


    fin_time = time.time()
    tot_time = fin_time-prev_time
    
    epoch_loss = epoch_loss/tot_samples
    result["ep_loss"] = epoch_loss
    result["ep_time"] = tot_time

    return result
    
    

################################ EVALUATION FUNCTIONS ################################

######### new merged val functions ########

def val_all_one_epoch_3diebench(
    model, device, train_loader, val_loader, img_size,
    optimizer_rot, optimizer_color, optimizer_res_class, agg_eval, cond_latent=None, optimizer_agg_class=None,
):
    mse = nn.MSELoss()
    cross_entropy = nn.CrossEntropyLoss()
    
    running_loss_rot_reg = 0.
    running_loss_color_reg = 0.
    running_loss_res_class = 0.
    # if agg_eval:
    #     running_loss_agg_class = 0.
    
    model.train()
    for i, (batch, act_latents, rel_latents, labels) in enumerate(tqdm(train_loader, disable=False, dynamic_ncols=True)):
        batch = batch.to(device)
        labels = labels.to(device)
        act_latents = act_latents.to(device)
        rel_latents = rel_latents.to(device)
        
        enc_inputs = batch[:, :2].reshape(-1, 3, img_size, img_size)
        reg_features = model.encoder(enc_inputs)
        reg_features = reg_features.reshape(-1, model.res_out_dim * 2)
        
        outputs_rot = model.rot_regressor(reg_features)
        outputs_color = model.color_regressor(reg_features)
        
        optimizer_rot.zero_grad()
        optimizer_color.zero_grad()
        
        loss_rot_reg = mse(outputs_rot, rel_latents[:, 0, :4])
        loss_color_reg = mse(outputs_color, rel_latents[:,0, 4:])
        loss_rot_reg.backward()
        loss_color_reg.backward()
        
        optimizer_rot.step()
        optimizer_color.step()
        
        running_loss_rot_reg += loss_rot_reg.item()
        running_loss_color_reg += loss_color_reg.item()
        
        res_out = model.res_classifier(reg_features[:, :model.res_out_dim])
        optimizer_res_class.zero_grad()
        loss_res_class = cross_entropy(res_out, labels)
        loss_res_class.backward()
        optimizer_res_class.step()
        running_loss_res_class += loss_res_class.item()

        if agg_eval:
            for j in range(rel_latents.size(1)):
                optimizer_agg_class[j].zero_grad()
                batch_ = batch[:,:j+2]
                act_latents_ = act_latents[:,:j+2]
                rel_latents_ = rel_latents[:,:j+1]
                
                if cond_latent == "rot":
                    _, agg_out, _, _, _ = model(batch_[:, :-1], batch_[:, -1], act_latents_[:,:-1,:4], act_latents_[:,-1,:4], rel_latents=rel_latents_[:, :, :4])
                elif cond_latent == "color":
                    _, agg_out, _, _, _ = model(batch_[:, :-1], batch_[:, -1], act_latents_[:,:-1,4:], act_latents_[:,-1,4:], rel_latents=rel_latents_[:, :, 4:])
                elif cond_latent == "rotcolor":
                    _, agg_out, _, _, _ = model(batch_[:, :-1], batch_[:, -1], act_latents_[:,:-1], act_latents_[:,-1], rel_latents=rel_latents_)
                else:
                    raise ValueError("Invalid cond_latent value")
                outputs_agg_class = model.agg_classifier[j](agg_out)
                loss_agg_class = cross_entropy(outputs_agg_class, labels)
                loss_agg_class.backward()
                optimizer_agg_class[j].step()
                # running_loss_agg_class += loss_agg_class.item()
    
    model.eval()
    all_rot_outputs = []
    all_color_outputs = []
    all_acts = []
    running_val_rot_loss = 0.
    running_val_color_loss = 0.
    running_val_res_class_loss = 0.
    if agg_eval:
        #running_val_agg_class_loss = 0.
        test_predictions_agg = [[] for _ in range(batch.size(1)-1)]
    test_predictions_res = []
    test_targets = []
    
    with torch.no_grad():
        for i, (batch, act_latents, rel_latents, labels) in enumerate(tqdm(val_loader, disable=False, dynamic_ncols=True)):
            batch = batch.to(device)
            labels = labels.to(device)
            act_latents = act_latents.to(device)
            rel_latents = rel_latents.to(device)
            
            enc_inputs = batch[:, :2].reshape(-1, 3, img_size, img_size)
            reg_features = model.encoder(enc_inputs)
            reg_features = reg_features.reshape(-1, model.res_out_dim * 2)
            
            outputs_rot = model.rot_regressor(reg_features)
            outputs_color = model.color_regressor(reg_features)
            
            val_loss_rot = mse(outputs_rot, rel_latents[:, 0, :4])
            val_loss_color = mse(outputs_color, rel_latents[:, 0, 4:])
            running_val_rot_loss += val_loss_rot.item()
            running_val_color_loss += val_loss_color.item()
            
            all_acts.append(rel_latents[:, 0].cpu().numpy())
            all_rot_outputs.append(outputs_rot.cpu().numpy())
            all_color_outputs.append(outputs_color.cpu().numpy())
            
            res_out = model.res_classifier(reg_features[:, :model.res_out_dim])
            val_loss_res_class = cross_entropy(res_out, labels)
            running_val_res_class_loss += val_loss_res_class.item()
            test_predictions_res.extend(torch.argmax(res_out, dim=1).cpu().numpy())
            test_targets.extend(labels.cpu().numpy())
            
            if agg_eval:
                for j in range(rel_latents.size(1)):
                    batch_ = batch[:,:j+2]
                    act_latents_ = act_latents[:,:j+2]
                    rel_latents_ = rel_latents[:,:j+1]
                    if cond_latent == "rot":
                        _, agg_out, _, _, _ = model(batch_[:, :-1], batch_[:, -1], act_latents_[:,:-1,:4], act_latents_[:,-1,:4], rel_latents=rel_latents_[:, :, :4])
                    elif cond_latent == "color":
                        _, agg_out, _, _, _ = model(batch_[:, :-1], batch_[:, -1], act_latents_[:,:-1,4:], act_latents_[:,-1,4:], rel_latents=rel_latents_[:, :, 4:])
                    elif cond_latent == "rotcolor":
                        _, agg_out, _, _, _ = model(batch_[:, :-1], batch_[:, -1], act_latents_[:,:-1], act_latents_[:,-1], rel_latents=rel_latents_)
                    else:
                        raise ValueError("Invalid cond_latent value")
                    outputs_agg_class = model.agg_classifier[j](agg_out)
                    test_predictions_agg[j].extend(torch.argmax(outputs_agg_class, dim=1).cpu().numpy())
                    # val_loss_agg_class = cross_entropy(outputs_agg_class, labels)
                    # running_val_agg_class_loss += val_loss_agg_class.item()
    
    results = {
        "R2_rot": r2_score(np.concatenate(all_acts, axis=0)[:, :4], np.concatenate(all_rot_outputs, axis=0)),
        "R2_color": r2_score(np.concatenate(all_acts, axis=0)[:, 4:], np.concatenate(all_color_outputs, axis=0)),
        "test_acc_res": accuracy_score(test_targets, test_predictions_res) * 100.0,
    }
    
    if agg_eval:
        results["test_acc_agg"] = [accuracy_score(test_targets, test_predictions_agg[j]) * 100.0 for j in range(batch.size(1)-1)]
        ##results["test_acc_agg"] = [0, 0, 0, 0, accuracy_score(test_targets, test_predictions_agg[4]) * 100.0]
    return model, results


#########


def val_all_one_epoch_aug(
    model, device, train_loader, val_loader, img_size,
    optimizer_crop, optimizer_blur, optimizer_jitter, optimizer_res_class, agg_eval, cond_latent=None, optimizer_agg_class=None):

    
    mse = nn.MSELoss()
    cross_entropy = nn.CrossEntropyLoss()
    
    running_loss_crop_reg = 0.
    running_loss_blur_reg = 0.
    running_loss_jitter_reg = 0.
    running_loss_res_class = 0.
    # if agg_eval:
    #     running_loss_agg_class = 0.
    
    model.train()
    for i, (augmented_images, augmented_params, labels, orig_img) in enumerate(tqdm(train_loader, disable=False, dynamic_ncols=True)):
        batch = augmented_images.to(device)
        labels = labels.to(device)
        orig_img = orig_img.to(device)
        augmented_params = augmented_params.to(device)
        augmented_params = torch.nn.functional.normalize(augmented_params, p=2, dim=1)
        
        rel_latents = augmented_params[:,1] - augmented_params[:,0]
        enc_inputs = batch[:,:2].reshape(-1, 3, img_size, img_size)
        reg_features = model.encoder(enc_inputs)
        reg_features = reg_features.reshape(-1, model.res_out_dim * 2)
        
        outputs_crop = model.crop_regressor(reg_features)
        outputs_blur = model.blur_regressor(reg_features)
        outputs_jitter = model.jitter_regressor(reg_features)
        
        optimizer_crop.zero_grad()
        optimizer_blur.zero_grad()
        optimizer_jitter.zero_grad()
        loss_crop_reg = mse(outputs_crop, rel_latents[:,:4])
        loss_blur_reg = mse(outputs_blur, rel_latents[:,8].unsqueeze(1))
        loss_jitter_reg = mse(outputs_jitter, rel_latents[:,4:8])
        
        loss_crop_reg.backward()
        loss_blur_reg.backward()
        loss_jitter_reg.backward()
        
        optimizer_crop.step()
        optimizer_blur.step()
        optimizer_jitter.step()
        
        running_loss_crop_reg += loss_crop_reg.item()
        running_loss_blur_reg += loss_blur_reg.item()
        running_loss_jitter_reg += loss_jitter_reg.item()
        
        res_features = model.encoder(orig_img)
        res_out = model.res_classifier(res_features)
        optimizer_res_class.zero_grad()
        loss_res_class = cross_entropy(res_out, labels)
        loss_res_class.backward()
        optimizer_res_class.step()
        running_loss_res_class += loss_res_class.item()
        
        if agg_eval:
            for j in range(batch.size(1)-1):
                optimizer_agg_class[j].zero_grad()
                batch_ = batch[:,:j+2]
                augmented_params_ = augmented_params[:,:j+2]
                if cond_latent == "crop":
                    cond_params = augmented_params_[:, :, :4]
                elif cond_latent == "blur":
                    cond_params = augmented_params_[:, :, 8].unsqueeze(2)
                elif cond_latent == "colorjitter":
                    cond_params = augmented_params_[:, :, 4:8]
                elif cond_latent == "aug":
                    cond_params = augmented_params_
                else:
                    raise ValueError("Invalid cond_latent value")
                
                _, agg_out, _, _, _ = model(batch_[:, :-1], batch_[:, -1], cond_params[:, :-1], cond_params[:, -1])
                outputs_agg_class = model.agg_classifier[j](agg_out)
                loss_agg_class = cross_entropy(outputs_agg_class, labels)
                loss_agg_class.backward()
                optimizer_agg_class[j].step()
                ##running_loss_agg_class += loss_agg_class.item()
        
    model.eval()
    all_crop_outputs = []
    all_blur_outputs = []
    all_jitter_outputs = []
    all_acts = []
    test_predictions_res = []
    if agg_eval:
        #running_val_agg_class_loss = 0.
        test_predictions_agg = [[] for _ in range(batch.size(1)-1)]
    test_targets = []
    
    with torch.no_grad():
        for i, (augmented_images, augmented_params, labels, orig_img) in enumerate(tqdm(val_loader, disable=False, dynamic_ncols=True)):
            batch = augmented_images.to(device)
            labels = labels.to(device)
            orig_img = orig_img.to(device)
            augmented_params = augmented_params.to(device)
            augmented_params = torch.nn.functional.normalize(augmented_params, p=2, dim=1)
            rel_latents = augmented_params[:,1] - augmented_params[:,0]
            enc_inputs = batch[:,:2].reshape(-1, 3, img_size, img_size)
            reg_features = model.encoder(enc_inputs)
            reg_features = reg_features.reshape(-1, model.res_out_dim * 2)
            
            outputs_crop = model.crop_regressor(reg_features)
            outputs_blur = model.blur_regressor(reg_features)
            outputs_jitter = model.jitter_regressor(reg_features)

            all_acts.append(rel_latents.cpu().numpy())
            all_crop_outputs.append(outputs_crop.cpu().numpy())
            all_blur_outputs.append(outputs_blur.cpu().numpy())
            all_jitter_outputs.append(outputs_jitter.cpu().numpy())
            
            res_features = model.encoder(orig_img)
            res_out = model.res_classifier(res_features)
            test_predictions_res.extend(torch.argmax(res_out, dim=1).cpu().numpy())
            test_targets.extend(labels.cpu().numpy())
            
            if agg_eval:
                for j in range(batch.size(1)-1):
                    augmented_params_ = augmented_params[:,:j+2]
                    batch_ = batch[:,:j+2]
                    if cond_latent == "crop":
                        cond_params = augmented_params_[:, :, :4]
                    elif cond_latent == "blur":
                        cond_params = augmented_params_[:, :, 8].unsqueeze(2)
                    elif cond_latent == "colorjitter":
                        cond_params = augmented_params_[:, :, 4:8]
                    elif cond_latent == "aug":
                        cond_params = augmented_params_
                    else:
                        raise ValueError("Invalid cond_latent value")
                    
                    _, agg_out, _, _, _ = model(batch_[:, :-1], batch_[:, -1], cond_params[:, :-1], cond_params[:, -1])
                
                    outputs_agg_class = model.agg_classifier[j](agg_out)
                    test_predictions_agg[j].extend(torch.argmax(outputs_agg_class, dim=1).cpu().numpy())
    
    r2_crop = r2_score(np.concatenate(all_acts, axis=0)[:,:4], np.concatenate(all_crop_outputs, axis=0))
    r2_blur = r2_score(np.concatenate(all_acts, axis=0)[:,8], np.concatenate(all_blur_outputs, axis=0))
    r2_jitter = r2_score(np.concatenate(all_acts, axis=0)[:,4:8], np.concatenate(all_jitter_outputs, axis=0))
    test_acc_res = accuracy_score(test_targets, test_predictions_res) * 100.0
    
    results = {
        "R2_crop": r2_crop,
        "R2_blur": r2_blur,
        "R2_jitter": r2_jitter,
        "test_acc_res_aug": test_acc_res
    }
    
    if agg_eval:
        results["test_acc_agg"] = [accuracy_score(test_targets, test_predictions_agg[j]) * 100.0 for j in range(batch.size(1)-1)]
    
    return model, results

##########

def val_all_one_epoch_pls(
    model, device, train_loader, val_loader, img_size, fovea_size,
    optimizer_reg, optimizer_res_class, agg_eval, optimizer_agg_class=None
):
    
    mse = nn.MSELoss()
    cross_entropy = nn.CrossEntropyLoss()
    
    running_loss_reg = 0.
    running_loss_res_class = 0.
    # if agg_eval:
    #     running_loss_agg_class = 0.
    
    model.train()
    for i, (batch, _, labels, patches, sac_positions, _) in enumerate(tqdm(train_loader, disable=False, dynamic_ncols=True)):
        batch = batch.to(device)
        labels = labels.to(device)
        patches = patches.to(device)
        sac_positions = sac_positions.to(device)
        sac_pos_norm1 = sac_positions[:,0]
        sac_pos_norm2 = sac_positions[:,1]
        rel_pos = sac_pos_norm2 - sac_pos_norm1
        
        enc_input = patches[:,:2].reshape(-1, 3, fovea_size, fovea_size)
        reg_features = model.encoder(enc_input)
        reg_features = reg_features.reshape(-1, model.res_out_dim * 2)
        
        outputs_reg = model.pos_regressor(reg_features)
        optimizer_reg.zero_grad()
        loss_reg = mse(outputs_reg, rel_pos)
        loss_reg.backward()
        optimizer_reg.step()
        running_loss_reg += loss_reg.item()
        
        res_features = model.encoder(batch)
        res_out = model.res_classifier(res_features)
        
        optimizer_res_class.zero_grad()
        loss_res_class = cross_entropy(res_out, labels)
        loss_res_class.backward()
        optimizer_res_class.step()
        running_loss_res_class += loss_res_class.item()
        ##print(patches.size(1))
        if agg_eval:
            for j in range(patches.size(1)-1):
                optimizer_agg_class[j].zero_grad()
                patches_ = patches[:,:j+2]
                action_latents = sac_positions[:,:j+2]
                foveated_x_obs = patches_[:,:-1].to(device)
                foveated_x_last = patches_[:,-1].to(device)
                _, agg_out, _, _, _ = model(foveated_x_obs, foveated_x_last, action_latents)
                outputs_agg_class = model.agg_classifier[j](agg_out)
                loss_agg_class = cross_entropy(outputs_agg_class, labels)
                loss_agg_class.backward()
                optimizer_agg_class[j].step()
                ##running_loss_agg_class += loss_agg_class.item()
        
    model.eval()
    all_reg_outputs = []
    all_acts = []
    running_val_reg_loss = 0.
    running_val_res_class_loss = 0.
    if agg_eval:
        #running_val_agg_class_loss = 0.
        test_predictions_agg = [[] for _ in range(patches.size(1)-1)]
    test_predictions_res = []
    test_targets = []
    
    with torch.no_grad():
        for i, (batch, _, labels, patches, sac_positions, _) in enumerate(tqdm(val_loader, disable=False, dynamic_ncols=True)):
            batch = batch.to(device)
            labels = labels.to(device)
            patches = patches.to(device)
            sac_positions = sac_positions.to(device)
            
            sac_pos_norm1 = sac_positions[:,0]
            sac_pos_norm2 = sac_positions[:,1]
            rel_pos = sac_pos_norm2 - sac_pos_norm1
            
            enc_input = patches[:,:2].reshape(-1, 3, fovea_size, fovea_size)
            reg_features = model.encoder(enc_input)
            reg_features = reg_features.reshape(-1, model.res_out_dim * 2)
            
            outputs_reg = model.pos_regressor(reg_features)
            val_loss = mse(outputs_reg, rel_pos)
            running_val_reg_loss += val_loss.item()
            all_acts.append(rel_pos.cpu().numpy())
            all_reg_outputs.append(outputs_reg.cpu().numpy())
            
            res_features = model.encoder(batch)
            res_out = model.res_classifier(res_features)
            val_loss_res_class = cross_entropy(res_out, labels)
            running_val_res_class_loss += val_loss_res_class.item()
            test_predictions_res.extend(torch.argmax(res_out, dim=1).cpu().numpy())
            test_targets.extend(labels.cpu().numpy())
            
            ##print(patches.size(1))
            if agg_eval:
                for j in range(patches.size(1)-1):
                    patches_ = patches[:,:j+2]
                    action_latents = sac_positions[:,:j+2]
                    foveated_x_obs = patches_[:,:-1].to(device)
                    foveated_x_last = patches_[:,-1].to(device)
                    _, agg_out, _, _, _ = model(foveated_x_obs, foveated_x_last, action_latents)
                    outputs_agg_class = model.agg_classifier[j](agg_out)
                    ##running_loss_agg_class += loss_agg_class.item()
                    test_predictions_agg[j].extend(torch.argmax(outputs_agg_class, dim=1).cpu().numpy())
            
    all_reg_outputs = np.concatenate(all_reg_outputs, axis=0)
    all_acts = np.concatenate(all_acts, axis=0)
    r2 = r2_score(all_acts, all_reg_outputs)
    test_acc_res = accuracy_score(test_targets, test_predictions_res) * 100.0
    
    results = {
        "R2_pos": r2,
        "test_acc_res_pls": test_acc_res,
    }
    
    if agg_eval:
        results["test_acc_agg"] = [accuracy_score(test_targets, test_predictions_agg[j]) * 100.0 for j in range(patches.size(1)-1)]
    
    return model, results
    
