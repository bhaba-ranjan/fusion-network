from comet_ml import Experiment
import torch
import os
import numpy as np
import math
from random import shuffle

from data_builder.indexer import IndexDataset
from data_builder.transformer import ApplyTransformation
from model_builder.multimodal.fusion_net import BcFusionModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import coloredlogs, logging
from model_builder.multimodal.multi_net import MultiModalNet
from data_builder.cmd_scaler import transform_to_gt_scale
from data_builder.gaussian_weights import get_gaussian_weights


from torch.optim.lr_scheduler import MultiStepLR,CosineAnnealingWarmRestarts

# Create an experiment with your api key
experiment = Experiment(
    api_key="Ly3Tc8W7kfxPmAxsArJjX9cgo",    
    project_name="multimodal-net-with-rnn",
    workspace="bhabaranjan",
)

experiment.add_tag('end-to-end-angler-bc')
experiment.log_asset('/scratch/bpanigr/fusion-network/scripts/model_builder/multimodal/multi_net.py')

coloredlogs.install()

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f'Using device ========================================>  {device}')

min_val_error = 100000

val_dict = {}

# root_path = '/home/ranjan/Workspace/my_works/fusion-network/recorded-data'
# model_storage_path = '/home/ranjan/Workspace/my_works/fusion-network/scripts'

root_path = '/scratch/bpanigr/fusion-network/recorded-data'
model_storage_path = '/scratch/bpanigr/model_weights/end-to-end'

weights = get_gaussian_weights(6,3)
weights = weights[:,:-1] 
weights = np.concatenate([weights, weights], axis=1)
weights = torch.tensor(weights)
weights = weights.to(device)

def get_loss_fun(loss_type = None):
    if loss_type == 'mse':
        return torch.nn.MSELoss()
    else:
       return torch.nn.L1Loss()

def get_data_loader(input_file_path, read_type, batch_size):
    logging.info(f'Reading {read_type} file from path {input_file_path}')
    indexer = IndexDataset(input_file_path)
    transformer = ApplyTransformation(indexer)
    data_loader = DataLoader(transformer, batch_size = batch_size, drop_last=False, prefetch_factor=1,num_workers=20)
    return data_loader
   

def get_loss(loss_fn, pts, gt_pts, data_src):
    error =  loss_fn(pts, gt_pts)     
    
    if data_src == 'validation':     
        experiment.log_metric(name = str('way_pts'+data_src), value= error.item())    
    else:
        experiment.log_metric(name = str('way_pts'+data_src), value=error.item())      
    return error


def run_validation(val_files, model, batch_size, epoch, optim):
       print("Running Validation..\n")
       running_error = []
       loss = get_loss_fun()
       model.eval()
       with torch.no_grad():
        for val_file in val_files:
            val_loader = None
            if val_file not in val_dict:
                val_loader = get_data_loader( val_file, 'validation', batch_size = batch_size )
                val_dict[val_file] = val_loader
            else:
                val_loader = val_dict[val_file]

            per_file_loss_fusion  = []
            per_file_loss_ǐmage = []
            per_file_loss_pcl = []
            per_file_total_loss = []
            per_file_loss_fsn_path = []
            for index, (stacked_images, pcl ,local_goal, gt_pts, gt_cmd_vel) in tqdm(enumerate(val_loader)):
                stacked_images = stacked_images.to(device)
                pcl = pcl.to(device)
                local_goal= local_goal.to(device)
                gt_pts = gt_pts.to(device)
                gt_cmd_vel= gt_cmd_vel.to(device)
                
                pred_cmd, img_path, pcl_path, fsn_path = model(stacked_images, pcl, local_goal)

                # print(gt_cmd_vel.shape)
                # print(pred_cmd.shape)

                gt_cmd_vel = transform_to_gt_scale(gt_cmd_vel, device)                
                pred_cmd = transform_to_gt_scale(pred_cmd, device)  
                
                error_fusion = get_loss(loss, pred_cmd, gt_cmd_vel, 'validation')    
                error_img = get_loss(loss, img_path/ weights, gt_pts/ weights, 'validation')
                error_pcl = get_loss(loss, pcl_path/ weights, gt_pts/ weights, 'validation')
                error_fusion_path = get_loss(loss, fsn_path/ weights, gt_pts/ weights, 'validation')

                error_total = error_fusion + error_img + error_pcl+ error_fusion_path

                per_file_loss_fusion.append(error_fusion.item())
                per_file_loss_ǐmage.append(error_img.item())
                per_file_loss_pcl.append(error_pcl.item())   
                per_file_total_loss.append(error_total.item())
                per_file_loss_fsn_path.append(error_fusion_path.item())
                                                        
                
            experiment.log_metric(name = str('val_'+val_file.split('/')[-1]+'_img'), value=np.average(per_file_loss_ǐmage), epoch= epoch+1)
            experiment.log_metric(name = str('val_'+val_file.split('/')[-1]+'_pcl'), value=np.average(per_file_loss_pcl), epoch= epoch+1)
            experiment.log_metric(name = str('val_'+val_file.split('/')[-1]+'_fusion'), value=np.average(per_file_loss_fusion), epoch= epoch+1)                           

            print(f'============= perfile loss fns: {np.average(per_file_loss_fusion)}  img: {np.average(per_file_loss_ǐmage)}  pcl: {np.average(per_file_loss_pcl)} error_fusion_path: {np.average(per_file_loss_fsn_path)} \n')
            running_error.append(np.average(per_file_total_loss))
        
        avg_loss_on_validation = np.average(running_error)
        # print(f'epoch:------>{epoch}')
        if (epoch+1) % 10 == 0:
            print(f"saving model weights at validation error {avg_loss_on_validation}")
            torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            }, f'{model_storage_path}/resume_frozen_end_to_end_velocities_{epoch+1}_{avg_loss_on_validation}.pth')

        print(f'=========================> Average Validation error is:   { avg_loss_on_validation } \n')
        return avg_loss_on_validation            


def run_training(train_files, val_dirs, batch_size, num_epochs):
    loss = get_loss_fun()
    model = MultiModalNet()    

    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=0.00000888)     
    # run_validation(val_dirs, model, batch_size, 0, optim)
    # return
    
    ckpt = torch.load('/scratch/bpanigr/model_weights/end-to-end/end_to_end_velocities_150_0.5871485762779396.pth')
    model.load_state_dict(ckpt['model_state_dict'])
    optim.load_state_dict(ckpt['optimizer_state_dict'])

    scheduler = MultiStepLR(optim, milestones= [80,160,300], gamma=.9)

    data_dict = {}
    for epoch in range(num_epochs):
        num_files = 0
        lr = scheduler.get_last_lr()        
        experiment.log_metric( name = "Learning Rate Decay", value = lr, epoch= epoch+1)
        running_loss = []
        shuffle(train_files)  
        # clear_dict(data_dict,epoch)      
        model.train()
        for train_file in train_files:        
            train_loader = None 
            print(train_file)
            if train_file not in data_dict:
                train_loader = get_data_loader( train_file, 'train', batch_size = batch_size )   
                data_dict[train_file] = train_loader
            else:
                train_loader = data_dict[train_file]

            num_files += 1
            per_file_loss_fusion = [] 
            per_file_loss_ǐmage = [] 
            per_file_loss_pcl = [] 
            per_file_total_loss = []
            per_file_loss_fsn_path = []
            for index, (stacked_images, pcl ,local_goal, gt_pts, gt_cmd_vel) in enumerate(train_loader):

                # print(f'gt_cmd: {gt_cmd_vel}')
                # print(f'prev_cmd_vel:{prev_cmd_vel}')
                
                stacked_images = stacked_images.to(device)
                pcl = pcl.to(device)
                local_goal= local_goal.to(device)                
                gt_cmd_vel= gt_cmd_vel.to(device)
                gt_pts = gt_pts.to(device)
                # print(f"{pcl.shape = }")
                optim.zero_grad()
                
                pred_cmd, img_path, pcl_path, fsn_path = model(stacked_images, pcl, local_goal)
                
                
                error_fusion = get_loss(loss, pred_cmd, gt_cmd_vel,'train_pcl')
                error_img = get_loss(loss, img_path, gt_pts, 'train_pcl')
                error_pcl = get_loss(loss, pcl_path, gt_pts, 'train_pcl')
                error_fusion_path = get_loss(loss, fsn_path, gt_pts, 'train_pcl')
                
                error_total = error_fusion + error_img + error_pcl+ error_fusion_path

                per_file_loss_fusion.append(error_fusion.item())
                per_file_loss_ǐmage.append(error_img.item())
                per_file_loss_pcl.append(error_pcl.item())                
                per_file_total_loss.append(error_total.item())
                per_file_loss_fsn_path.append(error_fusion_path.item())

                error_total.backward()
                optim.step()

                
                print(f' Step is:  {index}  Loss is fusion :: {error_fusion.item()}  img:{error_img.item()} pcl:{error_pcl.item()} fns_path: {error_fusion_path.item()}\n')
            
            experiment.log_metric(name = str(train_file.split('/')[-1]+ " mod:" +'img'), value=np.average(per_file_loss_ǐmage), epoch= epoch+1)
            experiment.log_metric(name = str(train_file.split('/')[-1]+" mod:" +'pcl'), value=np.average(per_file_loss_pcl), epoch= epoch+1)
            experiment.log_metric(name = str(train_file.split('/')[-1]+" mod:" +'fusion'), value=np.average(per_file_loss_fusion), epoch= epoch+1)
            print(f' \n ========== perfile loss fns: {np.average(per_file_loss_fusion)}  img: {np.average(per_file_loss_ǐmage)}  pcl: {np.average(per_file_loss_pcl)} fns_path: {np.average(per_file_loss_fsn_path)}   ==================\n')      
            running_loss.append(np.average(per_file_total_loss))   
            
        scheduler.step()
        
        print(f'================== epoch is: {epoch} and error is: {np.average(running_loss)}\n')

        if (epoch+1) % 2 == 0:
            val_error = run_validation(val_dirs, model, batch_size, epoch, optim)
            experiment.log_metric( name = "Avg Validation loss", value = val_error, epoch= epoch+1)
        # val_error_at_epoch.append(val_error)
        experiment.log_metric( name = "Avg Training loss", value = np.average(running_loss), epoch= epoch+1)
        


def main():
    train_path = f'{root_path}/train'
    validation_path = f'{root_path}/val'
    
    train_dirs = [ os.path.join(train_path, dir) for dir in os.listdir(train_path)]
    val_dirs = [ os.path.join(validation_path, dir) for dir in os.listdir(validation_path)]

    train_dirs.remove('/scratch/bpanigr/fusion-network/recorded-data/train/136021_wt')
    train_dirs.remove('/scratch/bpanigr/fusion-network/recorded-data/train/138181_wt')
    train_dirs.remove('/scratch/bpanigr/fusion-network/recorded-data/train/135968_wt_at')
    # # train_dirs.remove('/scratch/bpanigr/fusion-network/recorded-data/train/136514_sw_wt_sc')
    train_dirs.remove('/scratch/bpanigr/fusion-network/recorded-data/train/135967_at')

    batch_size = 200
    epochs = 350
    run_training(train_dirs, val_dirs, batch_size, epochs)

main()



