from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model
import torch
import os
import numpy as np
import sys

from data_builder.indexer import IndexDataset
from data_builder.transformer import ApplyTransformation
from model_builder.multimodal.fusion_net import BcFusionModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import coloredlogs, logging

from torch.optim.lr_scheduler import MultiStepLR


project_name = ""

if len(sys.argv) < 2:
    project_name = "test-multi-modal"
else:
    project_name = "multimodal-net-with-RNN"

# Create an experiment with your api key
experiment = Experiment(
    api_key="Ly3Tc8W7kfxPmAxsArJjX9cgo",
    project_name= project_name,
    workspace="bhabaranjan"
)

coloredlogs.install()

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f'Using device ========================================>  {device}')

min_val_error = 100000


def get_data_loader(input_file_path, read_type, batch_size):
    logging.info(f'Reading {read_type} file from path {input_file_path}')
    indexer = IndexDataset(input_file_path)
    transformer = ApplyTransformation(indexer)
    data_loader = DataLoader(transformer, batch_size = batch_size)
    return data_loader

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def run_validation(val_files, model, batch_size, epoch, optim):
       print("Running Validation..\n")
       running_error = []
       loss = torch.nn.MSELoss()
       with torch.no_grad():
        for val_file in val_files:        
            val_loader = get_data_loader( val_file, 'validation', batch_size = batch_size )
            per_file_loss_fusion  = []
            per_file_loss_ǐmage = []
            per_file_loss_pcl = []
            per_file_total_loss = []
            for index, (stacked_images, pcl ,local_goal, prev_cmd_vel, gt_cmd_vel) in tqdm(enumerate(val_loader)):
                stacked_images = stacked_images.to(device)
                pcl = pcl.to(device)
                local_goal= local_goal.to(device)
                prev_cmd_vel= prev_cmd_vel.to(device)
                gt_cmd_vel= gt_cmd_vel.to(device)
                
                pred_fusion, pred_img, pred_pcl = model(stacked_images, pcl, local_goal, prev_cmd_vel)
                
                

                error_fusion = loss(pred_fusion, gt_cmd_vel)
                error_img = loss(pred_img, gt_cmd_vel)
                error_pcl = loss(pred_pcl, gt_cmd_vel)
                
                error_total = error_fusion + ( 0.2 * error_img) + (0.8 * error_pcl)

                per_file_loss_fusion.append(error_fusion.item())
                per_file_loss_ǐmage.append(error_img.item())
                per_file_loss_pcl.append(error_pcl.item())                
                per_file_total_loss.append(error_total.item())
                
               
            experiment.log_metric(name = str('val_'+val_file.split('/')[-1]+'_img'), value=np.average(per_file_loss_ǐmage), epoch = epoch + 1)
            experiment.log_metric(name = str('val_'+val_file.split('/')[-1]+'_pcl'), value=np.average(per_file_loss_pcl), epoch = epoch + 1)
            experiment.log_metric(name = str('val_'+val_file.split('/')[-1]+'_fusion'), value=np.average(per_file_loss_fusion), epoch = epoch + 1)

            running_error.append(np.average(per_file_total_loss))

        avg_loss_on_validation = np.average(running_error)
        
        if (epoch+1) % 10 == 0 and epoch!=0:
            print(f"saving model weights at validation error {min_val_error}")
            model_checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            }
            torch.save(model_checkpoint, f'Model_at_epoch_{epoch+1}_val_loss{avg_loss_on_validation}')
            # log_model(experiment, model_checkpoint, model_name= f'Model_at_epoch_{epoch+1}')

        print(f'=========================> Average Validation error is:   {avg_loss_on_validation} \n')
        return avg_loss_on_validation
            


def run_training(train_files, val_dirs, batch_size, num_epochs):
    loss = torch.nn.MSELoss()
    model = BcFusionModel()
    model.to(device)

    val_error_at_epoch = []
    optim = torch.optim.Adadelta(model.parameters(), lr = 0.001) 
    learning_rate_scheduler = MultiStepLR(optimizer=optim, milestones=[15,30,60], gamma=10.0)
    epoch_loss = []
    for epoch in range(num_epochs):
        num_files = 0
        running_loss = []
        experiment.log_metric(name = 'lr', value=learning_rate_scheduler.get_last_lr(), epoch= epoch+1)
        for train_file in train_files:        
            train_loader = get_data_loader( train_file, 'train', batch_size = batch_size )   
            num_files += 1
            per_file_loss_fusion = [] 
            per_file_loss_ǐmage = [] 
            per_file_loss_pcl = [] 
            per_file_total_loss = []
            for index, (stacked_images, pcl ,local_goal, prev_cmd_vel, gt_cmd_vel) in enumerate(train_loader):
                
                stacked_images = stacked_images.to(device)
                pcl = pcl.to(device)
                local_goal= local_goal.to(device)
                prev_cmd_vel= prev_cmd_vel.to(device)
                gt_cmd_vel= gt_cmd_vel.to(device)
                # print(f"{gt_cmd_vel.shape = }")

                pred_fusion, pred_img, pred_pcl  = model(stacked_images, pcl, local_goal, prev_cmd_vel)
                
                # print(f"{pred_cmd_vel.shape = }")

                # pred_fusion  *= 1000
                # pred_img *= 1000
                # pred_pcl *= 1000
                # gt_cmd_vel *= 1000

                # print(f'fuison: {pred_fusion}')
                # print(f'gt: {gt_cmd_vel}')

                
                error_fusion = loss(pred_fusion, gt_cmd_vel)
                error_img = loss(pred_img, gt_cmd_vel)
                error_pcl = loss(pred_pcl, gt_cmd_vel)
                error_total = error_fusion + (0.2 * error_img) + (0.8 * error_pcl)
                

                optim.zero_grad()
                error_total.backward()
                optim.step()

                per_file_loss_fusion.append(error_fusion.item())
                per_file_loss_ǐmage.append(error_img.item())
                per_file_loss_pcl.append(error_pcl.item())
                per_file_total_loss.append(error_total.item())
                

                print(f'step is:   {index} and total error is:   {error_total.item()}  image: {error_img.item()}  pcl: {error_pcl.item()} fusion: {error_fusion.item()}\n')
            
            experiment.log_metric(name = str(train_file.split('/')[-1]+ " mod:" +'img'), value=np.average(per_file_loss_ǐmage), epoch= epoch+1)
            experiment.log_metric(name = str(train_file.split('/')[-1]+" mod:" +'pcl'), value=np.average(per_file_loss_pcl), epoch= epoch+1)
            experiment.log_metric(name = str(train_file.split('/')[-1]+" mod:" +'fusion'), value=np.average(per_file_loss_fusion), epoch= epoch+1)
            running_loss.append(np.average(per_file_total_loss))   
            
            # if num_files%6 == 0:  
            #     print("After trained on 6 files..")              
            #     run_validation(val_dirs, model, batch_size, epoch)
        
        # scheduler.step()        

        # epoch_loss.append(np.average(running_loss))                
        print(f'================== epoch is: {epoch} and error is: {np.average(running_loss)}==================\n')

        val_error = run_validation(val_dirs, model, batch_size, epoch, optim)
        # val_error_at_epoch.append(val_error)
        experiment.log_metric( name = "Avg Training loss", value = np.average(running_loss), epoch= epoch+1)
        experiment.log_metric( name = "Avg Validation loss", value = val_error, epoch= epoch+1)
        learning_rate_scheduler.step()
    # torch.save(model.state_dict(), "saved_fusion_model.pth")


def main():
    train_path = "../recorded-data/train"
    # train_path = "../recorded-data/sandbox"
    train_dirs = [ os.path.join(train_path, dir) for dir in os.listdir(train_path)]
    val_dirs = [ os.path.join('../recorded-data/val', dir) for dir in os.listdir('../recorded-data/val')]
    batch_size = 8
    epochs = 200
    run_training(train_dirs, val_dirs, batch_size, epochs)



main()