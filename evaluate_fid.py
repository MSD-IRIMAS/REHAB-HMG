def warn(*args, **kwargs):
    pass
import os
import torch
import warnings
import argparse
import numpy as np
import pandas as pd
warnings.warn = warn
from model.stgcn import STGCN
from utils.metrics import FID
from utils.visualize import create_directory
from dataset.dataset import Kimore, load_class
from torch.utils.data import DataLoader,Subset
from utils.utils import get_weights_loss,get_dirs
from sklearn.model_selection import train_test_split
from utils.normalize import normalize_skeletons, normalize_test_set

def get_args():
    parser = argparse.ArgumentParser(
    description="training STGCN to predict generated data score")

    parser.add_argument(
        '--stgcn',
        type=str,
        default='STGCN',
    )
    parser.add_argument(
        '--generative_model',
        type=str,
        default='CVAE',
    )

    parser.add_argument(
        '--dataset',
        help="Which dataset to use.",
        type=str,
        default='Kimore'
    )

    parser.add_argument(
        '--output-directory',
        type=str,
        default='results/'
    )

    parser.add_argument(
        '--runs',
        help="Number of experiments to do.",
        type=int,
        default=5
    )
    parser.add_argument(
        '--device',
        help="Device to run the training on.",
        type=str,
        choices=['cpu', 'cuda', 'mps'],
        default='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    )

    parser.add_argument(
        '--data_split',
        help="choose wether split the data or use it all",
        type=str,
        choices=['all', 'split'],
        default='split'
    )
    parser.add_argument(
        '--class_index',
        help="which class to generate",
        type=int,
        default=0
    )
    parser.add_argument(
        '--on',
        type=str,
        choices = ['real','generated'],
        default='generated',
    )

    args = parser.parse_args()

    return args
    
if __name__ == "__main__":

    args = get_args()

    output_directory_results = args.output_directory
    create_directory(output_directory_results)
   
    dataset_dir = 'data/' + args.dataset + '/'

    data,labels,scores = load_class(args.class_index,root_dir=dataset_dir)
   
    xtrain,xtest,ytrain,ytest,strain,stest= train_test_split(data,labels,scores,test_size=0.2,random_state=42)
    xtrain= normalize_skeletons(xtrain)
    train_set = Kimore(xtrain,ytrain,strain)
    train_loader = DataLoader(train_set,batch_size=10,shuffle =True)
    xtest= normalize_test_set(xtest)
    test_set = Kimore(xtest,ytest,stest)
    test_loader = DataLoader(test_set,batch_size=10,shuffle=False)

    if args.on == 'real':
        stgcn_directory = output_directory_results + 'stgcn/'
        if os.path.exists(stgcn_directory + 'fid.csv'):
            df = pd.read_csv(stgcn_directory + 'fid.csv')
        else:
            df = pd.DataFrame(columns=['class','FID'])

        # for class_index in range(n_classes):
       
        stgcn_run_dir = stgcn_directory + 'run_' + str(0) +'/class_'+str(args.class_index)+ '/best_stgcn.pth'
        fid_calculator = FID(model_path=stgcn_run_dir, device=args.device, data_loader=None)
        fid_score = fid_calculator.calculate_fid(train_loader, test_loader)
        new_row = pd.DataFrame([{'class': args.class_index, 'FID': np.mean(fid_score)}])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(stgcn_directory + 'fid.csv', index=False)

    elif args.on == 'generated':
        gen_weights = get_dirs(dir=gen_model_dir)
        for gen_weight in gen_weights:
            weights_loss = get_weights_loss(dir=gen_weight)
            print(weights_loss)
        stgcn_directory = output_directory_results + 'stgcn/'
        if os.path.exists(stgcn_directory + 'fid_gen.csv'):
            df = pd.read_csv(stgcn_directory + 'fid_gen.csv')
        else:
            df = pd.DataFrame(columns=['class','FID'])
        for class_index in range(n_classes):
            generator = CVAE(output_directory=gen_model_dir,
                    device=args.device,
                    w_rec=weights_loss['Wrec'],
                    w_kl=weights_loss['Wkl'])
            stgcn_run_dir = stgcn_directory + 'run_' + str(0) +'/class_'+str(class_index)+ '/best_stgcn.pth'
            fid_calculator = FID(model_path=stgcn_run_dir, device=args.device, data_loader=None)
            fid_score = fid_calculator.calculate_fid(train_loader, test_loader)
            new_row = pd.DataFrame([{'class': class_index, 'FID': np.mean(fid_score)}])
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(stgcn_directory + 'fid_gen.csv', index=False)
           
        


