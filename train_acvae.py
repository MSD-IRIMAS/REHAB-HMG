#-------------------------------------
# ------------------------This files train the Action conditioned VAE---------------------------------------
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import pandas as pd
import numpy as np
import argparse
from utils.visualize import create_directory
from dataset.dataset import Kimore, load_data,load_class
from model.acvae import ACVAE
from model.vae import VAE
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,Subset
import torch
from utils.normalize import normalize_skeletons, unnormalize_generated_skeletons



def get_args():
    parser = argparse.ArgumentParser(
    description="Choose which samples to train the GRU classifier on with the type of split.")

    parser.add_argument(
        '--generative-model',
        help="generative model to use .",
        type=str,
        choices=['ACVAE','VAE'],
        default='ACVAE',
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
        '--wrec',
        help="Weight for the reconstruction loss.",
        type=float,
        default=0.999
    )

    parser.add_argument(
        '--wkl',
        help="Weight for the kl loss.",
        type=float,
        default=1e-3
    )
    parser.add_argument(
        '--runs',
        help="Number of experiments to do.",
        type=int,
        default=5
    )


    parser.add_argument(
        '--epochs',
        help="Number of epochs to train the model.",
        type=int,
        default=2000
    )
    parser.add_argument(
        '--device',
        help="Device to run the training on.",
        type=str,
        choices=['cpu', 'cuda', 'mps'],
        default='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    )

    parser.add_argument(
        '--class_index',
        help="which class to generate from.",
        type=int,
        default=0
    )
    
    parser.add_argument(
        '--samples',
        help='how many samples to generate',
        type = int,
        default= 4,
    )

    args = parser.parse_args()

    return args
    
if __name__ == "__main__":

    args = get_args()

    output_directory_results = args.output_directory
    create_directory(output_directory_results)

    output_directory_gen_models = output_directory_results + 'Generative_models/'
    create_directory(output_directory_gen_models)

    output_directory_dataset = output_directory_gen_models + 'Action_conditioned/'
    create_directory(output_directory_dataset)

    output_directory_generator = output_directory_dataset + args.generative_model + '/'
    create_directory(output_directory_generator)

    output_directory_weights_losses = output_directory_generator + 'Wrec_' + str(args.wrec) + '_Wkl_' + str(args.wkl) + '/'
    create_directory(output_directory_weights_losses)
    
    dataset_dir = 'data/' + args.dataset + '/'
    

    for _run in range(args.runs):
            output_directory_run = output_directory_weights_losses + 'run_' + str(_run) + '/'
            create_directory(output_directory_run)
            output_directory_skeletons = output_directory_run  + 'class_' + str(args.class_index) + '/'
            create_directory(output_directory_skeletons)
            output_directory_skeletons_class = output_directory_skeletons + 'generated_samples/' 
            create_directory(output_directory_skeletons_class)
          
            if args.generative_model == 'ACVAE':
    
            #-----------------------Data Handling
                data,labels,scores = load_data(root_dir=dataset_dir)
                xtrain,xtest,ytrain,ytest,strain,stest= train_test_split(data,labels,scores,test_size=0.2,random_state=42)
                xtrain,min_X, max_X,min_Y,max_Y, min_Z,max_Z= normalize_skeletons(xtrain)
                train_set = Kimore(xtrain,ytrain,strain)
                train_loader = DataLoader(train_set,batch_size=16,shuffle =True)
                xtest,_,_,_,_,_,_= normalize_skeletons(xtest,min_X, max_X,min_Y,max_Y, min_Z,max_Z)
                test_set = Kimore(xtest,ytest,stest)
                test_loader = DataLoader(test_set,batch_size=16,shuffle=False)
            #------------------------Initialize the generative model
                generator = ACVAE(output_directory=output_directory_run,
                epochs=args.epochs,
                device=args.device)
                generator.train_function(train_loader,device=args.device)
                generator.visualize_latent_space(train_loader,device=args.device)
               
            elif args.generative_model == 'VAE':
                data,labels,scores = load_class(args.class_index,root_dir=dataset_dir)
                xtrain,xtest,ytrain,ytest,strain,stest= train_test_split(data,labels,scores,test_size=0.2,random_state=42)
                xtrain,min_X, max_X,min_Y,max_Y, min_Z,max_Z= normalize_skeletons(xtrain)
                train_set = Kimore(xtrain,ytrain,strain)
                train_loader = DataLoader(train_set,batch_size=16,shuffle =True)
                xtest,min_X, max_X,min_Y,max_Y, min_Z,max_Z= normalize_skeletons(xtest,min_X, max_X,min_Y,max_Y, min_Z,max_Z)
                test_set = Kimore(xtest,ytest,stest)
                test_loader = DataLoader(test_set,batch_size=16,shuffle=False)


                generator = VAE(output_directory=output_directory_skeletons,epochs=args.epochs,device=args.device)
                generator.train_function(dataloader=train_loader,device=args.device)
              