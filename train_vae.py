def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import pandas as pd
import numpy as np
import argparse
from utils.visualize import create_directory
from dataset.dataset import Kimore, load_data
from model.CVAEE import CVAEE
from model.CVAE import CVAE
from torch.utils.data import DataLoader
import torch

def get_args():
    parser = argparse.ArgumentParser(
    description="Choose which samples to train the GRU classifier on with the type of split.")

    parser.add_argument(
        '--generative-model',
        help="Which generative model to use .",
        type=str,
        choices=['CVAE','CVAEE'],
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
        '--weight-rec',
        help="Weight for the reconstruction loss.",
        type=float,
        default=0.99
    )

    parser.add_argument(
        '--weight-kl',
        help="Weight for the kl loss.",
        type=float,
        default=1e-3
    )

    parser.add_argument(
        '--epochs',
        help="Number of epochs to train the model.",
        type=int,
        default=1000
    )
    parser.add_argument(
        '--device',
        help="Device to run the training on.",
        type=str,
        choices=['cpu', 'cuda', 'mps'],
        default='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    )

    args = parser.parse_args()

    return args
    
if __name__ == "__main__":

    args = get_args()

    output_directory_results = args.output_directory
    create_directory(output_directory_results)

    output_directory_gen_models = output_directory_results + 'Generative_models/'
    create_directory(output_directory_gen_models)

    output_directory_dataset = output_directory_gen_models + args.dataset + '/'
    create_directory(output_directory_dataset)

    output_directory_generator = output_directory_dataset + args.generative_model + '/'
    create_directory(output_directory_generator)

    output_directory_weights_losses = output_directory_generator + 'Wrec_' + str(args.weight_rec) + '_Wkl_' + str(args.weight_kl) + '/'
    create_directory(output_directory_weights_losses)



    dataset_dir = 'data/' + args.dataset + '/'
    data,labels,scores = load_data(root_dir=dataset_dir)
    dataset = Kimore(data,labels,scores)
    dataloader = DataLoader(dataset,batch_size=16,shuffle = True,drop_last=True)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )

    for _run in range(args.runs):

            output_directory_run = output_directory_results + 'run_' + str(_run) + '/'
            create_directory(output_directory_run)

            if args.generative_model == 'CVAE':
                generator = CVAE(output_directory=output_directory_run,
                epochs=args.epochs,
                device=args.device,
                               
                                
                                w_rec=args.weight_rec,
                                w_kl=args.weight_kl,

                                )

            
                generator.train_function(dataloader,device=args.device)
                generator.visualize_latent_space(dataloader,device=args.device)

#SEPARATE THE GENERATION PROCESS OF THE TRAINING      
# SAVE THE EONCODER AND DECODER SEPARATLY 
                generator.generate_samples(device = args.device,class_index=0)      
                generator.generate_samples(device=args.device,class_index=1)
                generator.generate_samples(device=args.device,class_index=2)
                generator.generate_samples(device=args.device,class_index=3)
                generator.generate_samples(device=args.device,class_index=4) 
            elif args.generative_model == 'CVAEE':
                generator = CVAEE(output_directory=output_directory_run,
                epochs=args.epochs,
                device=args.device,
                w_rec=args.weight_rec,
                w_kl=args.weight_kl,

                                )

            
                generator.train_function(dataloader,device=args.device)
                generator.visualize_latent_space(dataloader,device=args.device)

#SEPARATE THE GENERATION PROCESS OF THE TRAINING      
# SAVE THE EONCODER AND DECODER SEPARATLY 
  