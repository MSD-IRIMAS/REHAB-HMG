def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import pandas as pd
import numpy as np
import argparse
from utils.visualize import create_directory
from dataset.dataset import Kimore, load_data
from model.CVAE import CVAE
from model.cvae2 import CVA
from torch.utils.data import DataLoader
import torch

def get_args():
    parser = argparse.ArgumentParser(
    description="Choose which samples to train the GRU classifier on with the type of split.")

    parser.add_argument(
        '--regression-model',
        help="regression model to use .",
        type=str,
        choices=['Regressor'],
        default='Regressor',
    )

    parser.add_argument(
        '--dataset',
        help="Which dataset to use.",
        type=str,
        default='Kimore'
    )
    parser.add_argument(
        '--class',
        help="Which class to use.",
        type=int,
        default=0
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

    output_directory_reg_models = output_directory_results + 'Regression_models/'
    create_directory(output_directory_reg_models)

    output_directory_dataset = output_directory_reg_models + args.dataset + '/'
    create_directory(output_directory_dataset)

    output_directory_generator = output_directory_dataset + args.regression_model + '/'
    create_directory(output_directory_generator)

    dataset_dir = 'data/' + args.dataset + '/'
    data,labels,scores = load_class(index_class,root_dir=dataset_dir)
    dataset = Kimore(data,labels,scores)
    dataloader = DataLoader(dataset,batch_size=8,shuffle = True)


    for _run in range(args.runs):
            output_directory_run = output_directory_results + 'run_' + str(_run) + '/'
            create_directory(output_directory_run)
            if args.regression_model == 'Regressor':
                regressor = Regressor(output_directory=output_directory_run,
                epochs=args.epochs,
                device=args.device)
                regressor.train(dataloader,device=args.device)
           
    
          
