def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import pandas as pd
import numpy as np
import argparse
from utils.visualize import create_directory
from dataset.dataset import Kimore, load_class
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,Subset
from model.stgcn import STGCN
import torch

def get_args():
    parser = argparse.ArgumentParser(
    description="training STGCN to predict generated data score")

    parser.add_argument(
        '--stgcn',
        type=str,
        default='STGCN',
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
        '--data_split',
        help="choose wether split the data or use it all",
        type=str,
        choices=['all', 'split'],
        default='split'
    )
    parser.add_argument(
        '--class_index',
        help="which class to use",
        type=int,
        default=0
    )

    args = parser.parse_args()

    return args
    
if __name__ == "__main__":

    args = get_args()

    output_directory_results = args.output_directory
    create_directory(output_directory_results)

    output_directory_gen_models = output_directory_results + 'stgcn/'
    create_directory(output_directory_gen_models)

    output_directory_dataset = output_directory_gen_models + args.dataset + '/'
    create_directory(output_directory_dataset)

    output_directory_generator = output_directory_dataset + args.stgcn + '/'
    create_directory(output_directory_generator)


    dataset_dir = 'data/' + args.dataset + '/'
    data,labels,scores = load_class(args.class_index,root_dir=dataset_dir)
    dataset = Kimore(data,labels,scores)
    train_indices, test_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)



    if args.data_split == 'all':
        for _run in range(args.runs):

                output_directory_run = output_directory_gen_models + 'run_' + str(_run) + '/'
                create_directory(output_directory_run)
                output_directory_skeletons = output_directory_run + 'generated_samples/'
                create_directory(output_directory_skeletons)

                output_directory_skeletons_class = output_directory_skeletons + 'class_' + str(args.class_index) + '/'
                create_directory(output_directory_skeletons_class)

               
    elif args.data_split == 'split':
        for _run in range(args.runs):

                output_directory_run = output_directory_gen_models + 'run_' + str(_run) + '/'
                create_directory(output_directory_run)
                output_directory_skeletons = output_directory_run + 'generated_samples/'
                create_directory(output_directory_skeletons)

                output_directory_skeletons_class = output_directory_run + 'class_' + str(args.class_index) + '/'
                create_directory(output_directory_skeletons_class)
                model = STGCN(
                    output_directory=output_directory_skeletons_class,
                    epochs=args.epochs,
                    device=args.device,
                    edge_importance_weighting=True)
                model.train_stgcn(device=args.device,train_loader=train_loader,test_loader=test_loader)
                model.predict_scores(test_loader,args.device)

               

