import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Suppress warnings

import pandas as pd
import numpy as np
import argparse
import os
import sys
sys.path.append('/home/hferrar/HMG/utils')
from utils.visualize import create_directory
from dataset.dataset import Kimore, load_data
from sklearn.model_selection import train_test_split
from model.CVAE import CVAE
from torch.utils.data import DataLoader, TensorDataset
import torch

def get_args():
    parser = argparse.ArgumentParser(
        description="Choose which samples to train the VAE on with the type of split.")

    parser.add_argument(
        '--dataset',
        type=str, 
        default='Kimore',
        help="Which dataset to use.")
    parser.add_argument(
        '--output-directory', 
        type=str, 
        default='results/')
    parser.add_argument(
        '--runs', 
        type=int, 
        default=1, 
        help="How many times you want to run the model")
    parser.add_argument(
        '--weight-rec', 
        type=float, 
        default=0.99, 
        help="Weight for the reconstruction loss.")
    parser.add_argument(
        '--weight-kl', 
        type=float, 
        default=1e-3, 
        help="Weight for the KL loss.")
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=2000, 
        help="Number of epochs to train the model.")
    parser.add_argument(
        '--device', 
        type=str, 
        choices=['cpu', 'cuda', 'mps'], 
        default='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    parser.add_argument(
        '--class_index', 
        type=int, 
        default=0, 
        help="Which class to generate from")
    parser.add_argument(
        '--generative-model', 
        type=str, 
        default='CVAE_cross', 
        help="Which generative model to use.")

    args = parser.parse_args()
    return args

def load_fold_data(fold_idx):
    train_data = np.load(f'data/folds/train_data_fold{fold_idx}.npy')
    train_labels = np.load(f'data/folds/train_labels_fold{fold_idx}.npy')
    train_scores = np.load(f'data/folds/train_scores_fold{fold_idx}.npy')
    test_data = np.load(f'data/folds/test_data_fold{fold_idx}.npy')
    test_labels = np.load(f'data/folds/test_labels_fold{fold_idx}.npy')
    test_scores = np.load(f'data/folds/test_scores_fold{fold_idx}.npy')
    return train_data, train_labels, train_scores, test_data, test_labels, test_scores

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

    for _run in range(args.runs):
        output_directory_run = output_directory_results + 'run_' + str(_run) + '/'
        create_directory(output_directory_run)
        output_directory_cross_val = output_directory_run+ 'cross_validation'+'/'
        create_directory(output_directory_cross_val)

       

        results = []

        for fold_idx in range(1, 6):
            print(f'Processing Fold {fold_idx}')
            output_directory_fold = output_directory_cross_val + 'fold_' + str(fold_idx) + '/'
            create_directory(output_directory_fold)
            output_directory_skeletons = output_directory_fold + 'generated_samples/'
            create_directory(output_directory_skeletons)

            output_directory_skeletons_class = output_directory_skeletons + 'class_' + str(args.class_index) + '/'
            create_directory(output_directory_skeletons_class)

            train_data, train_labels, train_scores, test_data, test_labels, test_scores = load_fold_data(fold_idx)
            train_dataset = Kimore(train_data,train_labels,train_scores)
            test_dataset = Kimore(test_data, test_labels, test_scores)
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
            
            if args.generative_model == 'CVAE':
                generator = CVAE(output_directory=output_directory_fold,
                                 epochs=args.epochs,
                                 device=args.device,
                                 w_rec=args.weight_rec,
                                 w_kl=args.weight_kl)
                generator.generate_samples_from_prior(args.device,args.class_index,output_directory_skeletons_class)
                generator.generate_samples_from_posterior(args.device,args.class_index,output_directory_skeletons_class,test_loader)
                

        #         generator.train_function(train_loader, args.device)
        #         test_loss = generator.evaluate_function(test_loader, args.device)
        #         results.append(test_loss)
        #         print(f'Fold {fold_idx} Test Loss: {test_loss}')

        # average_test_loss = np.mean(results)
        # print(f'Average Test Loss: {average_test_loss}')
