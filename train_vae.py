def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import pandas as pd
import numpy as np
import argparse
from utils.visualize import create_directory
from dataset.dataset import Kimore, load_data
from sklearn.model_selection import train_test_split
from model.cvae import CVAE
from model.cvae_label import CVAEL
from torch.utils.data import DataLoader,Subset
import torch

def get_args():
    parser = argparse.ArgumentParser(
    description="Choose which samples to train the GRU classifier on with the type of split.")

    parser.add_argument(
        '--generative-model',
        help="Which generative model to use .",
        type=str,
        choices=['CVAE','CVAEL'],
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

    parser.add_argument(
        '--data_split',
        help="choose wether split the data or use it all",
        type=str,
        choices=['all', 'split'],
        default='split'
    )
    parser.add_argument(
        '--class_index',
        help="which class to generate from",
        type=int,
        default=0
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
    train_indices, test_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    dataloader = DataLoader(dataset,batch_size=16,shuffle = True)

    if args.data_split == 'all':
        for _run in range(args.runs):

                output_directory_run = output_directory_results + 'run_' + str(_run) + '/'
                create_directory(output_directory_run)
                output_directory_skeletons = output_directory_run + 'generated_samples/'
                create_directory(output_directory_skeletons)

                output_directory_skeletons_class = output_directory_skeletons + 'class_' + str(args.class_index) + '/'
                create_directory(output_directory_skeletons_class)

                if args.generative_model == 'CVAE':
                    generator = CVAE(output_directory=output_directory_run,
                    epochs=args.epochs,
                    device=args.device,
                    w_rec=args.weight_rec,
                    w_kl=args.weight_kl)
                    generator.train_function(dataloader,device=args.device)
                    generator.visualize_latent_space(dataloader,device=args.device)
                    generator.generate_samples_from_prior(device = args.device,class_index=args.class_index,gif_directory=output_directory_skeletons_class)  
    elif args.data_split == 'split':
        for _run in range(args.runs):

                output_directory_run = output_directory_results + 'run_' + str(_run) + '/'
                create_directory(output_directory_run)
                output_directory_skeletons = output_directory_run + 'generated_samples/'
                create_directory(output_directory_skeletons)

                output_directory_skeletons_class = output_directory_skeletons + 'class_' + str(args.class_index) + '/'
                create_directory(output_directory_skeletons_class)

                if args.generative_model == 'CVAE':
                    generator = CVAE(output_directory=output_directory_run,
                    epochs=args.epochs,
                    device=args.device,
                    w_rec=args.weight_rec,
                    w_kl=args.weight_kl)
                    # generator.train_function(train_loader,device=args.device)
                    # generator.visualize_latent_space(train_loader,device=args.device)
                    generator.generate_samples_from_prior(device = args.device,class_index=args.class_index,gif_directory=output_directory_skeletons_class)  


