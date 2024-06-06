def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import pandas as pd
import numpy as np
import argparse
from utils.visualize import create_directory
from dataset.dataset import Kimore, load_data,load_class
from model.CVAEE import CVAEE
from torch.utils.data import DataLoader
import torch


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset',
        help="Which dataset to use.",
        type=str,
        default='Kimore'
    )

    parser.add_argument(
        '--generative-model',
        help="Which model to use.",
        type=str,
        choices=['CVAE','CVAEE'],
        default='CVAE'
    )

    parser.add_argument(
        '--run',
        type=int,
        default=0
    )


    parser.add_argument(
        '--weight-rec',
        type=float,
        default=0.9
    )

    parser.add_argument(
        '--weight-kl',
        type=float,
        default=1e-3
    )
    parser.add_argument(
        '--class_index',
        help ='classes from 0 to 4',
        type=int,
        default=0
    )

    parser.add_argument(
        '--output-directory',
        type=str,
        default='results/'
    )
    parser.add_argument(
        '--device',
        help="Device to run the training on.",
        type=str,
        choices=['cpu', 'cuda', 'mps'],
        default='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    )
    parser.add_argument(
        '--epochs',
        help="Number of epochs to train the model.",
        type=int,
        default=1000
    )

    args = parser.parse_args()

    return args

if __name__ == "__main__":

    args = get_args()




    output_directory_results = args.output_directory
    output_directory_gen_models = output_directory_results + 'Generative_models/'

    output_directory_gen_model = output_directory_gen_models + args.generative_model + '/'

    output_directory_gen_weight = output_directory_gen_model + 'Wrec_' + str(args.weight_rec) + '_Wkl_' + str(args.weight_kl)  + '/'

    weights_loss = {
        'Wrec' : args.weight_rec,
        'Wkl' : args.weight_kl,

    }


    output_directory_run = output_directory_results + 'run_' + str(args.run) + '/'

    output_directory_skeletons = output_directory_run + 'generated_samples/'
    create_directory(output_directory_skeletons)

    output_directory_skeletons_class = output_directory_skeletons + 'class_' + str(args.class_index) + '/'
    create_directory(output_directory_skeletons_class)
    dataset_dir = 'data/' + args.dataset + '/'
    data,labels,scores = load_class(class_index =0,root_dir=dataset_dir)
    dataset = Kimore(data,labels,scores)
    dataloader = DataLoader(dataset,batch_size=16,shuffle=False)

    generator = CVAEE(output_directory=output_directory_run,
                epochs=args.epochs,
                device=args.device,
                w_rec=weights_loss['Wrec'],
                w_kl=weights_loss['Wkl'])
   

    generator.generate_samples(device = args.device,class_index=args.class_index,gif_directory=output_directory_skeletons_class,dataloader=dataloader)


    ####--------------ADD the regressor