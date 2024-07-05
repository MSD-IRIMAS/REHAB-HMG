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
from model.regressor import REG
import torch
from utils.normalize import normalize_skeletons

def get_args():
    parser = argparse.ArgumentParser(
    description="training regression model to predict generated data score")

    parser.add_argument(
        '--regression_model',
        help = 'Choose a regression model',
        type=str,
        choices = ['STGCN','REG'],
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
    output_directory_reg_models = output_directory_results + 'regression_models/'
    create_directory(output_directory_reg_models)
    output_directory_regressor = output_directory_reg_models + args.regression_model + '/'
    create_directory(output_directory_regressor)


    dataset_dir = 'data/' + args.dataset + '/'
    data,labels,scores = load_class(args.class_index,root_dir=dataset_dir)
    xtrain,xtest,ytrain,ytest,strain,stest= train_test_split(data,labels,scores,test_size=0.2,random_state=42)
    xtrain,min_X, max_X,min_Y,max_Y, min_Z,max_Z = normalize_skeletons(xtrain)
    train_set = Kimore(xtrain,ytrain,strain)
    train_loader = DataLoader(train_set,batch_size=16,shuffle =True)
    xtest,_,_,_,_,_,_= normalize_skeletons(xtest,min_X, max_X,min_Y,max_Y, min_Z,max_Z)
    test_set = Kimore(xtest,ytest,stest)
    test_loader = DataLoader(test_set,batch_size=16,shuffle=False)  


    if args.data_split == 'all':
        for _run in range(args.runs):
            output_directory_run = output_directory_regressor + 'run_' + str(_run) + '/'
            create_directory(output_directory_run)

            output_directory_class = output_directory_run + 'class_' + str(args.class_index) + '/'
            create_directory(output_directory_class)
            

            if args.regression_model == 'STGCN':
                model = STGCN(
                    output_directory=output_directory_class,
                    epochs=args.epochs,
                    device=args.device,
                    edge_importance_weighting=True)
               
                model.train_stgcn(device=args.device,train_loader=train_loader,test_loader=test_loader)
                model.predict_scores(test_loader,args.device)
                model.plot_train_scores(device= args.device,train_loader=train_loader)

            elif args.regression_model == 'REG':
                model = REG(
                    output_directory=output_directory_class,
                    epochs=args.epochs,
                    device=args.device,
                   
                )
                model.train_fun(device=args.device,train_loader=train_loader,test_loader=test_loader)
                model.predict_scores(test_loader,args.device)
                model.plot_train_scores(device= args.device,train_loader=train_loader)
    
    elif args.data_split == 'split':
        for _run in range(args.runs):
            output_directory_run = output_directory_regressor + 'run_' + str(_run) + '/'
            create_directory(output_directory_run)

            output_directory_class = output_directory_run + 'class_' + str(args.class_index) + '/'
            create_directory(output_directory_class)
            

            if args.regression_model == 'STGCN':
                model = STGCN(
                    output_directory=output_directory_class,
                    epochs=args.epochs,
                    device=args.device,
                    edge_importance_weighting=True)
               
                model.train_stgcn(device=args.device,train_loader=train_loader,test_loader=test_loader)
                model.predict_scores(test_loader,args.device)
                model.plot_train_scores(device= args.device,train_loader=train_loader)

            elif args.regression_model == 'REG':
                model = REG(
                    output_directory=output_directory_class,
                    epochs=args.epochs,
                    device=args.device,
                    
                )
                model.train_fun(device=args.device,train_loader=train_loader,test_loader=test_loader)
                # model.predict_scores(test_loader,args.device)
                # model.plot_train_scores(device= args.device,train_loader=train_loader)
