def warn(*args, **kwargs):
    pass
import os
import torch
import warnings
import argparse
import numpy as np
import pandas as pd
warnings.warn = warn
from model.cvae import CVAE
from model.stgcn import STGCN
from utils.visualize import create_directory
from dataset.dataset import Kimore, load_class
from torch.utils.data import DataLoader,Subset
from utils.normalize import normalize_skeletons
from utils.utils import get_weights_loss,get_dirs
from sklearn.model_selection import train_test_split
from utils.metrics import FID,Coverage,MMS,FeatureExtractor,Density, APD


def get_args():
    parser = argparse.ArgumentParser(
    description="training STGCN to predict generated data score")

    parser.add_argument(
        '--regression_models',
        type=str,
        choices=['STGCN','REG'],
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
        default=0
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
    parser.add_argument(
        '--wrec',
        help="Weight for the reconstruction loss.",
        type=float,
        default=0.99
    )

    parser.add_argument(
        '--wkl',
        help="Weight for the kl loss.",
        type=float,
        default=1e-3
    )

    args = parser.parse_args()

    return args
    
if __name__ == "__main__":

    args = get_args()

    output_directory_results = args.output_directory + 'regression_models/'
    create_directory(output_directory_results)
   
    dataset_dir = 'data/' + args.dataset + '/'
    data,labels,scores = load_class(args.class_index,root_dir=dataset_dir)
   
    xtrain,xtest,ytrain,ytest,strain,stest= train_test_split(data,labels,scores,test_size=0.5,random_state=42)
    xtrain,min_X, max_X,min_Y,max_Y, min_Z,max_Z= normalize_skeletons(xtrain)
    train_set = Kimore(xtrain,ytrain,strain)
    train_loader = DataLoader(train_set,batch_size=16,shuffle =True)
    xtest,_,_,_,_,_,_= normalize_skeletons(xtest,min_X, max_X,min_Y,max_Y, min_Z,max_Z)
    test_set = Kimore(xtest,ytest,stest)
    test_loader = DataLoader(test_set,batch_size=16,shuffle=False)

    output_directory_run = args.output_directory + 'run_' + str(args.runs) + '/'
    create_directory(output_directory_run)
    output_directory_skeletons_class = output_directory_run + 'class_' + str(args.class_index) + '/'
    create_directory(output_directory_skeletons_class)




    if args.regression_models == 'STGCN':
        stgcn_directory = output_directory_results + 'STGCN/'
        stgcn_run_dir_class = stgcn_directory + 'run_' + str(args.runs) +'/class_'+str(args.class_index)
        stgcn_run_dir = stgcn_run_dir_class + '/best_stgcn.pth'
        feature_extractor_stgcn = FeatureExtractor(model_name=args.regression_models, model_path=stgcn_run_dir, device=args.device)
        fid_calculator_stgcn = FID(feature_extractor_stgcn)
        coverage_calculator_stgcn = Coverage(feature_extractor_stgcn)
        mms_calculator_stgcn = MMS(feature_extractor_stgcn)
        density_calculator_stgcn = Density(feature_extractor_stgcn)
        apd_calculator_stgcn = APD(feature_extractor_stgcn)
        print(feature_extractor_stgcn)
        if args.on == 'real':
            if os.path.exists(stgcn_directory + 'metrics_results.csv'):
                df = pd.read_csv(stgcn_directory + 'metrics_results.csv')
            else:
                df = pd.DataFrame(columns=['class','FID','COV','MMS','Density'])
            #FID metric
            fid_score = fid_calculator_stgcn.calculate_fid(train_loader, test_loader)
            fid_mean = np.mean(fid_score)
           
            #COV metric
            cov_score = coverage_calculator_stgcn.calculate(test_loader, train_loader)
            cov_mean = np.mean(cov_score)

            #MMS metric
            mms_score = mms_calculator_stgcn.calculate_mms(test_loader, train_loader)
            mms_mean = np.mean(mms_score)

            #Density metric
            mms_score = density_calculator_stgcn.calculate_density(test_loader, train_loader)
            density_mean = np.mean(mms_score)
            #APD metric
            apd_score = apd_calculator_stgcn.calculate_apd(test_loader)
            apd_mean = np.mean(apd_score)
          
            if args.class_index in df['class'].values:
                df.loc[df['class'] == args.class_index, 'FID'] = fid_mean
                df.loc[df['class'] == args.class_index, 'COV'] = cov_mean
                df.loc[df['class'] == args.class_index, 'MMS'] = mms_mean
                df.loc[df['class'] == args.class_index, 'Density'] = density_mean
                df.loc[df['class'] == args.class_index, 'APD'] = apd_mean
            else:
                new_row = pd.DataFrame([{'class': args.class_index, 'FID': fid_mean, 'COV': cov_mean, 'MMS': mms_mean,'Density':density_mean,'APD':apd_mean}])
                df = pd.concat([df, new_row], ignore_index=True)
                      
            df.to_csv(stgcn_directory + 'metrics_results.csv', index=False)

            
        elif args.on == 'generated':
            weights_loss = {
                'wrec' : args.wrec,
                'wkl' : args.wkl,}
            if os.path.exists(stgcn_directory + 'metrics_results_generated.csv'):
                df = pd.read_csv(stgcn_directory + 'metrics_results_generated.csv')
            else:
                df = pd.DataFrame(columns=['class','FID','COV','MMS'])
            
            #generate samples 
            generator = CVAE(output_directory=output_directory_skeletons_class,
                    device=args.device,
                    w_rec=weights_loss['wrec'],
                    w_kl=weights_loss['wkl'])
            generated_samples,gen_scores = generator.generate_samples_from_prior(device = args.device,class_index=args.class_index,gif_directory=stgcn_run_dir_class,dataloader=test_loader)
            gen_set = Kimore(generated_samples,labels,gen_scores)
            gen_loader = DataLoader(gen_set,batch_size=16,shuffle =True)
            
            #FID metric
            fid_score = fid_calculator_stgcn.calculate_fid(gen_loader, test_loader)
            fid_mean = np.mean(fid_score)
            print('----------------------------------------------fid done')
            if fid_mean < 0 :
                np.save('negative_generated_samples.npy',generated_samples)

            #COV metric
            cov_score = coverage_calculator_stgcn.calculate(xgenerated_loader=gen_loader, xreal_loader=test_loader)
            cov_mean = np.mean(cov_score)
            print('----------------------------------------------cov done')


            #MMS metric
            mms_score = mms_calculator_stgcn.calculate_mms(generated_data=gen_loader, real_data=test_loader)
            mms_mean = np.mean(mms_score)

            #Density metric
            mms_score = density_calculator_stgcn.calculate_density(gen_loader, test_loader)
            density_mean = np.mean(mms_score)

            apd_score = apd_calculator_stgcn.calculate_apd(gen_loader)
            apd_mean = np.mean(apd_score)
          
            if args.class_index in df['class'].values:
                df.loc[df['class'] == args.class_index, 'FID'] = fid_mean
                df.loc[df['class'] == args.class_index, 'COV'] = cov_mean
                df.loc[df['class'] == args.class_index, 'MMS'] = mms_mean
                df.loc[df['class'] == args.class_index, 'APD'] = apd_mean
            else:
                new_row = pd.DataFrame([{'class': args.class_index, 'FID': fid_mean, 'COV': cov_mean, 'MMS': mms_mean,'Density':density_mean,'APD':apd_mean}])
                df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(stgcn_directory + 'metrics_results_generated.csv', index=False)




    elif args.regression_models == 'REG':
        regressor_directory = output_directory_results + 'REG/'
        reg_run_dir_class = regressor_directory + 'run_' + str(args.runs) +'/class_'+str(args.class_index)
        reg_run_dir = reg_run_dir_class + '/best_regressor.pth'
        
        feature_extractor_reg = FeatureExtractor(model_name=args.regression_models, model_path=reg_run_dir, device=args.device)

        fid_calculator = FID(feature_extractor_reg)
        coverage_calculator = Coverage(feature_extractor_reg)
        mms_calculator= MMS(feature_extractor_reg)
        density_calculator_reg = Density(feature_extractor_reg)
        apd_calculator_reg = APD(feature_extractor_reg)


        if args.on == 'real':
            if os.path.exists(regressor_directory + 'metrics_results.csv'):
                df = pd.read_csv(regressor_directory + 'metrics_results.csv')
            else:
                df = pd.DataFrame(columns=['class','FID','COV','MMS'])
            #FID metric
            fid_score = fid_calculator.calculate_fid(train_loader, test_loader)
            fid_mean = np.mean(fid_score)
            print('----------------------------------------------fid done')

            #COV metric
            cov_score = coverage_calculator.calculate(train_loader, test_loader)
            cov_mean = np.mean(cov_score)
            print('----------------------------------------------cov done')

            #MMS metric
            mms_score = mms_calculator.calculate_mms(train_loader, test_loader)
            mms_mean = np.mean(mms_score)
            #Density metric
            density_score = density_calculator_reg.calculate_density(xreal_loader=train_loader,xgenerated_loader= test_loader)
            density_mean = np.mean(density_score)


            apd_score = apd_calculator_reg.calculate_apd(gen_loader)
            apd_mean = np.mean(apd_score)
            if args.class_index in df['class'].values:
                df.loc[df['class'] == args.class_index, 'FID'] = fid_mean
                df.loc[df['class'] == args.class_index, 'COV'] = cov_mean
                df.loc[df['class'] == args.class_index, 'MMS'] = mms_mean
                df.loc[df['class'] == args.class_index, 'APD'] = apd_mean
            else:
                new_row = pd.DataFrame([{'class': args.class_index, 'FID': fid_mean, 'COV': cov_mean, 'MMS': mms_mean,'Density':density_mean,'APD':apd_mean}])
                df = pd.concat([df, new_row], ignore_index=True)
                      
            df.to_csv(regressor_directory + 'metrics_results.csv', index=False)
            
        elif args.on == 'generated':
            weights_loss = {
                'wrec' : args.wrec,
                'wkl' : args.wkl,}
            if os.path.exists(regressor_directory + 'metrics_results_generated.csv'):
                df = pd.read_csv(regressor_directory + 'metrics_results_generated.csv')
            else:
                df = pd.DataFrame(columns=['class','FID','COV','MMS'])
            
            #generate samples 
            generator = CVAE(output_directory=output_directory_skeletons_class,
                    device=args.device,
                    w_rec=weights_loss['wrec'],
                    w_kl=weights_loss['wkl'])
            generated_samples,gen_scores = generator.generate_samples_from_prior(device = args.device,class_index=args.class_index,gif_directory=reg_run_dir_class,dataloader=test_loader)
            gen_set = Kimore(generated_samples,labels,gen_scores)
            gen_loader = DataLoader(gen_set,batch_size=16,shuffle =True)
            
            #FID metric
            fid_score = fid_calculator.calculate_fid(gen_loader, test_loader)
            fid_mean = np.mean(fid_score)
            print('----------------------------------------------fid done')
            #COV metric
            cov_score = coverage_calculator.calculate(gen_loader, test_loader)
            cov_mean = np.mean(cov_score)
            print('----------------------------------------------cov done')
            #MMS metric
            mms_score = mms_calculator.calculate_mms(gen_loader, test_loader)
            mms_mean = np.mean(mms_score)
            #Density metric
            density_score = density_calculator_reg.calculate_density(gen_loader, test_loader)
            density_mean = np.mean(density_score)
            apd_score = apd_calculator_reg.calculate_apd(gen_loader)
            apd_mean = np.mean(apd_score)

            if args.class_index in df['class'].values:
                df.loc[df['class'] == args.class_index, 'FID'] = fid_mean
                df.loc[df['class'] == args.class_index, 'COV'] = cov_mean
                df.loc[df['class'] == args.class_index, 'MMS'] = mms_mean
                df.loc[df['class'] == args.class_index, 'APD'] = apd_mean
            else:
                new_row = pd.DataFrame([{'class': args.class_index, 'FID': fid_mean, 'COV': cov_mean, 'MMS': mms_mean,'Density':density_mean,'APD':apd_mean}])
                df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(regressor_directory + 'metrics_results_generated.csv', index=False)


           
        


