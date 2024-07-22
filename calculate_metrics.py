def warn(*args, **kwargs):
    pass
import os
import torch
import warnings
import argparse
import random
import numpy as np
import pandas as pd
warnings.warn = warn
from model.svae import SVAE
from model.AS_CVAE import ASCVAE
from model.vae import VAE
from model.vae_label import CVAEL
from model.stgcn import STGCN
from utils.visualize import create_directory
from dataset.dataset import Kimore, load_class,load_data
from torch.utils.data import DataLoader,Subset
from utils.normalize import normalize_skeletons
from utils.utils import get_weights_loss,get_dirs,noise_data
from sklearn.model_selection import train_test_split
from utils.metrics import FID,Coverage,MMS,FeatureExtractor,Density, APD



random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True  # If using CUDA
torch.backends.cudnn.benchmark 



def get_args():
    parser = argparse.ArgumentParser(
    description="which regressor to predict generated data score")

    parser.add_argument(
        '--regression_models',
        type=str,
        choices=['STGCN','REG'],
        default='STGCN',
    )
    parser.add_argument(
        '--generative_model',
        type=str,
        choices=['CVAEL','ASCVAE'],
        default='ASCVAE',
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
        default=5
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
        default=0.999
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
    

    output_directory_gen_models = args.output_directory + 'Generative_models/'
    create_directory(output_directory_gen_models)

    if args.generative_model =='ASCVAE':
        output_directory_dataset = output_directory_gen_models + 'Score+Action_conditioned' + '/'
       
    elif args.generative_model == 'CVAEL':
        output_directory_dataset = output_directory_gen_models + 'Action_conditioned' + '/'
    create_directory(output_directory_dataset)

    output_directory_generator = output_directory_dataset + args.generative_model + '/'
    create_directory(output_directory_generator)

    output_directory_weights_losses = output_directory_generator + 'Wrec_' + str(args.wrec) + '_Wkl_' + str(args.wkl) + '/'
    create_directory(output_directory_weights_losses)
    
    
    for run_ in range(args.runs):
        # for class_index in range(args.class_index):
                # data,labels,scores = load_class(class_index,root_dir=dataset_dir)
            data,labels,scores = load_data(root_dir=dataset_dir)
            xtrain,xtest,ytrain,ytest,strain,stest= train_test_split(data,labels,scores,test_size=0.5,random_state=42)
            xtrain,min_X, max_X,min_Y,max_Y, min_Z,max_Z= normalize_skeletons(xtrain)
            noisy_data= noise_data(xtrain)
            noisy_set = Kimore(noisy_data,ytrain,strain)
            train_set = Kimore(xtrain,ytrain,strain)
            train_loader = DataLoader(train_set,batch_size=10,shuffle =True)
            noisy_loader = DataLoader(noisy_set,batch_size=10,shuffle=True)
            xtest,_,_,_,_,_,_= normalize_skeletons(xtest,min_X, max_X,min_Y,max_Y, min_Z,max_Z)
            test_set = Kimore(xtest,ytest,stest)
            test_loader = DataLoader(test_set,batch_size=10,shuffle=False)

            output_directory_run = output_directory_weights_losses + 'run_' + str(run_) + '/'
            create_directory(output_directory_run)
            output_directory_skeletons = output_directory_run + 'generated_samples/'
            create_directory(output_directory_skeletons)
            # output_directory_skeletons_class = output_directory_run + 'class_' + str(class_index) + '/'
            # create_directory(output_directory_skeletons_class)
            weights_loss = {
                        'wrec' : args.wrec,
                        'wkl' : args.wkl}
         
              
            generator = ASCVAE(output_directory=output_directory_run,
                            device=args.device,
                            w_rec=weights_loss['wrec'],
                            w_kl=weights_loss['wkl'])
            as_generated_samples,train_scores = generator.generate_samples_from_prior(device = args.device,gif_directory=output_directory_skeletons,dataloader=train_loader)
            as_test_generated_samples,test_scores = generator.generate_samples_from_prior(device = args.device,gif_directory=output_directory_skeletons,dataloader=test_loader)

        
        
    
            generator= CVAEL(output_directory=output_directory_run, device=args.device,epochs=2000)
            a_generated_samples = generator.generate_samples_from_prior(device = args.device,gif_directory=output_directory_skeletons,data_loader=train_loader)
            a_test_generated_samples = generator.generate_samples_from_prior(device = args.device,gif_directory=output_directory_skeletons,data_loader=test_loader)
           
                        
                    
            a_gen_set = Kimore(a_generated_samples, ytrain, strain)
            a_gen_loader = DataLoader(gen_set, batch_size=10, shuffle=True)


            a_gen_set = Kimore(a_test_generated_samples, ytest, stest)
            a_test_gen_loader = DataLoader(gen_set, batch_size=10, shuffle=True)


            as_gen_set = Kimore(as_generated_samples, ytrain, strain)
            as_gen_loader = DataLoader(gen_set, batch_size=10, shuffle=True)


            as_gen_set = Kimore(as_test_generated_samples, ytest, stest)
            as_test_gen_loader = DataLoader(gen_set, batch_size=10, shuffle=True)
        




            regressor_directory = output_directory_results + 'REG/'
            reg_run_dir_class = regressor_directory + 'run_' + str(run_) +'/class_'+str(0)
            reg_run_dir = reg_run_dir_class + '/best_regressor.pth'
            
            feature_extractor_reg = FeatureExtractor(model_name=args.regression_models, model_path=reg_run_dir, device=args.device)
            fid_calculator = FID(feature_extractor_reg)
            coverage_calculator = Coverage(feature_extractor_reg)
            mms_calculator= MMS(feature_extractor_reg)
            density_calculator_reg = Density(feature_extractor_reg)
            apd_calculator_reg = APD(feature_extractor_reg)

            if os.path.exists(output_directory_skeletons_class + f'/{args.regression_models}_train_vs_noisy.csv'):
                    df = pd.read_csv(output_directory_skeletons_class + f'/{args.regression_models}_train_vs_noisy.csv')
            else:
                    df = pd.DataFrame(columns=['ON','FID','COV','MMS','Density','APD'])

            fid_score = fid_calculator.calculate_fid(train_loader, noisy_loader)
            fid_mean = np.mean(fid_score)
            
            #COV metric
            cov_score = coverage_calculator.calculate(train_loader, noisy_loader)
            cov_mean = np.mean(cov_score)

            #MMS metric
            mms_score = mms_calculator.calculate_mms(train_loader, noisy_loader)
            mms_mean = np.mean(mms_score)

            #Density metric
            mms_score = density_calculator_reg.calculate_density(train_loader, noisy_loader)
            density_mean = np.mean(mms_score)
            #APD metric
            apd_score = apd_calculator_reg.calculate_apd(train_loader)
            apd_mean = np.mean(apd_score)



            if 'Real' in df['ON'].values:
                    df.loc[df['ON'] == 'Real', 'FID'] = fid_mean
                    df.loc[df['ON'] == 'Real', 'COV'] = cov_mean
                    df.loc[df['ON'] == 'Real', 'MMS'] = mms_mean
                    df.loc[df['ON'] == 'Real', 'Density'] = density_mean
                    df.loc[df['ON'] == 'Real', 'APD'] = apd_mean
            else:
                    new_row = pd.DataFrame([{'ON':'Real', 'FID': fid_mean, 'COV': cov_mean, 'MMS': mms_mean,'Density':density_mean,'APD':apd_mean}])
                    df = pd.concat([df, new_row], ignore_index=True)
                
            fid_score = fid_calculator.calculate_fid(as_gen_loader, train_loader)
            fid_mean = np.mean(fid_score)
            
            #COV metric
            cov_score = coverage_calculator.calculate(xgenerated_loader=as_gen_loader, xreal_loader=train_loader)
            cov_mean = np.mean(cov_score)

            #MMS metric
            mms_score = mms_calculator.calculate_mms(generated_data=as_gen_loader, real_data=train_loader)
            mms_mean = np.mean(mms_score)

            #Density metric
            mms_score = density_calculator_reg.calculate_density(as_gen_loader, train_loader)
            density_mean = np.mean(mms_score)

            apd_score = apd_calculator_reg.calculate_apd(as_gen_loader)
            apd_mean = np.mean(apd_score)
            if 'ASCVAE' in df['ON'].values:
                    df.loc[df['ON'] == 'ASCVAE', 'FID'] = fid_mean
                    df.loc[df['ON'] == 'ASCVAE', 'COV'] = cov_mean
                    df.loc[df['ON'] == 'ASCVAE', 'MMS'] = mms_mean
                    df.loc[df['ON'] == 'ASCVAE', 'Density'] = density_mean
                    df.loc[df['ON'] == 'ASCVAE', 'APD'] = apd_mean
            else:
                    new_row = pd.DataFrame([{'ON':'ASCVAE', 'FID': fid_mean, 'COV': cov_mean, 'MMS': mms_mean,'Density':density_mean,'APD':apd_mean}])
                    df = pd.concat([df, new_row], ignore_index=True)
            
            fid_score = fid_calculator.calculate_fid(a_gen_loader, train_loader)
            fid_mean = np.mean(fid_score)
            
            #COV metric
            cov_score = coverage_calculator.calculate(xgenerated_loader=a_gen_loader, xreal_loader=train_loader)
            cov_mean = np.mean(cov_score)

            #MMS metric
            mms_score = mms_calculator.calculate_mms(generated_data=a_gen_loader, real_data=train_loader)
            mms_mean = np.mean(mms_score)

            #Density metric
            mms_score = density_calculator_reg.calculate_density(a_gen_loader, train_loader)
            density_mean = np.mean(mms_score)

            apd_score = apd_calculator_reg.calculate_apd(a_gen_loader)
            apd_mean = np.mean(apd_score)
            if 'ACVAE' in df['ON'].values:
                    df.loc[df['ON'] == 'ACVAE', 'FID'] = fid_mean
                    df.loc[df['ON'] == 'ACVAE', 'COV'] = cov_mean
                    df.loc[df['ON'] == 'ACVAE', 'MMS'] = mms_mean
                    df.loc[df['ON'] == 'ACVAE', 'Density'] = density_mean
                    df.loc[df['ON'] == 'ACVAE', 'APD'] = apd_mean
            else:
                    new_row = pd.DataFrame([{'ON':'ACVAE', 'FID': fid_mean, 'COV': cov_mean, 'MMS': mms_mean,'Density':density_mean,'APD':apd_mean}])
                    df = pd.concat([df, new_row], ignore_index=True)
        
            df.to_csv(output_directory_skeletons_class + f'/{args.regression_models}_train_vs_noisy.csv', index=False)
            print('')









    # #########################################################################################################################################################################""

    #         if os.path.exists(output_directory_skeletons_class + f'/{args.regression_models}_train_vs_test{class_index}.csv'):
    #                 df = pd.read_csv(output_directory_skeletons_class + f'/{args.regression_models}_train_vs_test{class_index}.csv')
    #         else:
    #                 df = pd.DataFrame(columns=['ON','FID','COV','MMS','Density','APD'])

    #         fid_score = fid_calculator.calculate_fid(train_loader, test_loader)
    #         fid_mean = np.mean(fid_score)
            
    #         #COV metric
    #         cov_score = coverage_calculator.calculate(train_loader, test_loader)
    #         cov_mean = np.mean(cov_score)

    #         #MMS metric
    #         mms_score = mms_calculator.calculate_mms(train_loader, test_loader)
    #         mms_mean = np.mean(mms_score)

    #         #Density metric
    #         density_core = density_calculator_reg.calculate_density(train_loader, test_loader)
    #         density_mean = np.mean(density_core)
    #         #APD metric
    #         apd_score = apd_calculator_reg.calculate_apd(train_loader)
    #         apd_mean = np.mean(apd_score)

    #         if 'train_vs_test' in df['ON'].values:
    #                 df.loc[df['ON'] == 'train_vs_test', 'FID'] = fid_mean
    #                 df.loc[df['ON'] == 'train_vs_test', 'COV'] = cov_mean
    #                 df.loc[df['ON'] == 'train_vs_test', 'MMS'] = mms_mean
    #                 df.loc[df['ON'] == 'train_vs_test', 'Density'] = density_mean
    #                 df.loc[df['ON'] == 'train_vs_test', 'APD'] = apd_mean
    #         else:
    #                 new_row = pd.DataFrame([{'ON':'train_vs_test', 'FID': fid_mean, 'COV': cov_mean, 'MMS': mms_mean,'Density':density_mean,'APD':apd_mean}])
    #                 df = pd.concat([df, new_row], ignore_index=True)  
                
    
            

    #         #FID metric
    #         fid_score = fid_calculator.calculate_fid(test_gen_loader, train_loader)
    #         fid_mean = np.mean(fid_score)
    #         print('----------------------------------------------fid done')
    #         if fid_mean < 0 :
    #             np.save('negative_generated_samples.npy',generated_samples)

    #         #COV metric
    #         cov_score = coverage_calculator.calculate(xgenerated_loader=test_gen_loader, xreal_loader=train_loader)
    #         cov_mean = np.mean(cov_score)
    #         print('----------------------------------------------cov done')

    #         #MMS metric
    #         mms_score = mms_calculator.calculate_mms(generated_data=test_gen_loader, real_data=train_loader)
    #         mms_mean = np.mean(mms_score)

    #         #Density metric
    #         density_score = density_calculator_reg.calculate_density(test_gen_loader, train_loader)
    #         density_mean = np.mean(density_score)

    #         apd_score = apd_calculator_reg.calculate_apd(test_gen_loader)
    #         apd_mean = np.mean(apd_score)
    #         if 'trainVStest_gen' in df['ON'].values:
    #                 df.loc[df['ON'] == 'trainVStest_gen', 'FID'] = fid_mean
    #                 df.loc[df['ON'] == 'trainVStest_gen', 'COV'] = cov_mean
    #                 df.loc[df['ON'] == 'trainVStest_gen', 'MMS'] = mms_mean
    #                 df.loc[df['ON'] == 'trainVStest_gen', 'Density'] = density_mean
    #                 df.loc[df['ON'] == 'trainVStest_gen', 'APD'] = apd_mean
    #         else:
    #                 new_row = pd.DataFrame([{'ON':'trainVStest_gen', 'FID': fid_mean, 'COV': cov_mean, 'MMS': mms_mean,'Density':density_mean,'APD':apd_mean}])
    #                 df = pd.concat([df, new_row], ignore_index=True)  
        
        
    #         df.to_csv(output_directory_skeletons_class + f'/{args.regression_models}_train_vs_test{class_index}.csv', index=False)






