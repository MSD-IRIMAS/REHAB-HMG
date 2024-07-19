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
from dataset.dataset import Kimore, load_class
from torch.utils.data import DataLoader,Subset
from utils.normalize import normalize_skeletons
from utils.utils import get_weights_loss,get_dirs,noise_data
from sklearn.model_selection import train_test_split
from utils.metrics import FID,Coverage,MMS,FeatureExtractor,Density, APD



random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True  # If using CUDA
torch.backends.cudnn.benchmark = False



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
        choices=['SVAE','CVAEL','VAE','ASCVAE'],
        default='SVAE',
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

    if args.generative_model  in ['SVAE','ASCVAE']:
        output_directory_dataset = output_directory_gen_models + 'Score+Action_conditioned' + '/'
       
    elif args.generative_model  in ['VAE','CVAEL']:
        output_directory_dataset = output_directory_gen_models + 'Action_conditioned' + '/'
    create_directory(output_directory_dataset)




    output_directory_generator = output_directory_dataset + args.generative_model + '/'
    create_directory(output_directory_generator)

    output_directory_weights_losses = output_directory_generator + 'Wrec_' + str(args.wrec) + '_Wkl_' + str(args.wkl) + '/'
    create_directory(output_directory_weights_losses)
    
    

    
    
    for run_ in range(args.runs):
           
        for class_index in range(args.class_index):
                data,labels,scores = load_class(class_index,root_dir=dataset_dir)
   
                xtrain,xtest,ytrain,ytest,strain,stest= train_test_split(data,labels,scores,test_size=0.5,random_state=42)
                xtrain,min_X, max_X,min_Y,max_Y, min_Z,max_Z= normalize_skeletons(xtrain)
                noisy_data= noise_data(xtrain)
                noisy_set = Kimore(noisy_data,ytrain,strain)

                train_set = Kimore(xtrain,ytrain,strain)
                train_loader = DataLoader(train_set,batch_size=16,shuffle =False)
                noisy_loader = DataLoader(noisy_set,batch_size=16,shuffle=False)
                xtest,_,_,_,_,_,_= normalize_skeletons(xtest,min_X, max_X,min_Y,max_Y, min_Z,max_Z)
                test_set = Kimore(xtest,ytest,stest)
                test_loader = DataLoader(test_set,batch_size=16,shuffle=False)
                output_directory_run = output_directory_weights_losses + 'run_' + str(run_) + '/'
                create_directory(output_directory_run)
                output_directory_skeletons = output_directory_run + 'generated_samples/'
                create_directory(output_directory_skeletons)

                output_directory_skeletons_class = output_directory_run + 'class_' + str(class_index) + '/'
                create_directory(output_directory_skeletons_class)
                weights_loss = {
                            'wrec' : args.wrec,
                            'wkl' : args.wkl,}
                # Define a function to get the generator based on the model type
                def get_generator(model_type, output_dir, device):
                        if args.generative_model == 'SVAE':
                                generator= SVAE(output_directory=output_dir, device=device, w_rec=weights_loss['wrec'], w_kl=weights_loss['wkl'])
                                generated_samples,gen_scores = generator.generate_samples_from_prior(device = args.device,gif_directory=output_directory_skeletons_class,dataloader=train_loader)
                                test_generated_samples,test_gen_scores = generator.generate_samples_from_prior(device = args.device,gif_directory=output_directory_skeletons_class,dataloader=test_loader)
                                return generated_samples,gen_scores,test_generated_samples,test_gen_scores 


                        elif args.generative_model == 'ASCVAE':
                                generator= ASCVAE(output_directory=output_directory_run, device=device, w_rec=weights_loss['wrec'], w_kl=weights_loss['wkl'])
                                generated_samples,gen_scores = generator.generate_samples_from_prior(device = args.device,gif_directory=output_directory_run,dataloader=train_loader,class_index=class_index)
                                test_generated_samples,test_gen_scores = generator.generate_samples_from_prior(device = args.device,gif_directory=output_directory_run,dataloader=test_loader,class_index=class_index)
                                return generated_samples,gen_scores,test_generated_samples,test_gen_scores 

            
              
                generated_samples, gen_scores, test_generated_samples,test_gen_scores  = get_generator(args.generative_model, output_directory_skeletons_class, args.device)
                        
                        # Create the DataLoader for the generated samples
                gen_set = Kimore(generated_samples, labels, gen_scores)
                gen_loader = DataLoader(gen_set, batch_size=16, shuffle=True)

                gen_set = Kimore(test_generated_samples, labels, test_gen_scores)
                test_gen_loader = DataLoader(gen_set, batch_size=16, shuffle=True)
     
                

                if args.regression_models == 'STGCN':


                        stgcn_directory = output_directory_results + 'STGCN/'
                        stgcn_run_dir_class = stgcn_directory + 'run_' + str(run_) +'/class_'+str(class_index)
                        stgcn_run_dir = stgcn_run_dir_class + '/best_stgcn.pth'
                        feature_extractor_stgcn = FeatureExtractor(model_name=args.regression_models, model_path=stgcn_run_dir, device=args.device)
                        print('----------------------------------------------------------',stgcn_run_dir)

                        fid_calculator_stgcn = FID(feature_extractor_stgcn)
                        coverage_calculator_stgcn = Coverage(feature_extractor_stgcn)
                        mms_calculator_stgcn = MMS(feature_extractor_stgcn)
                        density_calculator_stgcn = Density(feature_extractor_stgcn)
                        apd_calculator_stgcn = APD(feature_extractor_stgcn)
                        csv_path = output_directory_skeletons_class + f'/{args.regression_models}_train_vs_noisy_vs_gen_{class_index}.csv'
                        if os.path.exists(csv_path):
                                df = pd.read_csv(csv_path)
                        else:
                                df = pd.DataFrame(columns=['ON','FID','COV','MMS','Density','APD'])

                        fid_score = fid_calculator_stgcn.calculate_fid(train_loader, noisy_loader)
                        fid_mean = np.mean(fid_score)
                        
                        #COV metric
                        cov_score = coverage_calculator_stgcn.calculate(train_loader, noisy_loader)
                        cov_mean = np.mean(cov_score)

                        #MMS metric
                        mms_score = mms_calculator_stgcn.calculate_mms(train_loader, noisy_loader)
                        mms_mean = np.mean(mms_score)

                        #Density metric
                        mms_score = density_calculator_stgcn.calculate_density(train_loader, noisy_loader)
                        density_mean = np.mean(mms_score)
                        #APD metric
                        apd_score = apd_calculator_stgcn.calculate_apd(train_loader)
                        apd_mean = np.mean(apd_score)


                        if 'trainVSnoise' in df['ON'].values:
                                df.loc[df['ON'] == 'trainVSnoise', 'FID'] = fid_mean
                                df.loc[df['ON'] == 'trainVSnoise', 'COV'] = cov_mean
                                df.loc[df['ON'] == 'trainVSnoise', 'MMS'] = mms_mean
                                df.loc[df['ON'] == 'trainVSnoise', 'Density'] = density_mean
                                df.loc[df['ON'] == 'trainVSnoise', 'APD'] = apd_mean
                        else:
                                new_row = pd.DataFrame([{'ON':'trainVSnoise', 'FID': fid_mean, 'COV': cov_mean, 'MMS': mms_mean,'Density':density_mean,'APD':apd_mean}])
                                df = pd.concat([df, new_row], ignore_index=True)
                            
                
                        

                        #FID metric
                        fid_score = fid_calculator_stgcn.calculate_fid(gen_loader, train_loader)
                        fid_mean = np.mean(fid_score)
                        print('----------------------------------------------fid done')
                        if fid_mean < 0 :
                            np.save('negative_generated_samples.npy',generated_samples)

                        #COV metric
                        cov_score = coverage_calculator_stgcn.calculate(xgenerated_loader=gen_loader, xreal_loader=train_loader)
                        cov_mean = np.mean(cov_score)
                        print('----------------------------------------------cov done')

                        #MMS metric
                        mms_score = mms_calculator_stgcn.calculate_mms(generated_data=gen_loader, real_data=train_loader)
                        mms_mean = np.mean(mms_score)

                        #Density metric
                        mms_score = density_calculator_stgcn.calculate_density(gen_loader, train_loader)
                        density_mean = np.mean(mms_score)

                        apd_score = apd_calculator_stgcn.calculate_apd(gen_loader)
                        apd_mean = np.mean(apd_score)
                    

                        if 'trainVSgen' in df['ON'].values:
                                df.loc[df['ON'] == 'trainVSgen', 'FID'] = fid_mean
                                df.loc[df['ON'] == 'trainVSgen', 'COV'] = cov_mean
                                df.loc[df['ON'] == 'trainVSgen', 'MMS'] = mms_mean
                                df.loc[df['ON'] == 'trainVSgen', 'Density'] = density_mean
                                df.loc[df['ON'] == 'trainVSgen', 'APD'] = apd_mean
                        else:
                                new_row = pd.DataFrame([{'ON':'trainVSgen', 'FID': fid_mean, 'COV': cov_mean, 'MMS': mms_mean,'Density':density_mean,'APD':apd_mean}])
                                df = pd.concat([df, new_row], ignore_index=True)
                    
                        df.to_csv(csv_path, index=False)


                #########################################################################################################################################################################""

                        if os.path.exists(output_directory_skeletons_class + f'/{args.regression_models}_train_vs_test{class_index}.csv'):
                                df = pd.read_csv(output_directory_skeletons_class + f'/{args.regression_models}_train_vs_test{class_index}.csv')
                        else:
                                df = pd.DataFrame(columns=['ON','FID','COV','MMS','Density','APD'])

                        fid_score = fid_calculator_stgcn.calculate_fid(train_loader, test_loader)
                        fid_mean = np.mean(fid_score)
                        
                        #COV metric
                        cov_score = coverage_calculator_stgcn.calculate(train_loader, test_loader)
                        cov_mean = np.mean(cov_score)

                        #MMS metric
                        mms_score = mms_calculator_stgcn.calculate_mms(train_loader, test_loader)
                        mms_mean = np.mean(mms_score)

                        #Density metric
                        mms_score = density_calculator_stgcn.calculate_density(train_loader, test_loader)
                        density_mean = np.mean(mms_score)
                        #APD metric
                        apd_score = apd_calculator_stgcn.calculate_apd(train_loader)
                        apd_mean = np.mean(apd_score)


                            
                        if 'train_vs_test' in df['ON'].values:
                                df.loc[df['ON'] == 'train_vs_test', 'FID'] = fid_mean
                                df.loc[df['ON'] == 'train_vs_test', 'COV'] = cov_mean
                                df.loc[df['ON'] == 'train_vs_test', 'MMS'] = mms_mean
                                df.loc[df['ON'] == 'train_vs_test', 'Density'] = density_mean
                                df.loc[df['ON'] == 'train_vs_test', 'APD'] = apd_mean
                        else:
                                new_row = pd.DataFrame([{'ON':'train_vs_test', 'FID': fid_mean, 'COV': cov_mean, 'MMS': mms_mean,'Density':density_mean,'APD':apd_mean}])
                                df = pd.concat([df, new_row], ignore_index=True)  
                      
                        #FID metric
                        fid_score = fid_calculator_stgcn.calculate_fid(test_gen_loader, train_loader)
                        fid_mean = np.mean(fid_score)
                        print('----------------------------------------------fid done')
                        if fid_mean < 0 :
                            np.save('negative_generated_samples.npy',generated_samples)

                        #COV metric
                        cov_score = coverage_calculator_stgcn.calculate(xgenerated_loader=test_gen_loader, xreal_loader=train_loader)
                        cov_mean = np.mean(cov_score)
                        print('----------------------------------------------cov done')

                        #MMS metric
                        mms_score = mms_calculator_stgcn.calculate_mms(generated_data=test_gen_loader, real_data=train_loader)
                        mms_mean = np.mean(mms_score)

                        #Density metric
                        mms_score = density_calculator_stgcn.calculate_density(test_gen_loader, train_loader)
                        density_mean = np.mean(mms_score)

                        apd_score = apd_calculator_stgcn.calculate_apd(test_gen_loader)
                        apd_mean = np.mean(apd_score)
                    

                        if 'trainVStest_gen' in df['ON'].values:
                                df.loc[df['ON'] == 'trainVStest_gen', 'FID'] = fid_mean
                                df.loc[df['ON'] == 'trainVStest_gen', 'COV'] = cov_mean
                                df.loc[df['ON'] == 'trainVStest_gen', 'MMS'] = mms_mean
                                df.loc[df['ON'] == 'trainVStest_gen', 'Density'] = density_mean
                                df.loc[df['ON'] == 'trainVStest_gen', 'APD'] = apd_mean
                        else:
                                new_row = pd.DataFrame([{'ON':'trainVStest_gen', 'FID': fid_mean, 'COV': cov_mean, 'MMS': mms_mean,'Density':density_mean,'APD':apd_mean}])
                                df = pd.concat([df, new_row], ignore_index=True)  
                    
                        df.to_csv(output_directory_skeletons_class + f'/{args.regression_models}_train_vs_test{class_index}.csv', index=False)






                elif args.regression_models == 'REG':
                        regressor_directory = output_directory_results + 'REG/'
                        reg_run_dir_class = regressor_directory + 'run_' + str(run_) +'/class_'+str(class_index)
                        reg_run_dir = reg_run_dir_class + '/best_regressor.pth'
                        
                        feature_extractor_reg = FeatureExtractor(model_name=args.regression_models, model_path=reg_run_dir, device=args.device)

                        fid_calculator = FID(feature_extractor_reg)
                        coverage_calculator = Coverage(feature_extractor_reg)
                        mms_calculator= MMS(feature_extractor_reg)
                        density_calculator_reg = Density(feature_extractor_reg)
                        apd_calculator_reg = APD(feature_extractor_reg)



                        if os.path.exists(output_directory_skeletons_class + f'/{args.regression_models}_train_vs_noisy_vs_gen_{class_index}.csv'):
                                df = pd.read_csv(output_directory_skeletons_class + f'/{args.regression_models}_train_vs_noisy_vs_gen_{class_index}.csv')
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



                        if 'trainVSnoise' in df['ON'].values:
                                df.loc[df['ON'] == 'trainVSnoise', 'FID'] = fid_mean
                                df.loc[df['ON'] == 'trainVSnoise', 'COV'] = cov_mean
                                df.loc[df['ON'] == 'trainVSnoise', 'MMS'] = mms_mean
                                df.loc[df['ON'] == 'trainVSnoise', 'Density'] = density_mean
                                df.loc[df['ON'] == 'trainVSnoise', 'APD'] = apd_mean
                        else:
                                new_row = pd.DataFrame([{'ON':'trainVSnoise', 'FID': fid_mean, 'COV': cov_mean, 'MMS': mms_mean,'Density':density_mean,'APD':apd_mean}])
                                df = pd.concat([df, new_row], ignore_index=True)
                            
                            
                


                        #FID metric
                        fid_score = fid_calculator.calculate_fid(gen_loader, train_loader)
                        fid_mean = np.mean(fid_score)
                        print('----------------------------------------------fid done')
                        if fid_mean < 0 :
                            np.save('negative_generated_samples.npy',generated_samples)

                        #COV metric
                        cov_score = coverage_calculator.calculate(xgenerated_loader=gen_loader, xreal_loader=train_loader)
                        cov_mean = np.mean(cov_score)
                        print('----------------------------------------------cov done')

                        #MMS metric
                        mms_score = mms_calculator.calculate_mms(generated_data=gen_loader, real_data=train_loader)
                        mms_mean = np.mean(mms_score)

                        #Density metric
                        mms_score = density_calculator_reg.calculate_density(gen_loader, train_loader)
                        density_mean = np.mean(mms_score)

                        apd_score = apd_calculator_reg.calculate_apd(gen_loader)
                        apd_mean = np.mean(apd_score)
                        if 'trainVSgen' in df['ON'].values:
                                df.loc[df['ON'] == 'trainVSgen', 'FID'] = fid_mean
                                df.loc[df['ON'] == 'trainVSgen', 'COV'] = cov_mean
                                df.loc[df['ON'] == 'trainVSgen', 'MMS'] = mms_mean
                                df.loc[df['ON'] == 'trainVSgen', 'Density'] = density_mean
                                df.loc[df['ON'] == 'trainVSgen', 'APD'] = apd_mean
                        else:
                                new_row = pd.DataFrame([{'ON':'trainVSgen', 'FID': fid_mean, 'COV': cov_mean, 'MMS': mms_mean,'Density':density_mean,'APD':apd_mean}])
                                df = pd.concat([df, new_row], ignore_index=True)
                    
                        df.to_csv(output_directory_skeletons_class + f'/{args.regression_models}_train_vs_noisy_vs_gen_{class_index}.csv', index=False)
                #########################################################################################################################################################################""

                        if os.path.exists(output_directory_skeletons_class + f'/{args.regression_models}_train_vs_test{class_index}.csv'):
                                df = pd.read_csv(output_directory_skeletons_class + f'/{args.regression_models}_train_vs_test{class_index}.csv')
                        else:
                                df = pd.DataFrame(columns=['ON','FID','COV','MMS','Density','APD'])

                        fid_score = fid_calculator.calculate_fid(train_loader, test_loader)
                        fid_mean = np.mean(fid_score)
                        
                        #COV metric
                        cov_score = coverage_calculator.calculate(train_loader, test_loader)
                        cov_mean = np.mean(cov_score)

                        #MMS metric
                        mms_score = mms_calculator.calculate_mms(train_loader, test_loader)
                        mms_mean = np.mean(mms_score)

                        #Density metric
                        density_core = density_calculator_reg.calculate_density(train_loader, test_loader)
                        density_mean = np.mean(density_core)
                        #APD metric
                        apd_score = apd_calculator_reg.calculate_apd(train_loader)
                        apd_mean = np.mean(apd_score)

                        if 'train_vs_test' in df['ON'].values:
                                df.loc[df['ON'] == 'train_vs_test', 'FID'] = fid_mean
                                df.loc[df['ON'] == 'train_vs_test', 'COV'] = cov_mean
                                df.loc[df['ON'] == 'train_vs_test', 'MMS'] = mms_mean
                                df.loc[df['ON'] == 'train_vs_test', 'Density'] = density_mean
                                df.loc[df['ON'] == 'train_vs_test', 'APD'] = apd_mean
                        else:
                                new_row = pd.DataFrame([{'ON':'train_vs_test', 'FID': fid_mean, 'COV': cov_mean, 'MMS': mms_mean,'Density':density_mean,'APD':apd_mean}])
                                df = pd.concat([df, new_row], ignore_index=True)  
                            
                
                      

                        #FID metric
                        fid_score = fid_calculator.calculate_fid(test_gen_loader, train_loader)
                        fid_mean = np.mean(fid_score)
                        print('----------------------------------------------fid done')
                        if fid_mean < 0 :
                            np.save('negative_generated_samples.npy',generated_samples)

                        #COV metric
                        cov_score = coverage_calculator.calculate(xgenerated_loader=test_gen_loader, xreal_loader=train_loader)
                        cov_mean = np.mean(cov_score)
                        print('----------------------------------------------cov done')

                        #MMS metric
                        mms_score = mms_calculator.calculate_mms(generated_data=test_gen_loader, real_data=train_loader)
                        mms_mean = np.mean(mms_score)

                        #Density metric
                        density_score = density_calculator_reg.calculate_density(test_gen_loader, train_loader)
                        density_mean = np.mean(density_score)

                        apd_score = apd_calculator_reg.calculate_apd(test_gen_loader)
                        apd_mean = np.mean(apd_score)
                        if 'trainVStest_gen' in df['ON'].values:
                                df.loc[df['ON'] == 'trainVStest_gen', 'FID'] = fid_mean
                                df.loc[df['ON'] == 'trainVStest_gen', 'COV'] = cov_mean
                                df.loc[df['ON'] == 'trainVStest_gen', 'MMS'] = mms_mean
                                df.loc[df['ON'] == 'trainVStest_gen', 'Density'] = density_mean
                                df.loc[df['ON'] == 'trainVStest_gen', 'APD'] = apd_mean
                        else:
                                new_row = pd.DataFrame([{'ON':'trainVStest_gen', 'FID': fid_mean, 'COV': cov_mean, 'MMS': mms_mean,'Density':density_mean,'APD':apd_mean}])
                                df = pd.concat([df, new_row], ignore_index=True)  
                    
                    
                        df.to_csv(output_directory_skeletons_class + f'/{args.regression_models}_train_vs_test{class_index}.csv', index=False)






