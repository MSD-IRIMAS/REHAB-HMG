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


    score_output_directory_weights_losses = output_directory_gen_models + 'Score+Action_conditioned' + '/ASCVAE/' +'Wrec_' + str(args.wrec) + '_Wkl_' + str(args.wkl) + '/'
      

    label_output_directory_weights_losses = output_directory_gen_models + 'Action_conditioned' + '/CVAEL/'+ 'Wrec_' + str(args.wrec) + '_Wkl_' + str(args.wkl) + '/'
        

    
    
    for class_index in range(args.class_index):
                weights_loss = {
                            'wrec' : args.wrec,
                            'wkl' : args.wkl,}
                data,labels,scores = load_class(class_index,root_dir=dataset_dir)
   
                xtrain,xtest,ytrain,ytest,strain,stest= train_test_split(data,labels,scores,test_size=0.3,random_state=42)
                xtrain,min_X, max_X,min_Y,max_Y, min_Z,max_Z= normalize_skeletons(xtrain)
                noisy_data= noise_data(xtrain)
                noisy_set = Kimore(noisy_data,ytrain,strain)

                train_set = Kimore(xtrain,ytrain,strain)
                train_loader = DataLoader(train_set,batch_size=16,shuffle =False)
                noisy_loader = DataLoader(noisy_set,batch_size=16,shuffle=False)
                xtest,_,_,_,_,_,_= normalize_skeletons(xtest,min_X, max_X,min_Y,max_Y, min_Z,max_Z)
                test_set = Kimore(xtest,ytest,stest)
                test_loader = DataLoader(test_set,batch_size=16,shuffle=False)
                strain=strain.squeeze(1)
                stest=stest.squeeze(1)
                
                
                
                score_output_directory_run = score_output_directory_weights_losses + 'run_' + str(0) + '/'
                create_directory(score_output_directory_run)
                score_output_directory_skeletons = score_output_directory_run + 'generated_samples/'
                create_directory(score_output_directory_skeletons)

                score_output_directory_skeletons_class = score_output_directory_run + 'class_' + str(class_index) + '/'
                create_directory(score_output_directory_skeletons_class)
                
                

                score_generator= ASCVAE(output_directory=score_output_directory_run, device=args.device, w_rec=weights_loss['wrec'], w_kl=weights_loss['wkl'])
                as_generated_samples = score_generator.generate_by_class(device = args.device,gif_directory=score_output_directory_skeletons_class,scores=strain,class_index=class_index)
                as_test_generated_samples = score_generator.generate_by_class(device = args.device,gif_directory=score_output_directory_skeletons_class,scores=stest,class_index=class_index)
               
                as_gen_set = Kimore(as_generated_samples, ytrain, strain)
           
                as_gen_loader = DataLoader(as_gen_set, batch_size=16, shuffle=True)
                as_gen_set = Kimore(as_test_generated_samples, ytest, stest)
                as_test_gen_loader = DataLoader(as_gen_set, batch_size=16, shuffle=True)

                label_output_directory_run = label_output_directory_weights_losses + 'run_' + str(0) + '/'
                create_directory(label_output_directory_run)
                label_output_directory_skeletons = label_output_directory_run + 'generated_samples/'
                create_directory(label_output_directory_skeletons)

                label_output_directory_skeletons_class = label_output_directory_skeletons + 'class_' + str(class_index) + '/'
                create_directory(label_output_directory_skeletons_class)
                generator= CVAEL(output_directory=label_output_directory_run, device=args.device,epochs=2000)
                a_generated_samples = generator.generate_samples_from_prior(device = args.device,gif_directory=label_output_directory_skeletons_class,num_samples=len(xtrain),class_index=class_index)
                a_test_generated_samples = generator.generate_samples_from_prior(device = args.device,gif_directory=label_output_directory_skeletons_class,num_samples=len(xtest),class_index=class_index)
        
    
                a_gen_set = Kimore(a_generated_samples, labels, strain)
                a_gen_loader = DataLoader(a_gen_set, batch_size=16, shuffle=True)
                a_gen_set = Kimore(a_test_generated_samples, labels, stest)
                a_test_gen_loader = DataLoader(a_gen_set, batch_size=16, shuffle=True)
     
                

                if args.regression_models == 'STGCN':


                        stgcn_directory = output_directory_results + 'STGCN/'
                        stgcn_run_dir_class = stgcn_directory + 'run_' + str(0) +'/class_'+str(class_index)
                        stgcn_run_dir = stgcn_run_dir_class + '/best_stgcn.pth'
                        feature_extractor_stgcn = FeatureExtractor(model_name=args.regression_models, model_path=stgcn_run_dir, device=args.device)
                        print('----------------------------------------------------------',stgcn_run_dir)

                        fid_calculator_stgcn = FID(feature_extractor_stgcn)
                        coverage_calculator_stgcn = Coverage(feature_extractor_stgcn)
                        mms_calculator_stgcn = MMS(feature_extractor_stgcn)
                        density_calculator_stgcn = Density(feature_extractor_stgcn)
                        apd_calculator_stgcn = APD(feature_extractor_stgcn)
                        csv_path = stgcn_run_dir_class + f'/{args.regression_models}_train_vs_noisy_vs_gen_{class_index}.csv'
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
                        print(len(train_loader))
                        fid_score = fid_calculator_stgcn.calculate_fid(as_gen_loader, train_loader)
                        fid_mean = np.mean(fid_score)
                        print('----------------------------------------------fid done')
          

                        #COV metric
                        cov_score = coverage_calculator_stgcn.calculate(xgenerated_loader=as_gen_loader, xreal_loader=train_loader)
                        cov_mean = np.mean(cov_score)
                        print('----------------------------------------------cov done')

                        #MMS metric
                        mms_score = mms_calculator_stgcn.calculate_mms(generated_data=as_gen_loader, real_data=train_loader)
                        mms_mean = np.mean(mms_score)

                        #Density metric
                        mms_score = density_calculator_stgcn.calculate_density(as_gen_loader, train_loader)
                        density_mean = np.mean(mms_score)

                        apd_score = apd_calculator_stgcn.calculate_apd(as_gen_loader)
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
                        



                        #FID metric
                        print(len(train_loader))
                        fid_score = fid_calculator_stgcn.calculate_fid(a_gen_loader, train_loader)
                        fid_mean = np.mean(fid_score)
                        print('----------------------------------------------fid done')
          

                        #COV metric
                        cov_score = coverage_calculator_stgcn.calculate(xgenerated_loader=a_gen_loader, xreal_loader=train_loader)
                        cov_mean = np.mean(cov_score)
                        print('----------------------------------------------cov done')

                        #MMS metric
                        mms_score = mms_calculator_stgcn.calculate_mms(generated_data=a_gen_loader, real_data=train_loader)
                        mms_mean = np.mean(mms_score)

                        #Density metric
                        mms_score = density_calculator_stgcn.calculate_density(a_gen_loader, train_loader)
                        density_mean = np.mean(mms_score)

                        apd_score = apd_calculator_stgcn.calculate_apd(a_gen_loader)
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



                    
                        df.to_csv(csv_path, index=False)







#########################################################################################################################################################################""

                        if os.path.exists(stgcn_run_dir_class + f'/{args.regression_models}_train_vs_test{class_index}.csv'):
                                df = pd.read_csv(stgcn_run_dir_class + f'/{args.regression_models}_train_vs_test{class_index}.csv')
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
                        fid_score = fid_calculator_stgcn.calculate_fid(as_test_gen_loader, train_loader)
                        fid_mean = np.mean(fid_score)
                        print('----------------------------------------------fid done')
                    
                        cov_score = coverage_calculator_stgcn.calculate(xgenerated_loader=as_test_gen_loader, xreal_loader=train_loader)
                        cov_mean = np.mean(cov_score)
                        print('----------------------------------------------cov done')

                        #MMS metric
                        mms_score = mms_calculator_stgcn.calculate_mms(generated_data=as_test_gen_loader, real_data=train_loader)
                        mms_mean = np.mean(mms_score)

                        #Density metric
                        mms_score = density_calculator_stgcn.calculate_density(as_test_gen_loader, train_loader)
                        density_mean = np.mean(mms_score)

                        apd_score = apd_calculator_stgcn.calculate_apd(as_test_gen_loader)
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



                        #FID metric
                        fid_score = fid_calculator_stgcn.calculate_fid(a_test_gen_loader, train_loader)
                        fid_mean = np.mean(fid_score)
                        print('----------------------------------------------fid done')
                    
                        cov_score = coverage_calculator_stgcn.calculate(xgenerated_loader=a_test_gen_loader, xreal_loader=train_loader)
                        cov_mean = np.mean(cov_score)
                        print('----------------------------------------------cov done')

                        #MMS metric
                        mms_score = mms_calculator_stgcn.calculate_mms(generated_data=a_test_gen_loader, real_data=train_loader)
                        mms_mean = np.mean(mms_score)

                        #Density metric
                        mms_score = density_calculator_stgcn.calculate_density(a_test_gen_loader, train_loader)
                        density_mean = np.mean(mms_score)

                        apd_score = apd_calculator_stgcn.calculate_apd(a_test_gen_loader)
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
 
                    
                        df.to_csv(stgcn_run_dir_class + f'/{args.regression_models}_train_vs_test{class_index}.csv', index=False)






                elif args.regression_models == 'REG':
                        regressor_directory = output_directory_results + 'REG/'
                        reg_run_dir_class = regressor_directory + 'run_' + str(0) +'/class_'+str(class_index)
                        reg_run_dir = reg_run_dir_class + '/best_regressor.pth'
                        
                        feature_extractor_reg = FeatureExtractor(model_name=args.regression_models, model_path=reg_run_dir, device=args.device)

                        fid_calculator = FID(feature_extractor_reg)
                        coverage_calculator = Coverage(feature_extractor_reg)
                        mms_calculator= MMS(feature_extractor_reg)
                        density_calculator_reg = Density(feature_extractor_reg)
                        apd_calculator_reg = APD(feature_extractor_reg)



                        if os.path.exists(reg_run_dir_class + f'/{args.regression_models}_train_vs_noisy_vs_gen_{class_index}.csv'):
                                df = pd.read_csv(reg_run_dir_class + f'/{args.regression_models}_train_vs_noisy_vs_gen_{class_index}.csv')
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
                        fid_score = fid_calculator.calculate_fid(as_gen_loader, train_loader)
                        fid_mean = np.mean(fid_score)
                  
                        cov_score = coverage_calculator.calculate(xgenerated_loader=as_gen_loader, xreal_loader=train_loader)
                        cov_mean = np.mean(cov_score)
                        print('----------------------------------------------cov done')

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




                        #FID metric
                        fid_score = fid_calculator.calculate_fid(a_gen_loader, train_loader)
                        fid_mean = np.mean(fid_score)
                  
                        cov_score = coverage_calculator.calculate(xgenerated_loader=a_gen_loader, xreal_loader=train_loader)
                        cov_mean = np.mean(cov_score)
                        print('----------------------------------------------cov done')

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
                        
                    
                        df.to_csv(reg_run_dir_class + f'/{args.regression_models}_train_vs_noisy_vs_gen_{class_index}.csv', index=False)
                #########################################################################################################################################################################""

                        if os.path.exists(reg_run_dir_class + f'/{args.regression_models}_train_vs_test{class_index}.csv'):
                                df = pd.read_csv(reg_run_dir_class + f'/{args.regression_models}_train_vs_test{class_index}.csv')
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
                        fid_score = fid_calculator.calculate_fid(as_test_gen_loader, train_loader)
                        fid_mean = np.mean(fid_score)
                        print('----------------------------------------------fid done')
                       

                        #COV metric
                        cov_score = coverage_calculator.calculate(xgenerated_loader=as_test_gen_loader, xreal_loader=train_loader)
                        cov_mean = np.mean(cov_score)
                        print('----------------------------------------------cov done')

                        #MMS metric
                        mms_score = mms_calculator.calculate_mms(generated_data=as_test_gen_loader, real_data=train_loader)
                        mms_mean = np.mean(mms_score)

                        #Density metric
                        density_score = density_calculator_reg.calculate_density(as_test_gen_loader, train_loader)
                        density_mean = np.mean(density_score)

                        apd_score = apd_calculator_reg.calculate_apd(as_test_gen_loader)
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



                        fid_score = fid_calculator.calculate_fid(a_test_gen_loader, train_loader)
                        fid_mean = np.mean(fid_score)
                        print('----------------------------------------------fid done')
                        
                        #COV metric
                        cov_score = coverage_calculator.calculate(xgenerated_loader=a_test_gen_loader, xreal_loader=train_loader)
                        cov_mean = np.mean(cov_score)
                        print('----------------------------------------------cov done')

                        #MMS metric
                        mms_score = mms_calculator.calculate_mms(generated_data=a_test_gen_loader, real_data=train_loader)
                        mms_mean = np.mean(mms_score)

                        #Density metric
                        density_score = density_calculator_reg.calculate_density(a_test_gen_loader, train_loader)
                        density_mean = np.mean(density_score)

                        apd_score = apd_calculator_reg.calculate_apd(a_test_gen_loader)
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
                    
                    
                        df.to_csv(reg_run_dir_class + f'/{args.regression_models}_train_vs_test{class_index}.csv', index=False)


