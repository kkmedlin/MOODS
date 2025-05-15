import random
import torch.nn as nn
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

#from data import pytorch_prep, create_imbalanced_samplers, connect4, abalone, spamBase, shuttle
from smallData import pytorch_prep, create_imbalanced_samplers, smoteGANdata
from moods import MOODS

array = [  
        #abalone(),
        #spamBase(),
        #connect4(),
        #shuttle(),
        smoteGANdata()  #option: 0 for Pageblocks, 1 for Ecoli, 2 for Poker, 3 for Wine, 4 for yeast
]

title = [ 'Ecoli' ]
 
for (i,element) in enumerate(array):        
        #setting up train/test data for classification
        x_train_np, y_train_np, x_test_np, y_test_np, x_valid_np, y_valid_np = element
        x_train, y_train, x_test, y_test, x_valid, y_valid, y_train_np, y_test_np, y_valid_np = pytorch_prep( x_train_np, y_train_np, x_test_np, y_test_np, x_valid_np, y_valid_np)
        #create samplers for classification by running train/test data through create_samplers function in data.py
        X_state, y_label, X_state_test, y_label_test, X_state_valid, y_label_valid, majority_train_data_tensor, \
                minority_train_data_tensor = create_imbalanced_samplers(x_train, y_train, x_test, y_test, x_valid, \
                    y_valid, y_train_np, y_test_np, y_valid_np)
        M = len(majority_train_data_tensor)
        m = len(minority_train_data_tensor)
        #setting up num_runs and num_steps 
        n_runs = 1
        n_steps = 1
        n_metrics=6
    
        #setting up vectors/matrices for storing classification results   
        zMAJ = []
        zMIN = []
        inMarginMAJ = []
        inMarginMIN = []
        markovChain = []
        all_index = [] 
        all_effOneValidMIN = []
        all_effOneValid = []
        all_effOneTestMIN = []
        all_effOneTest = []
        all_jGrad = []
        all_majSampleSize = []
        all_minSampleSize = []
        minoritySize = 0
        accepted_index = []   
        accepted_accuracy = []   
        accepted_effOneValidMIN = []
        accepted_effOneValid = []
        accepted_effOneTestMIN = []
        accepted_effOneTest = []
        accepted_jGrad = []
        accepted_majSampleSize = []
        accepted_minSampleSize = []
        accepted_precisionMIN = []
        accepted_precisionAVG = []
        
        #running 'MOODS' 
        for j in range(n_runs):
            random_seed = 111 + 10000 * j
            random.seed(random_seed)
            print("run and random seed:",j, random_seed) 
            return__info = MOODS( majority_train_data_tensor, minority_train_data_tensor, n_steps, X_state_test, y_label_test, X_state_valid, y_label_valid)
            model, markovChain, minoritySize, index_all, all_F1Train, all_minF1Train,\
            all_F1Test, all_minF1Test, \
                jayGrad_all, majSampleSize_all, \
                    minSampleSize_all, index_accepted, accuracy_accepted, accepted_F1Train, accepted_minF1Train, \
                accepted_F1Test, accepted_minF1Test, precisionAVG_accepted, precisionMIN_accepted, \
                    jayGrad_accepted, majSampleSize_accepted, minSampleSize_accepted,  \
                            zeeMAJ, zeeMIN     = return__info
        
            #capture Accepted Training Set 'markov chain' from LAST STEP combined sample 
            y_Markov = markovChain[:,-1]
            X_Markov = markovChain[:,0:-1]
            pd.DataFrame(y_Markov).to_csv(str(title[i])+'_y_acceptedTRAIN_'+str([j])+'th_run.csv')
            pd.DataFrame(X_Markov).to_csv(str(title[i])+'_X_acceptedTRAIN_'+str([j])+'th_run.csv')             
            markovChain = []
            
            #fill vector with index of ALL steps 
            all_index.append(index_all)
            
            #fill vector with iteration j zMIN and zMAJ values
            zMIN_np = np.array(zeeMIN)
            zMIN.append(zMIN_np)
            zMAJ_np = np.array(zeeMAJ)
            zMAJ.append(zMAJ_np)
            #store, export to CSV, and print j-th run's ALL metrics -- accuracy, effOne, loss, update ratio, preciion
            zeeMIN_byRun = np.array(zeeMIN,dtype=np.float32)
            pd.DataFrame(zeeMIN_byRun).to_csv(str(title[i])+'_zMIN_'+str([j])+'th_run.csv')
            zeeMAJ_byRun = np.array(zeeMAJ,dtype=np.float32)
            pd.DataFrame(zeeMAJ_byRun).to_csv(str(title[i])+'_zMAJ_'+str([j])+'th_run.csv')
            
            # #fill vector with index of ACCEPTED steps 
            accepted_index.append(index_accepted)
            
            #fill vectors with ACCEPTED metrics -- accuracy, effOne, loss, update ratio, precision
            accepted_accuracy_np = np.array(accuracy_accepted)
            accepted_accuracy.append(accepted_accuracy_np)
            accepted_effOneValidMIN_np = np.array(accepted_minF1Train)
            accepted_effOneValidMIN.append(accepted_effOneValidMIN_np)
            accepted_effOneValid_np = np.array(accepted_F1Train)
            accepted_effOneValid.append(accepted_effOneValid_np)
            accepted_effOneTest_np = np.array(accepted_F1Test)
            accepted_effOneTest.append(accepted_effOneTest_np)
            accepted_effOneTestMIN_np = np.array(accepted_minF1Test)
            accepted_effOneTestMIN.append(accepted_effOneTestMIN_np)
            accepted_jayGrad_np = np.array(jayGrad_accepted)
            accepted_jGrad.append(accepted_jayGrad_np)
            accepted_majSampleSize_np = np.array(majSampleSize_accepted)
            accepted_majSampleSize.append(accepted_majSampleSize_np)
            accepted_minSampleSize_np = np.array(minSampleSize_accepted)
            accepted_minSampleSize.append(accepted_minSampleSize_np)
            accepted_precisionMIN_np = np.array(precisionMIN_accepted)
            accepted_precisionMIN.append(accepted_precisionMIN_np)
            accepted_precisionAVG_np = np.array(precisionAVG_accepted)
            accepted_precisionAVG.append(accepted_precisionAVG_np)
            
            #store, export to CSV, and print j-th run's ACCEPTED metrics -- accuracy, effOne, loss, update ratio, preciion
            accepted_effOneTrain_byRun = np.array(accepted_F1Train,dtype=np.float32)
            pd.DataFrame(accepted_effOneTrain_byRun).to_csv(str(title[i])+'_accepted_F1Validation_'+str([j])+'th_run.csv')
            accepted_effOneTrainMIN_byRun = np.array(accepted_minF1Train,dtype=np.float32)
            pd.DataFrame(accepted_effOneTrainMIN_byRun).to_csv(str(title[i])+'_accepted_minF1Validation_'+str([j])+'th_run.csv')
            accepted_effOneTest_byRun = np.array(accepted_F1Test,dtype=np.float32)
            pd.DataFrame(accepted_effOneTest_byRun).to_csv(str(title[i])+'_F1Test_'+str([j])+'th_run.csv')
            accepted_effOneTestMIN_byRun = np.array(accepted_minF1Test,dtype=np.float32)
            pd.DataFrame(accepted_effOneTestMIN_byRun).to_csv(str(title[i])+'_minF1Test_'+str([j])+'th_run.csv')
            accepted_precisionMIN_byRun = np.array(precisionMIN_accepted,dtype=np.float32)
            pd.DataFrame(accepted_precisionMIN_byRun).to_csv(str(title[i])+'_minPrecisionTest_'+str([j])+'th_run.csv')
            accepted_precisionAVG_byRun = np.array(precisionAVG_accepted,dtype=np.float32)
            pd.DataFrame(accepted_precisionAVG_byRun).to_csv(str(title[i])+'_avgPrecisionTest_'+str([j])+'th_run.csv')
            accepted_accuracy_byRun = np.array(accuracy_accepted,dtype=np.float32)
            pd.DataFrame(accepted_accuracy_byRun).to_csv(str(title[i])+'_accuracyTest_'+str([j])+'th_run.csv')
            accepted_majSampleSize_byRun = np.array(majSampleSize_accepted,dtype=np.float32)
            pd.DataFrame(accepted_majSampleSize_byRun).to_csv(str(title[i])+'_accepted_majSampleSizeTrain_'+str([j])+'th_run.csv')   
            accepted_minSampleSize_byRun = np.array(minSampleSize_accepted,dtype=np.float32)
            pd.DataFrame(accepted_minSampleSize_byRun).to_csv(str(title[i])+'_accepted_minSampleSizeTrain_'+str([j])+'th_run.csv')    
            accepted_jayGrad_byRun = np.array(jayGrad_accepted,dtype=np.float32)
            #pd.DataFrame(accepted_costRatio_byRun).to_csv(str(title[i])+'_accepted_L2normGradient_'+str([j])+'th_run.csv') 
        
        
        
        
        
        
        