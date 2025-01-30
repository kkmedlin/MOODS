import random
import torch.nn as nn
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

#from dataValid import pytorch_prep, create_imbalanced_samplers, Gisette, connect4, abalone, spamBase, shuttle
from smallData import pytorch_prep, create_imbalanced_samplers, smoteGANdata
from moods import MOODS

array = [  
        #Gisette(),
        #abalone(),
        #spamBase(),
        #connect4(),
        #shuttle(),
        smoteGANdata()  #option: 0 for Pageblocks, 1 for Ecoli, 2 for Poker, 3 for Wine, 4 for yeast, 5 for abalone, 6 for ionosphere, 7 for spambase
]

title = [ 'WineNaive' ]
 
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
        n_steps = 2
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
        
        #running Method called 'MUBO' and generating results
       
        for j in range(n_runs):
            #random_seed = 57 + (5732 * (j))
            random_seed = 111 + 5732 * j
            random.seed(random_seed)
            print("run and random seed:",j, random_seed) 
            #outputs of PLEUM: 'all_accuracy', 'index_accepted_steps', 'accepted_accuracy', 'accepted_f1', array of metrics: accuracy, F1, precision, recall, AUC
            return__info = BALBO( majority_train_data_tensor, minority_train_data_tensor, n_steps, X_state_test, y_label_test, X_state_valid, y_label_valid)
            model, markovChain, minoritySize, index_all, all_F1Train, all_minF1Train,\
            all_F1Test, all_minF1Test, \
                jayGrad_all, majSampleSize_all, \
                    minSampleSize_all, index_accepted, accuracy_accepted, accepted_F1Train, accepted_minF1Train, \
                accepted_F1Test, accepted_minF1Test, precisionAVG_accepted, precisionMIN_accepted, \
                    jayGrad_accepted, majSampleSize_accepted, minSampleSize_accepted,  \
                            zeeMAJ, zeeMIN     = return__info
        
            #Print model's state_dict
            #print("Post MCMC Model's state_dict:")
            #for param_tensor in model.state_dict():
                #print(param_tensor, "\t", model.state_dict()[param_tensor].size())
            #Save model's state_dict
            torch.save(model.state_dict(), 'C:/Users/kamed/Desktop/argonne_K/git/MUBO'+title[i]+'_model_' +str([j])+'th_run')
            #torch.save(model.state_dict(), '/nas/longleaf/home/k8medlin/paper2/balboF1score/Gisette/'+title[i]+'_model_' +str([j])+'th_run')
            #capture Accepted Training Set 'markov chain' from LAST STEP combined sample 
            y_Markov = markovChain[:,-1]
            X_Markov = markovChain[:,0:-1]
            pd.DataFrame(y_Markov).to_csv(str(title[i])+'_y_acceptedTRAIN_'+str([j])+'th_run.csv')
            pd.DataFrame(X_Markov).to_csv(str(title[i])+'_X_acceptedTRAIN_'+str([j])+'th_run.csv')
            #print("Accepted training set saved")             
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
            
            """
        
            correctMarginMIN_np = np.array(correctNumInMarginMIN)
            correctMarginMIN.append(correctMarginMIN_np)
            correctMarginMAJ_np = np.array(correctNumInMarginMAJ)
            correctMarginMAJ.append(correctMarginMAJ_np)
            #store, export to CSV, and print j-th run's ALL metrics -- accuracy, effOne, loss, update ratio, preciion
            correctNumInMarginMIN_byRun = np.array(correctNumInMarginMIN,dtype=np.float32)
            pd.DataFrame(correctNumInMarginMIN_byRun).to_csv(str(title[i])+'_correctNumInMarginMIN_'+str([j])+'th_run.csv')
            correctNumInMarginMAJ_byRun = np.array(correctNumInMarginMAJ,dtype=np.float32)
            pd.DataFrame(correctNumInMarginMAJ_byRun).to_csv(str(title[i])+'_correctNumInMarginMAJ_'+str([j])+'th_run.csv')
            
            #fill vectors with ALL metrics -- accuracy, effOne, loss, update ratio, preciion
            
            all_effOneValid_np = np.array(all_F1Train)
            all_effOneValid.append(all_effOneValid_np)
            all_effOneValidMIN_np = np.array(all_minF1Train)
            all_effOneValidMIN.append(all_effOneValidMIN_np)
            all_effOneTest_np = np.array(all_F1Test)
            all_effOneTest.append(all_effOneTest_np)
            all_effOneTestMIN_np = np.array(all_minF1Test)
            all_effOneTestMIN.append(all_effOneTestMIN_np)
            all_jGrad_np = np.array(jayGrad_all)
            all_jGrad.append(all_jGrad_np)
            all_majSampleSize_np = np.array(majSampleSize_all)
            all_majSampleSize.append(all_majSampleSize_np)
            all_minSampleSize_np = np.array(minSampleSize_all)
            all_minSampleSize.append(all_minSampleSize_np)
            #store, export to CSV, and print j-th run's ALL metrics -- accuracy, effOne, loss, update ratio, preciion
            all_effOneTrain_byRun = np.array(all_F1Train,dtype=np.float32)
            pd.DataFrame(all_effOneTrain_byRun).to_csv(str(title[i])+'_all_F1Validation_'+str([j])+'th_run.csv')
            all_effOneTrainMIN_byRun = np.array(all_minF1Train,dtype=np.float32)
            pd.DataFrame(all_effOneTrainMIN_byRun).to_csv(str(title[i])+'_all_minF1Validation_'+str([j])+'th_run.csv')
            all_effOneTest_byRun = np.array(all_F1Test,dtype=np.float32)
            pd.DataFrame(all_effOneTest_byRun).to_csv(str(title[i])+'_all_F1Test_'+str([j])+'th_run.csv')
            all_effOneTestMIN_byRun = np.array(all_minF1Test,dtype=np.float32)
            pd.DataFrame(all_effOneTestMIN_byRun).to_csv(str(title[i])+'_all_minF1Test_'+str([j])+'th_run.csv')
            all_majSampleSize_byRun = np.array(majSampleSize_all,dtype=np.float32)
            pd.DataFrame(all_majSampleSize_byRun).to_csv(str(title[i])+'_all_majSampleSizeTrain_'+str([j])+'th_run.csv')   
            all_minSampleSize_byRun = np.array(minSampleSize_all,dtype=np.float32)
            pd.DataFrame(all_minSampleSize_byRun).to_csv(str(title[i])+'_all_minSampleSizeTrain_'+str([j])+'th_run.csv')  
            all_jayGrad_byRun = np.array(jayGrad_all,dtype=np.float32)
            #pd.DataFrame(all_jayGrad_byRun).to_csv(str(title[i])+'_all_L2normGradient_'+str([j])+'th_run.csv')  
            """ 
            
            
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
            #print("MAIN accepted_effOneTestMIN:",accepted_effOneTestMIN)
            
            
            accepted_jayGrad_np = np.array(jayGrad_accepted)
            accepted_jGrad.append(accepted_jayGrad_np)
            accepted_majSampleSize_np = np.array(majSampleSize_accepted)
            accepted_majSampleSize.append(accepted_majSampleSize_np)
            accepted_minSampleSize_np = np.array(minSampleSize_accepted)
            accepted_minSampleSize.append(accepted_minSampleSize_np)
            
            #print("trouble shooting precision:")
            accepted_precisionMIN_np = np.array(precisionMIN_accepted)
            accepted_precisionMIN.append(accepted_precisionMIN_np)
            #print("MAIN accepted_precisionMIN:",accepted_precisionMIN)
            accepted_precisionAVG_np = np.array(precisionAVG_accepted)
            accepted_precisionAVG.append(accepted_precisionAVG_np)
            #print("MAIN accepted_precisionAVG:",accepted_precisionAVG)
            
            
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
        
        #print(j,"-th run scores (last accepted step):")
        #print("Test f1 - average:", accepted_effOneTest_byRun[-1])
        #print("Test f1 - minority", accepted_effOneTestMIN_byRun[-1])
        
        
        
        """
        plt.hist(zMIN, bins=3,color=['#EE6666', '#3388BB', '#9988DD', '#EECC55', '#88BB44'])
        #plt.hist(marginalMIN, color=['palegreen'], label = 'correct marginal minority')
        #plt.hist(marginalMIN, bins=1, color = 'indigo', label = 'correct marginal minority')
        plt.xlabel('z_w')
        plt.ylabel('count')
        plt.title('z_w values of minority training data')
        plt.savefig(str(title[0])+"_minorityZwValues.png", dpi='figure', metadata=None,
                bbox_inches=None, pad_inches=0.1,
                facecolor='auto', edgecolor='auto',
                backend=None )
        plt.close()
        
        plt.hist(zMAJ, bins=5, color=['#EE6666', '#3388BB', '#9988DD', '#EECC55', '#88BB44'])
        #plt.hist(marginalMIN, color=['palegreen'], label = 'correct marginal minority')
        #plt.hist(marginalMIN, bins=1, color = 'indigo', label = 'correct marginal minority')
        plt.xlabel('z_w')
        plt.ylabel('count')
        plt.title('z_w values of majority training data')
        plt.savefig(str(title[0])+"_majorityZwValues.png", dpi='figure', metadata=None,
                bbox_inches=None, pad_inches=0.1,
                facecolor='auto', edgecolor='auto',
                backend=None )
        plt.close()
        """
                
        
        #print("MEAN Metrics attained by running testing data through model having been trained with", n_steps, "steps per run:")
        #print(str(title[i])+ ": MEAN metrics (accuracy, F1_score, precision, recall, auc, MCC) over", n_runs, "runs: ", np.mean( metrics, axis = 0))  
        #print(str(title[i])+ " STD of metrics over", n_runs, "runs: ", np.std( metrics, axis = 0)) 
        
        ######################## graphics #############################
        from matplotlib import cycler
        colors =  ['#EE6666', '#3388BB', '#9988DD',
                        '#EECC55', '#88BB44', '#FFBBBB', '#EE6666', '#3388BB', '#9988DD',
                        '#EECC55', '#88BB44', '#FFBBBB', '#EE6666', '#3388BB', '#9988DD',
                        '#EECC55', '#88BB44', '#FFBBBB', '#EE6666', '#3388BB', '#9988DD',
                        '#EECC55', '#88BB44', '#FFBBBB']
        markers = ["o", "s", "d", "*", "^", "P", "o", "s", "d", "*", "^", "P",\
                "o", "s", "d", "*", "^", "P", "o", "s", "d", "*", "^", "P"]
        plt.rc('axes', facecolor='white', edgecolor='grey',
        axisbelow=True, grid=True)
        plt.rc('grid', color='#E6E6E6', linestyle='solid')
        plt.rc('xtick', direction='out', color='grey')
        plt.rc('ytick', direction='out', color='grey')
        plt.rc('patch', edgecolor='#E6E6E6')
        plt.rc('lines', linewidth=7)
        plt.rcParams['xtick.labelsize']=22
        plt.rcParams['ytick.labelsize']=22
        
     
        
        fig, axes = plt.subplots(4, 1, figsize=(9, 15))
        plt.subplots_adjust(hspace=0.4)
        axes = axes.flatten()
        
        fig, axes = plt.subplots(4, 1, figsize=(9, 15))
        plt.subplots_adjust(hspace=0.4)
        axes = axes.flatten()
        for i in range(n_runs):
                if i==1 or i==2 or i==5 or i==10 or i==14:
                        x = accepted_index[i] 
                        y = 1 - accepted_effOneValidMIN[i]    
                        axes[0].scatter(x, y,label= '1 - Validation Minority F1', c=colors[i], marker=markers[i])
                        axes[0].plot(x, y,label= '1 - Train Minority F1', c=colors[i])

                        y = 1 - accepted_effOneValid[i]    
                        axes[1].scatter(x, y,label= '1 - Validation F1', c=colors[i], marker=markers[i])
                        axes[1].plot(x, y,label= '1 - Train F1', c=colors[i])
                        
                        y = accepted_effOneTestMIN[i]    
                        axes[2].scatter(x, y,label= 'Test Minority F1', c=colors[i], marker=markers[i])
                        axes[2].plot(x, y,label= 'Test Minority F1', c=colors[i])
                        
                        y = accepted_effOneTest[i]    
                        axes[3].scatter(x, y,label= 'Test F1', c=colors[i], marker=markers[i])
                        axes[3].plot(x, y,label= 'Test F1', c=colors[i])
                
                
        axes[0].set_title('1 - Minority F1 (Validation Data)', fontsize=25)
        #axes[2].set_title('Majority Data Loss', fontsize=25)
        axes[1].set_title('1 - Overall F1 (Validation Data)', fontsize=25)
        #axes[1].set_title('L2norm Gradient Loss (Inner Minimization)', fontsize=17)
        axes[2].set_title('Minority F1 (Testing Data)', fontsize=25)
        axes[3].set_title('Overall F1 (Testing)', fontsize=25)
        plt.grid()
        fig.suptitle(str(title[0]), fontsize=21)
        axes[3].set_xlabel('Accepted Steps', fontsize=22)
        plt.savefig(str(title[0])+"_metricsFromAcceptedSteps.png", dpi='figure', metadata=None,
                bbox_inches=None, pad_inches=0.1,
                facecolor='auto', edgecolor='auto',
                backend=None )
        plt.close()
        
        
        fig, axes = plt.subplots(1, 1, figsize=(9, 15))
        for i in range(n_runs):
                x = (1 - accepted_effOneValidMIN[i])  
                y = (1 - accepted_effOneValid[i])  
                #c = [1*n for n in range(len(x))]
                #print(c)
                #axes.scatter(x, y,label= 'Pareto Points', c=c, marker=markers[i], s=100)
                axes.scatter(x, y,label= 'Pareto Points', c=colors[i], marker=markers[i], s = 300)
                #axes.plot(x, y,label= 'Pareto Front', c=colors[i])
                
        
        plt.ylabel('1 - Overall F1 (Validation Data)', fontsize=22)
        plt.xlabel('1 - Minority F1 (Validation Data)', fontsize=22)
        plt.grid()
        fig.suptitle(str(title[0]), fontsize=21)
        plt.savefig(str(title[0])+"_ParetoPoints.png", dpi='figure', metadata=None,
                bbox_inches='tight', pad_inches=0.15,
                facecolor='auto', edgecolor='auto',
                backend=None )
        plt.close()
        
        fig, axes = plt.subplots(1, 1, figsize=(9, 15))
        for i in range(n_runs):
                x = (1 - accepted_effOneValidMIN[i][-1])  
                y = (1 - accepted_effOneValid[i][-1])  
                #c = [1*n for n in range(len(x))]
                #print(c)
                #axes.scatter(x, y,label= 'Pareto Points', c=c, marker=markers[i], s=100)
                axes.scatter(x, y,label= 'Pareto Front', c=colors[i], marker=markers[i], s = 300)
                #axes.plot(x, y,label= 'Pareto Front', c=colors[i])
                
        
        plt.ylabel('1 - Overall F1 (Validation Data)', fontsize=22)
        plt.xlabel('1 - Minority F1 (Validation Data)', fontsize=22)
        plt.grid()
        fig.suptitle(str(title[0]), fontsize=21)
        plt.savefig(str(title[0])+"_ParetoFront.png", dpi='figure', metadata=None,
                bbox_inches='tight', pad_inches=0.15,
                facecolor='auto', edgecolor='auto',
                backend=None )
        plt.close()
        """
        
        fig, axes = plt.subplots(2,1, sharey=True, tight_layout=True)
        #axes[0].hist(zMAJ, bins=5, color=['#EE6666'])
        print(zMAJ)
        #print(zMAJ.shape)
        axes[0].hist(zMAJ, bins=5, color=['#EE6666', '#3388BB', '#9988DD',
                        '#EECC55', '#88BB44'])
        axes[0].set_title('z_w values of majority training')
        axes[0].set_ylabel('count')
        #axes[1].hist(zMAJ, bins=5, color=['#EE6666'])
        axes[1].hist(zMIN, bins=5, color=['#EE6666', '#3388BB', '#9988DD',
                        '#EECC55', '#88BB44'])
        axes[1].set_xlabel('z_w')
        axes[1].set_ylabel('count')
        axes[1].set_title('z_w values of minority training')
        fig.suptitle(str(title[0]), fontsize=13)
        plt.savefig(str(title[0])+"_ZwValues.png", dpi='figure', metadata=None,
                bbox_inches=None, pad_inches=0.1,
                facecolor='auto', edgecolor='auto',
                backend=None )
        plt.close()
        
        """
        
        
        
        