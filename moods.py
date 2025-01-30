import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import copy
import random
from sklearn.metrics import f1_score, confusion_matrix, classification_report, precision_score, recall_score
from imblearn.over_sampling import SVMSMOTE
from numpy import array
import math
import matplotlib.pyplot as plt
# select smallData for Ionosphere
# select data for Gisette Abalone Spambase Shuttle Connect4 

#from smallData import Our_Dataset
from data import Our_Dataset
from model import Basic_DNN
from collections import Counter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#accept/reject step
def random_coin(F1_t, F1_tMinusOne, F1min_t, F1min_tMinusOne):
    if ((1 - F1_t) <= (1-F1_tMinusOne)) and ((1-F1min_t) <= (1 - F1min_tMinusOne)):
        #print("accepted training 2 F1_t:",F1_t)
        #print("accepted training 2 F1min_t:",F1min_t)
        #print("training 2 F1_tMinusOne:",F1_tMinusOne)
        #print("training 2 F1min_tMinusOne:",F1min_tMinusOne)
        return True
    else:
        #print("rejected training 2 F1_t:",F1_t)
        #print("rejected training 2 F1min_t:",F1min_t)
        return False
    
def sample_size(l, M_t):
        for j in range(1,7):
            if M_t/(2**(j)) < l: 
                #print("sample_size M_t:", M_t/(2**(j)))
                break
        return j
    
random.seed()
def burnIn(model, Majority_data_training, Minority_data_training, X_state_test, y_label_test, n_steps=10, n_points=64):
    
    No_of_burnIn_steps = 1 
    m = len(Minority_data_training)
    lenMajBurn = math.ceil(0.5*m)
    lenMinBurn = m
    burnIn_sample = None
    for i in range(No_of_burnIn_steps):
        burnIn_stepT_MAJ = Majority_data_training[torch.randint(len(Majority_data_training), (lenMajBurn,))]
        burnIn_stepT_MIN = Minority_data_training[torch.randint(len(Minority_data_training), (lenMinBurn,))]
        # combine under-sampled majority data with all available minority data
        burnIn_stepT_sample = torch.cat((burnIn_stepT_MAJ,burnIn_stepT_MIN),0)  #current_sample
        if i==0:
            burnIn_sample= copy.deepcopy(burnIn_stepT_sample)
            burnIn_MAJ= copy.deepcopy(burnIn_stepT_MAJ)
            burnIn_MIN= copy.deepcopy(burnIn_stepT_MIN)
        else:
            burnIn_sample= torch.cat([burnIn_sample, burnIn_stepT_sample], 0)
            burnIn_MAJ= torch.cat([burnIn_MAJ, burnIn_stepT_MAJ], 0)
            burnIn_MIN= torch.cat([burnIn_MIN, burnIn_stepT_MIN], 0)
        y_label = burnIn_sample[...,:,-1].long()   
        X_state = burnIn_sample[...,:,0:-1]
        y_label_MAJ = burnIn_MAJ[...,:,-1].long()
        X_state_MAJ = burnIn_MAJ[...,:,0:-1]
        y_label_MIN = burnIn_MIN[...,:,-1].long()
        X_state_MIN = burnIn_MIN[...,:,0:-1]
        bs = 32 # batch size
        dataset = Our_Dataset(X_state,y_label)
        dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)  
        burnIn_norm2Gradient = model.fit(dataloader)  
    return burnIn_norm2Gradient, burnIn_sample, burnIn_MIN, burnIn_MAJ, y_label_MAJ, X_state_MAJ, y_label_MIN, X_state_MIN
    
def MOODS(Majority_data_training, Minority_data_training, No_of_steps, X_state_test, y_label_test, X_state_valid, y_label_valid, n_points=16):
    
                
    M = len(Majority_data_training)
    m = len(Minority_data_training)
    M_t = m
    burnIn_majority_sample = Majority_data_training[torch.randint(len(Majority_data_training), (n_points,))]
    burnIn_minority_sample = Minority_data_training[torch.randint(len(Minority_data_training), (n_points,))]
    burnIn_model_sample = torch.cat((burnIn_minority_sample,burnIn_majority_sample),0)   
    burnIn_X_state = burnIn_model_sample[...,:,0:-1]    
    model = Basic_DNN(burnIn_X_state, 1e-04).to(device) #####################GPU step#######################
    for i in range(1):
        burnIn_norm2Gradient, initial_sample, initial_MIN, initial_MAJ, initial_y_MAJ, initial_X_MAJ, initial_y_MIN, initial_X_MIN  = burnIn(model, Majority_data_training, Minority_data_training, X_state_test, y_label_test) #combined_burnIn_sample
    
    #print("burnIn initial_sample length:", len(initial_sample))
    #print("burnIn initial_MIN length:", len(initial_MIN))
    
    #Setting up variables and deep copies 
    index_all_steps = []   
    all_minF1Valid = []
    all_F1Valid = []
    all_minF1Test = []
    all_F1Test = []
    all_norm2Gradient = []
    all_majSampleSize = []
    all_minSampleSize = []
    markov_chain = []
    index_accepted_steps = []   
    accepted_accuracy = []    
    accepted_minF1Valid = []
    #last_minF1Train = []
    accepted_F1Valid = []
    #last_F1Train = []
    accepted_minF1Test = []
    accepted_F1Test = []
    accepted_norm2Gradient = []
    accepted_majSampleSize = []
    accepted_minSampleSize = []
    accepted_minPrecision = []
    accepted_avgPrecision = []
    
    #tMinusOne_loss = 10000 #initiating COMBINED loss
    
    
    ## Initiate outer loop
    tMinusOne_sample = copy.deepcopy(initial_sample) #initiating combined current sample with burnedIn combined sample
    tMinusOne_MAJ = copy.deepcopy(initial_MAJ)
    tMinusOne_MIN = copy.deepcopy(initial_MIN)
    tMinusOne_y_MAJ = copy.deepcopy(initial_y_MAJ)
    tMinusOne_X_MAJ = copy.deepcopy(initial_X_MAJ)
    tMinusOne_y_MIN = copy.deepcopy(initial_y_MIN)
    tMinusOne_X_MIN = copy.deepcopy(initial_X_MIN)
    tMinusOne_X = torch.cat( [tMinusOne_X_MAJ, tMinusOne_X_MIN], 0 )
    tMinusOne_Y = torch.cat( [tMinusOne_y_MAJ, tMinusOne_y_MIN], 0 )
    """
    tMinusOne_loss = model.loss(tMinusOne_X, tMinusOne_Y)
    tMinusOne_loss_mean = abs((1/len(tMinusOne_X))*tMinusOne_loss).float().sum().detach().numpy()
    tMinusOne_loss_MAJ = tMinusOne_loss[ :len(tMinusOne_X_MAJ)]
    tMinusOne_loss_MIN = tMinusOne_loss[(-len(tMinusOne_X_MIN)): ]
    tMinusOne_loss_MAJ_sum = abs(tMinusOne_loss_MAJ).float().sum().detach().numpy()
    tMinusOne_loss_MAJ_mean = abs((1/len(tMinusOne_X))*tMinusOne_loss_MAJ).float().sum().detach().numpy()
    tMinusOne_loss_MIN_mean = abs((1/len(tMinusOne_X))*tMinusOne_loss_MIN).float().sum().detach().numpy()
    """
    burnIn_F1min = model.effOneMinority(X_state_valid, y_label_valid)  
    burnIn_F1 = model.effOneAverage(X_state_valid, y_label_valid) 
    burnIn_TrainPRECmin = model.recallMinority(X_state_valid, y_label_valid)  
    burnIn_TrainPREC = model.recallAverage(X_state_valid, y_label_valid)
    
    burnIn_TestF1min = model.effOneMinority(X_state_test, y_label_test)  
    burnIn_TestF1 = model.effOneAverage(X_state_test, y_label_test) 
    burnIn_TestprecisionMIN = model.recallMinority(X_state_test, y_label_test)  
    burnIn_Testprecision = model.recallAverage(X_state_test, y_label_test)  
    
    #stepTMinusOne_effOne = burnIn_effOne
    #print("burnIn F1 - minority (testing data):", burnIn_TestF1min)
    #print("burnIn F1 - average (testing data):", burnIn_TestF1)
    
    #print("burnIn F1 - minority (validation data):", burnIn_F1min)
    #print("burnIn F1 - average (valiation data):", burnIn_F1)
    
    #print("burnIn initial_sample length - training data:", len(initial_sample))
    #print("burnIn initial_MIN length - training data:", len(initial_MIN))
    
    F1min_tMinusOne = burnIn_F1min
    F1_tMinusOne = burnIn_F1
    # initiate vectors for outputs -- important
    index_all_steps.append(0)
    #all_minorityLoss.append(Jmin_tMinusOne)
    #all_lT.append(L_tMinusOne)
    all_minF1Valid.append(F1min_tMinusOne)
    all_F1Valid.append(F1_tMinusOne)
    
    all_minF1Test.append(burnIn_TestF1min)
    all_F1Test.append(burnIn_TestF1)
    
    
    
    # initiate vectors for outputs -- other saved info
    all_norm2Gradient.append(burnIn_norm2Gradient)
    all_majSampleSize.append(len(tMinusOne_X_MAJ))
    all_minSampleSize.append(len(tMinusOne_X_MIN))
    #accepted_norm2Gradient.append(burnIn_norm2Gradient)
    #accepted_accuracy.append(burnIn_accuracy) 
    #accepted_majorityLoss_sum.append(tMinusOne_loss_MAJ_sum)
    #accepted_majSampleSize.append(len(tMinusOne_X_MAJ))
    #accepted_minSampleSize.append(len(tMinusOne_X_MIN))
    #accepted_minPrecision.append(burnIn_precisionMIN)
    #accepted_avgPrecision.append(burnIn_precisionAVG)
    
    #set-up probabiliites by index
    
    #set up probabilities
    probabilities = torch.from_numpy(np.ones(len(Majority_data_training))/len(Majority_data_training))
    #print("length probabilities vector before for loop:", len(probabilities))
    #print("shape of majority data before for loop:", Majority_data_training.shape)
    #print("probabilities head:",probabilities)
    
    index = array(list(range(0,len(Majority_data_training))))
    #print("index at creation:", len(index))
    
    #Outer loop steps
    for i in range(1, No_of_steps):
        
        #print("accepted minority:",len(tMinusOne_MIN))
        #print("accepted majority:",len(tMinusOne_MAJ))
        #print("accepted total:", len(tMinusOne_sample))
        
        if len(Majority_data_training) == 0:
            print("no training data remains. break")
            break
        #if len(Majority_data_training) == M_t == 1: M_t = 0
        
        if M_t > len(Majority_data_training):
            z = sample_size(len(Majority_data_training),M_t) 
            M_t = math.ceil(M_t/2**(z+1))
            #print("M_t > len(Majority_data_training). new (decreased) M_t:", M_t)
        
        #print("M_t <= len(Majority_data_training). M_t (not decreased):", M_t)
        #if np.max(probabilities)<10e-6: M_t = 0
        #if len(tMinusOne_MAJ)>=M: M_t=0
        ##########################
    
        # under sample majority data randomly
        probabilities = probabilities/(torch.sum(probabilities))
        if torch.max(probabilities)< 10e-10: 
            print("probabilities break")
            break
        sampling_index = probabilities.multinomial(num_samples = M_t)
        #print("sampling_index shape:", sampling_index.shape)
        #print("Majority_data_training shape:", Majority_data_training.shape)
        majority_sample = Majority_data_training[sampling_index]
        #print("majority_sample length:", len(majority_sample))
        #minority_sample = Minority_data_training[np.random.choice(len(Minority_data_training), (lenMinBurn,), replace = False)]
        #majority_sample = Majority_data_training[torch.randint(len(Majority_data_training), (M_t,))]
        #sm = SVMSMOTE(random_state=2,k_neighbors=1,sampling_strategy='minority')
         ############### change this for oversampling synthetic data
        
        
        if len(accepted_F1Valid)==0:
            stepT_minority = Minority_data_training
            stepT_majority = tMinusOne_MAJ
            stepT_sample = torch.cat((stepT_minority, stepT_majority),0) 
            stepT_X = stepT_sample[...,:,0:-1]
            stepT_y = stepT_sample[...,:,-1].long()
            
            
            stepT_majority_X = stepT_majority[...,:,0:-1]
            stepT_majority_y = stepT_majority[...,:,-1].long()
            stepT_minority_X = stepT_minority[...,:,0:-1]
            stepT_minority_y = stepT_minority[...,:,-1].long()
            
            stepT_X = stepT_sample[...,:,0:-1]
            stepT_Y = stepT_sample[...,:,-1].long()
            
            stepT_MAJ = stepT_majority
            stepT_MIN = stepT_minority
            
        if len(accepted_F1Valid)>=0 and len(tMinusOne_MIN) >= (len(tMinusOne_MAJ) + len(majority_sample)):
            stepT_minority = Minority_data_training
            stepT_majority = torch.cat((tMinusOne_MAJ, majority_sample),0)
            stepT_sample = torch.cat((stepT_minority, stepT_majority),0) 
            stepT_X = stepT_sample[...,:,0:-1]
            stepT_y = stepT_sample[...,:,-1].long()
            
            stepT_majority_X = stepT_majority[...,:,0:-1]
            stepT_majority_y = stepT_majority[...,:,-1].long()
            stepT_minority_X = stepT_minority[...,:,0:-1]
            stepT_minority_y = stepT_minority[...,:,-1].long()
            
            stepT_X = stepT_sample[...,:,0:-1]
            stepT_Y = stepT_sample[...,:,-1].long()
            
            stepT_MAJ = stepT_majority
            stepT_MIN = stepT_minority
            
        if len(accepted_F1Valid)>0 and len(tMinusOne_MIN) < (len(tMinusOne_MAJ) + len(majority_sample)): 
            stepT_preSample = torch.cat( (tMinusOne_MIN, tMinusOne_MAJ, majority_sample),0  )
            stepT_X_pre = stepT_preSample[...,:,0:-1]
            stepT_y_pre = stepT_preSample[...,:,-1]
            sm = SVMSMOTE(random_state=2,k_neighbors=1,sampling_strategy='minority')
            
            
            stepT_X, stepT_y = sm.fit_resample(stepT_X_pre, stepT_y_pre)
            #stepT_y_SVM = stepT_y_SVM.astype(np.float32)
            #print("stepT_y_SVM.dtype:", stepT_y_SVM.dtype)
            
            #('Resampled dataset shape %s' % Counter(stepT_y))

            synthetic_index = len(tMinusOne_MIN) + len(tMinusOne_MAJ) + len(majority_sample)
    
            synthetic_minority_X = stepT_X[(synthetic_index):len(stepT_X)]
            synthetic_minority_y = stepT_y[(synthetic_index):len(stepT_y)]
            synthetic_minority_y.reshape(-1,1)
            
            
            #print("SVM_minority_X type:",synthetic_minority_X.dtype)
            #print("SVM_minority_y type:",synthetic_minority_y.dtype)
            #print("size SVM_minority_X:",len(synthetic_minority_X))
            #print("checking SVM lengths")
            #print("length previously accepted minority:",len(tMinusOne_MIN))
            #print("length previously accepted majority:", len(tMinusOne_MAJ))
            #print("length new majority_sample:", len(majority_sample))
            #print("length synthetic minority:",len(synthetic_minority_y))
            #print("total training length:",len(stepT_y))
            #print("SVM_minority_X:",SVM_minority_X)
            #print("SVM_minority_y:",SVM_minority_y)
            
            synthetic_minority = torch.from_numpy(np.append(synthetic_minority_X,synthetic_minority_y.reshape(-1,1),axis=1))
            
            #SVM_minority = torch.cat( (SVM_minority_X, SVM_minority_y), axis=1 )
            stepT_sample = torch.from_numpy(np.append(stepT_X,stepT_y.reshape(-1,1),axis=1))
            
            #print("stepT_X_SVM.dtype:", stepT_X.dtype)
            #print("stepT_y_SVM.dtype:", stepT_y.dtype)
            
            #print("size SVM_minority:",len(synthetic_minority))
            #print("SVM_minority:",synthetic_minority)
            #print("size stepT_sample:",len(stepT_sample))
    
            stepT_MAJ = torch.cat( (tMinusOne_MAJ, majority_sample),0  )
            stepT_MIN = torch.cat( (tMinusOne_MIN, synthetic_minority),0  )
            
            stepT_majority_X = stepT_MAJ[...,:,0:-1]
            stepT_majority_y = stepT_MAJ[...,:,-1].long()
            stepT_minority_X = stepT_MIN[...,:,0:-1]
            stepT_minority_y = stepT_MIN[...,:,-1].long()
            
            stepT_sample = torch.cat( (stepT_MIN, stepT_MAJ),0  )
            #stepT_sample = torch.cat( (tMinusOne_sample, stepT_sample),0  )
            stepT_X = stepT_sample[...,:,0:-1]
            stepT_Y = stepT_sample[...,:,-1].long()
            #print("after svm smote step", i, "there are this many datapoints:")
            #print("proposed minority:",len(stepT_MIN))
            #print("proposed majority:",len(stepT_MAJ))
            #print("proposed total:", len(stepT_sample))
        
    
        # separate features from labels
        y_label = stepT_sample[...,:,-1].long()  
        X_state = stepT_sample[...,:,0:-1]    
        #initiate the model for inner loop step
        bs = 32 
        dataset = Our_Dataset(X_state,y_label)
        dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)  
        norm2Gradient = model.fit(dataloader)   ##############  Inner-loop ##################
        
        #test data F1 and accuracy
        stepT_accuracy = model.accuracy(X_state_test, y_label_test)   
        stepT_F1minTest = model.effOneMinority(X_state_test, y_label_test) 
        stepT_F1avgTest = model.effOneAverage(X_state_test, y_label_test)
        
        
        #train data loss
        stepT_F1minValid = model.effOneMinority(X_state_valid, y_label_valid) 
        stepT_F1avgValid = model.effOneAverage(X_state_valid, y_label_valid)
        F1_t = stepT_F1avgValid
        F1min_t = stepT_F1minValid 
        
        #initial t-1 minority loss and overall loss
   
        if i==1: 
            burnIn_TrainF1min = F1min_t
            burnIn_TrainF1avg = F1_t
        #if L_t < 10e-15: L_t = 10e-15
        ### Index and Collect Scores ###
        
        index_all_steps.append(i)            
        all_F1Valid.append(F1_t)
        all_minF1Valid.append(F1min_t)
        all_norm2Gradient.append(norm2Gradient)
        all_F1Test.append(stepT_F1avgTest)
        all_minF1Test.append(stepT_F1minTest)
        all_majSampleSize.append(len(stepT_majority_X))
        all_minSampleSize.append(len(stepT_minority_X))
        
    
        ############## minority F1 maximization #######################################
        
        #print("accepted_minF1Valid:",accepted_minF1Valid)
            
            
            
        if i == 1: 
            F1min_tMinusOne = 0.0
        if 1 < i < 10:
            #F1min_tMinusOne = min(np.mean(avg_F1minT[-np.ceil(len(avg_F1minT)/2).astype(int): ]), np.mean(avg_F1minT[-np.ceil(len(avg_F1minT)).astype(int): ]))
            #print("accepted_minF1Train:", accepted_minF1Train)
            F1min_tMinusOne = accepted_minF1Valid[-1] + 10e-6
        if 10 <= i:
            #F1min_tMinusOne = min(np.mean(avg_F1minT[-5: ]), np.mean(avg_F1minT[-np.ceil(len(avg_F1minT)).astype(int): ]))
            F1min_tMinusOne = accepted_minF1Valid[-1] + 10e-6
            
         ############## average F1 maximization #######################################

        
        #print("accepted_F1Valid:",accepted_F1Valid)
        if i == 1: 
            F1_tMinusOne = 0.0
        if 1 < i < 10:
            #F1_tMinusOne = min(np.mean(avg_F1T[-np.ceil(len(avg_F1minT)/2).astype(int): ]), np.mean(avg_F1T[-np.ceil(len(avg_F1T)).astype(int): ]))
            F1_tMinusOne = accepted_F1Valid[-1] + 10e-10
        if 10 <= i:
            #F1_tMinusOne = min(np.mean(avg_F1T[-5: ]), np.mean(avg_F1T[-np.ceil(len(avg_F1T)).astype(int): ]))
            F1_tMinusOne = accepted_F1Valid[-1] + 10e-10
        
        ###################### accept/reject step ##############################
        if random_coin(F1_t, F1_tMinusOne, F1min_t, F1min_tMinusOne):
            print("###########################    step ", i, " ACCEPTED   ###########################")
            
            F1min_tMinusOne = F1min_t
            F1_tMinusOne = F1_t
            tMinusOne_sample = stepT_sample
            tMinusOne_MIN = stepT_MIN
            tMinusOne_MAJ = stepT_MAJ
            tMinusOne_y_MAJ = stepT_majority_y
            tMinusOne_X_MAJ = stepT_majority_X
            tMinusOne_y_MIN = stepT_minority_y
            tMinusOne_X_MIN = stepT_minority_X
            index_accepted_steps.append(i)
            
            
            ##################### extracting Z values from softmax: from majority trianing data ####################
              
            #softmaxMAJ0 = model.forward(stepT_majority_X.to(device)).detach().cpu()[:,0]
            zMAJ = model.zValue(stepT_majority_X.to(device)).detach().cpu()[:,1]
            #print("zMAJ:",zMAJ)
            yHatMAJ = model.predict(stepT_majority_X.to(device)).detach().cpu()
            #print("yHatMAJ:", yHatMAJ)
            
            #print("non-adjusted zMAJ[i]",zMAJ[i])
            #print("average zMAJ:",np.mean(zMAJ))
            svZmaj = -12
            for i in range(len(zMAJ)): 
                if zMAJ[i] <= 0.5 and zMAJ[i] > svZmaj:
                    svZmaj = zMAJ[i]
            #print("svZmaj:",svZmaj) 
            
            ##################### extracting Z values from softmax: from minority trianing data ####################
            
            #softmax0 = model.forward(stepT_minority_X.to(device)).detach().cpu()[:,0]
            #softmax1 = model.forward(stepT_minority_X.to(device)).detach().cpu()[:,1]
            zMIN = model.zValue(stepT_minority_X.to(device)).detach().cpu()[:,1]
            #print("zMIN:",zMIN)
            yHat = model.predict(stepT_minority_X.to(device)).detach().cpu()
            #print("yHatMIN:",yHat)
            
            svZmin = 12
            for i in range(len(zMIN)):
                if zMIN[i] >= 0.5 and zMIN[i] < svZmin:
                    svZmin = zMIN[i]
            #print("svZmin:",svZmin)  
            
            ##################################### collectingdatapoints with Z on wrong side of 0.5 ########################
             
            numWrongSideMIN = 0
            for i in range(len(zMIN)):
                if zMIN[i] < 0.5:
                    numWrongSideMIN = numWrongSideMIN +1
            #print("numWrongSideMIN:", numWrongSideMIN)

                
            numWrongSideMAJ = 0
            for i in range(len(zMAJ)):
                if zMAJ[i] > 0.5:
                    numWrongSideMAJ = numWrongSideMAJ +1
            #print("numWrongSideMAJ:", numWrongSideMAJ)
            

            ##################################### collectingdatapoints with Z within margin of 0.5 ########################
            
            remaining_index = np.delete(index,sampling_index,axis=0)
            remaining_majority = Majority_data_training[remaining_index,:]
            remaining_probabilities = probabilities[remaining_index]
            Majority_data_training = remaining_majority
            index = array(list(range(0,len(Majority_data_training))))
            probabilities = remaining_probabilities
            
            if i > 1: M_t = M_t + 1
            print("M_t after step accepted:", M_t)
            #TRAINING data metrics
            accepted_minF1Valid.append(F1min_t)
            #last_minF1Train = F1min_t
            accepted_F1Valid.append(F1_t)
            #last_F1Train = F1_t
            #print("accepted_minF1Train:", accepted_minF1Train)
            accepted_norm2Gradient.append(norm2Gradient)
            accepted_majSampleSize.append(len(stepT_majority_X))
            accepted_minSampleSize.append(len(stepT_minority_X))
            accepted_accuracy.append(stepT_accuracy) 
            accepted_F1avgTest = model.effOneAverage(X_state_test,y_label_test)  
            accepted_F1Test.append(accepted_F1avgTest)
            accepted_F1minTestScore = model.effOneMinority(X_state_test,y_label_test)  
            accepted_minF1Test.append(accepted_F1minTestScore)
            accepted_minPrecisionScore = model.precisionMinority(X_state_test,y_label_test)  
            accepted_minPrecision.append(accepted_minPrecisionScore)
            accepted_avgPrecisionScore = model.precisionAverage(X_state_test,y_label_test)  
            accepted_avgPrecision.append(accepted_avgPrecisionScore)
            
        else:
            print("###########################    step ", i, " REJECTED   ###########################")
            if M_t > 10: M_t = M_t - 1
            probabilities[sampling_index] = probabilities[sampling_index]/2
            print("M_t after step rejected:", M_t)
            
    markov_chain = tMinusOne_sample.numpy()
    minSampleSize = m
    print("############################    last step outputs   ############################")
    print("############################    minority   ############################")
    print("numInMIN:",len(stepT_minority_X))
    print("numWrongSideMIN:", numWrongSideMIN)        
    print("average zMIN:",zMIN.mean())
    print("variance zMIN:",zMIN.var())
    print("median zMIN:", zMIN.median())
    print("svZmin:",svZmin)
    
    print("############################    majority   ############################")
    print("numInMAJ:",len(stepT_majority_X))
    print("numWrongSideMAJ:", numWrongSideMAJ)
    print("average zMAJ:",zMAJ.mean())
    print("variance zMAJ:",zMAJ.var())
    print("median zMAJ:", zMAJ.median())
    print("svZmaj:",svZmaj)
   
    print("############################    F1 scores   ############################")
    print("accepted (valid) F1min:", F1min_t)
    print("accepted (valid) F1:", F1_t)
    print("accepted (test) F1min:", accepted_F1minTestScore)
    print("accepted (test) F1:", accepted_F1avgTest)
    print("#############################     end of run      #############################")
    """
    if numInMarginMAJ>0 and numInMarginMIN>0: 
        plt.hist(marginalMAJ, bins=1, color = 'green', label = str(percentCorrectInMarginMAJ) + 'majority correct')
        plt.hist(marginalMIN, bins=1, color = 'orange', label = str(percentCorrectInMarginMIN)+'minority correct')
        plt.legend(loc="upper left")
        plt.xlabel('z_w')
        plt.ylabel('count')
        plt.title('z_w values on the margins')
        plt.show()
            
    if numInMarginMIN>0 and numInMarginMAJ==0: 
        plt.hist(marginalMIN, bins=1, color = 'orange', label = str(percentCorrectInMarginMIN)+'minority correct')
        plt.legend(loc="upper left")
        plt.xlabel('z_w')
        plt.ylabel('count')
        plt.title('z_w values on the minority margin')
        plt.show()
    """
    return model, markov_chain, minSampleSize, index_all_steps, all_F1Valid, all_minF1Valid, \
        all_F1Test, all_minF1Test, \
            all_norm2Gradient, all_majSampleSize, \
                all_minSampleSize, index_accepted_steps, accepted_accuracy, accepted_F1Valid, accepted_minF1Valid, \
                    accepted_F1Test, accepted_minF1Test, accepted_avgPrecision, accepted_minPrecision, \
                        accepted_norm2Gradient, accepted_majSampleSize, accepted_minSampleSize, \
                            zMAJ, zMIN