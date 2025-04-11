%%(pca,24Î¬)

%%(²»ÓÃpca)

clc;
clear;

S_Feature = load('F:\Codes\LBP-TOP\4Frames_Features\Sa\Sa_Sa_Feature.mat');
S_Feature = S_Feature.Sa_Sa_Feature;

P_Feature = load('F:\Codes\LBP-TOP\4Frames_Features\Sa\Ha_Sa_Feature.mat');
P_Feature = P_Feature.Ha_Sa_Feature;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Train_Subject_No = zeros(22,21);
Test_Subject_No  = zeros(22,1);

Train_Subject_No(1,:) = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]; 
Test_Subject_No(1,:) = 1;
    
Train_Subject_No(2,:) = [1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]; 
Test_Subject_No(2,:) = 2;

Train_Subject_No(3,:) = [1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]; 
Test_Subject_No(3,:) = 3;

Train_Subject_No(4,:) = [1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]; 
Test_Subject_No(4,:) = 4;

Train_Subject_No(5,:) = [1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]; 
Test_Subject_No(5,:) = 5;

Train_Subject_No(6,:) = [1,2,3,4,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]; 
Test_Subject_No(6,:) = 6;

Train_Subject_No(7,:) = [1,2,3,4,5,6,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]; 
Test_Subject_No(7,:) = 7;

Train_Subject_No(8,:) = [1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22]; 
Test_Subject_No(8,:) = 8;

Train_Subject_No(9,:) = [1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22]; 
Test_Subject_No(9,:) = 9;

Train_Subject_No(10,:) = [1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,21,22]; 
Test_Subject_No(10,:) = 10;

Train_Subject_No(11,:) = [1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,19,20,21,22]; 
Test_Subject_No(11,:) = 11;

Train_Subject_No(12,:) = [1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22]; 
Test_Subject_No(12,:) = 12;

Train_Subject_No(13,:) = [1,2,3,4,5,6,7,8,9,10,11,12,14,15,16,17,18,19,20,21,22]; 
Test_Subject_No(13,:) = 13;

Train_Subject_No(14,:) = [1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20,21,22]; 
Test_Subject_No(14,:) = 14;

Train_Subject_No(15,:) = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20,21,22]; 
Test_Subject_No(15,:) = 15;

Train_Subject_No(16,:) = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19,20,21,22]; 
Test_Subject_No(16,:) = 16;

Train_Subject_No(17,:) = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,19,20,21,22]; 
Test_Subject_No(17,:) = 17;

Train_Subject_No(18,:) = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,20,21,22]; 
Test_Subject_No(18,:) = 18;

Train_Subject_No(19,:) = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21,22]; 
Test_Subject_No(19,:) = 19;

Train_Subject_No(20,:) = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21,22]; 
Test_Subject_No(20,:) = 20;

Train_Subject_No(21,:) = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22]; 
Test_Subject_No(21,:) = 21;

Train_Subject_No(22,:) = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]; 
Test_Subject_No(22,:) = 22;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Accuracy1 = zeros(1, 22);
Accuracy2 = zeros(1, 22);

for K = 1 : 22
    
    K
    
    for H = 1 : length(Train_Subject_No(K,:)) 
        
        if (H==1)
            
            S_Train_Feature = S_Feature{1,Train_Subject_No(K,H)};
            
            P_Train_Feature = P_Feature{1,Train_Subject_No(K,H)};
            
        else
            
            S_Train_Feature = [S_Train_Feature; S_Feature{1,Train_Subject_No(K,H)}];
            
            P_Train_Feature = [P_Train_Feature; P_Feature{1,Train_Subject_No(K,H)}];
        
        end
        
    end 
    
    [NumSample, D] = size(S_Train_Feature);
    S_Train_Label = ones(NumSample,1);
    [NumSample, D] = size(P_Train_Feature);
    P_Train_Label = -1*ones(NumSample,1);
    
    Train_Feature = [S_Train_Feature; P_Train_Feature];
    Train_Label = [S_Train_Label; P_Train_Label];
     
    S_Test_Feature = S_Feature{1, Test_Subject_No(K,1)};
    P_Test_Feature = P_Feature{1, Test_Subject_No(K,1)};
    Test_Feature = [S_Test_Feature; P_Test_Feature];
    
    [NumSample, D] = size(S_Test_Feature);
    S_Test_Label = ones(NumSample,1);
    [NumSample, D] = size(P_Test_Feature);
    P_Test_Label = -1*ones(NumSample,1);
    Test_Label = [S_Test_Label; P_Test_Label];
    
%     [coeff, score, latent, tsquared, explained, mu] = pca(Train_Feature);
%     coeff = coeff(:, 1:24);
%     Train_Mean = mean(Train_Feature,1);
%     Train_Mean = repmat(Train_Mean, 42, 1);
%     Train_Feature = Train_Feature - Train_Mean;
%     Train_Feature = Train_Feature * coeff; 
%     
%     Test_Feature = Test_Feature - Train_Mean(1 : 2, :);
%     Test_Feature = Test_Feature * coeff; 
    
    svmModel = fitcsvm(Train_Feature, Train_Label, 'KernelFunction','linear','ClassNames',[-1,1]);
    Test_Pred_Label = predict(svmModel, Test_Feature);
     
    Accuracy1(1, K) = sum(Test_Pred_Label == Test_Label)/length(Test_Label);
    
    Index = find(Test_Label == 1);
    Temp = Test_Pred_Label(Index,1);
    
    if (length(find(Temp==1))/length(Index))>0.5
       
        Accuracy2(1, K) = Accuracy2(1, K) + 1;  
    
    end
    
    Index = find(Test_Label == -1);
    Temp = Test_Pred_Label(Index,1);
    
    if (length(find(Temp==-1))/length(Index))>0.5
       
        Accuracy2(1, K) = Accuracy2(1, K) + 1;  
    
    end
    
    Accuracy2(1, K) = Accuracy2(1, K)/2;
    
end