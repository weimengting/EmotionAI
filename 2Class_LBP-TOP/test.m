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