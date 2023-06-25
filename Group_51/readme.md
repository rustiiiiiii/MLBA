# Group_51

# How to use the code

$ python3 MLBA_ANN.py

> Data was clean

# The following best results were obtained while training and testing on the train_data

## Used SVM

    Accuracy was 0.67

## Used MLP

Accuracy was 0.69

> Since we were getting better results on MLP model, we used it for our submission output file.

# Feature generation models were

    >ANOVA 
    >F Regression
    >Mutual Information Gain
    >Variance Threhold
    >Recursive Feature Elimination
    >Mutual Information Regession

    After trying out the above feature genration models, we found ANOVA  to be the best model

# Models Used

    > ANN with MLP Classifier
    > SVM


#Final score on kaggle: 0.56086

#Step to run the code
    In the "Mlba_Final_code.ipynb" choose the "Run all" option to run all the cells. 

    The code will ask the user to input the path of train and test csv files

    After providing the path, the code will generate a output file containing the prediction called "output_final_anova.csv".

    
#libraries required: sklearn, oandas,sys,os and numpy.