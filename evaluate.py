import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_residuals(y, yhat):
    ''' 
    takes in a target column and the predicted/modeled regression col based on the targets column and creates a residual plot
    '''
    residual = np.subtract(yhat, y)

    #sns.scatterplot(data=train_scaled, x=single_var,y=y)

    sns.scatterplot(x=y,y=residual,color="red",label="residual of regression")
    plt.axhline(y=0)
    plt.legend()
    plt.title("scatter plot of residual of regression")
    plt.show()

    return

def regression_errors(y, yhat):
    ''' 
    returns the following values:
    sum of squared errors (SSE)
    explained sum of squares (ESS)
    total sum of squares (TSS)
    mean squared error (MSE)
    root mean squared error (RMSE)
    '''


    residual = np.subtract(yhat, y)
    residual2 = residual ** 2

    SSE = sum(residual2)
    print("SSE (sum of squared errors) = ", SSE)

    ESS = sum((yhat - y.mean())**2)
    print("ESS (explained sum of squares) = ", ESS)

    TSS = ESS + SSE ## TSS = TOTAL SUM OF SQUARES (EXPLAINED PLUS ERROR)
    print("TSS (Total sum of squares) = ", TSS)

    MSE = SSE/len(y)
    print("MSE (mean squared error) = ", MSE)

    RMSE = (MSE)**.5
    print("RMSE (root mean squared error) = ", RMSE)

    R2 = ESS/TSS ## RATIO
    print(f"Percent of variance in tax value explained by area = ", round(R2*100,1), "%")

    return(SSE, MSE, RMSE)

def baseline_mean_errors(y):
    ''' 
    takes in target column
    computes the SSE, MSE, and RMSE for the baseline model
    '''
    yhat_baseline = y.mean()

    residual_baseline = yhat_baseline - y
    
    residual_baseline2 = residual_baseline ** 2

    SSE_baseline = sum(residual_baseline2)
    print("SSE baseline = ", SSE_baseline)

    MSE_baseline = SSE_baseline/len(y)
    print("MSE baseline = ", MSE_baseline)

    RMSE_baseline = (MSE_baseline)**.5
    print("RMSE baseline = ", RMSE_baseline)

    return (SSE_baseline, MSE_baseline, RMSE_baseline)

def better_than_baseline(y, yhat): 
    '''
    takes in target column and regression column
    returns true if your model performs better than the baseline, otherwise false
    '''
    SSE, MSE, RMSE = regression_errors(y, yhat)
    SSE_baseline, MSE_baseline, RMSE_baseline = baseline_mean_errors(y)

    flag = True
    if SSE < SSE_baseline:
        print("model performed better than baseline on SSE")
    else:
        flag = False
        print("model performed worse than baseline on SSE")

    if MSE < MSE_baseline:
        print("model performed better than baseline on MSE")
    else:
        flag = False
        print("model performed worse than baseline on MSE")

    if RMSE < RMSE_baseline:
        print("model performed better than baseline on RMSE")
    else:
        flag = False
        print("model performed worse than baseline on RMSE")

    return