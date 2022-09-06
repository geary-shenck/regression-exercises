
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import acquire
from sklearn.preprocessing import MinMaxScaler

def split_function(df,target):
    ''' 
    splits a dataframe and returns train, test, and validate dataframes
    '''
    train,test = train_test_split(df,test_size= .2, random_state=123,stratify = df[target])
    train,validate = train_test_split(train,test_size= .25, random_state=123,stratify = train[target])

    print(f"prepared df shape: {df.shape}")
    print(f"train shape: {train.shape}")
    print(f"validate shape: {validate.shape}")
    print(f"test shape: {test.shape}")

    return train, test, validate

def split_continous(df):
    ''' 
    splits a dataframe and returns train, test, and validate dataframes
    '''
    train,test = train_test_split(df,test_size= .2, random_state=123)
    train,validate = train_test_split(train,test_size= .25, random_state=123)

    print(f"prepared df shape: {df.shape}")
    print(f"train shape: {train.shape}")
    print(f"validate shape: {validate.shape}")
    print(f"test shape: {test.shape}")

    return train, test, validate



#import warnings
#warnings.filterwarnings("ignore")


def prep_zillow(df0):
    ''' 
    no arguements
    gets - acquires zillow data with lesson guided restrictions (for features)
    replaces any whitespace with nans
    renames featues
    drops nans (large enough data set)
    returns the prepared dataframe
    '''
# get the data and review the data

    #df0 = acquire.get_zillow_single_fam()
    ## datatypes look good with 2,152,863 records

    ## start the clean up
    ## remove cells with whitespace and replace with NaN in a new working dataframe

    df = df0.replace(r'^\s*$', np.nan, regex=True)
    df = df.rename(columns = {"bedroomcnt":"bedrooms",
                                "bathroomcnt":"bathrooms",
                            "calculatedfinishedsquarefeet":"area",
                            "taxvaluedollarcnt":"tax value",
                            "taxamount":"taxes yearly",
                            "yearbuilt":"year built"
                            })


    ## drop n/a and review
    df = df.dropna()

    ##ranges on these look very crazy, diving deeper
    #for col in df.columns:
    #    print("range of",col, ": {:,}".format((df[col].describe()["max"] - df[col].describe()["min"]).astype(int)), "({:,}".format(df[col].describe()["max"]), "max - min {:,})".format(df[col].describe()["min"]))

    return df


def minmaxscaler(train,validate,test,target_cols):
    ''' 
    takes in train,validate,test, and list for targeting
    scales target cols for set
    '''

    scaler = MinMaxScaler()

    train[target_cols] = scaler.fit_transform(train[target_cols]) ##only use fit_transform for training, after that use transform (equations are created)
    validate[target_cols] = scaler.transform(validate[target_cols])
    test[target_cols] = scaler.transform(test[target_cols])
    return train,validate,test