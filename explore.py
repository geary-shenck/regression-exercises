import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats





def univariate_explore(df):
    ''' 
    takes in dataframe, and puts out a histogram of each category, binning relatively low
    '''
    plt.figure(figsize=(25, 5))
    for i, col in enumerate(df.columns.tolist()): # List of columns
        plot_number = i + 1 # i starts at 0, but plot nos should start at 1
        plt.subplot(1,len(df.columns.tolist()), plot_number) # Create subplot.
        plt.title(col) # Title with column name.
        df[col].hist(bins=10) # Display histogram for column.
        plt.grid(False) # Hide gridlines.







def plot_variable_pairs(df,num_vars):
    ''' 
    that accepts a dataframe and numerical variables as input and plots all of the 
    pairwise relationships along with the regression line for each pair.
    '''


    l=1
    for col1 in num_vars:
        for col2 in num_vars:
            if not num_vars.index(col2) >= num_vars.index(col1):
                plt.figure(figsize=(6, 6))
    #            plt.subplot(1,3,l)
                l +=1
                sns.regplot(    data=df,
                                x=col1,
                                y=col2,
                                line_kws={"color": "red"})
                plt.title(f"{col2} value by {col1} value")
                plt.show()

    l=1
    for col1 in num_vars:
        for col2 in num_vars:
            if not num_vars.index(col2) >= num_vars.index(col1):
                plt.figure(figsize=(6, 6))
    #            plt.subplot(1,3,l)
                l +=1
                sns.lmplot(     data=df,
                                x=col1,
                                y=col2,
                                line_kws={"color": "red"},
                                x_estimator=np.mean)
                plt.title(f"{col2} value by {col1} value")
                plt.show()

    l=1
    for col1 in num_vars:
        for col2 in num_vars:
            if not num_vars.index(col2) >= num_vars.index(col1):
                plt.figure(figsize=(6, 6))
    #            plt.subplot(1,3,l)
                l +=1
                sns.jointplot(  data=df,
                                x=col1,
                                y=col2,
                                kind="reg",
                                joint_kws={'line_kws':{'color':'red'}})
                plt.title(f"{col2} value by {col1} value")
                plt.show()
    return






def plot_categorical_and_continuous_vars(df,num_vars,cat_vars):
    ''' 
    input(dataframe, list of numerical, list of categorical)
    accepts your dataframe and the name of the columns that hold 
    the continuous and categorical features and outputs 3 different plots 
    for visualizing a categorical variable and a continuous variable.
    '''

    plt.figure(figsize=(25, 15))

    i = 0
    l = 0
    for col1 in num_vars:
        i += 1
        j = 0

        for col2 in cat_vars:
            j += 1
            l += 1

            plt.subplot(len(num_vars),len(cat_vars),l)
            plot_order = df[col2].sort_values(ascending=False).unique()
            sns.boxplot(x=col2, y=col1, data=df, order = plot_order)
            plt.title(f"value of {col1} organized by {col2}")
    plt.show()

    plt.figure(figsize=(25, 15))
    i = 0
    l = 0
    for col1 in num_vars:
        i += 1
        j = 0

        for col2 in cat_vars:
            j += 1
            l += 1

            plt.subplot(len(num_vars),len(cat_vars),l)
            plot_order = df[col2].sort_values(ascending=False).unique()
            sns.stripplot(x=col2, y=col1, data=df, order = plot_order)
            plt.title(f"value of {col1} organized by {col2}")
    plt.show()

    plt.figure(figsize=(25, 15))
    i = 0
    l = 0
    for col1 in num_vars:
        i += 1
        j = 0

        for col2 in cat_vars:
            j += 1
            l += 1

            plt.subplot(len(num_vars),len(cat_vars),l)
            plot_order = df[col2].sort_values(ascending=False).unique()
            sns.violinplot(x=col2, y=col1, data=df, order = plot_order)
            plt.title(f"value of {col1} organized by {col2}")
    return






def heatmap_corr(train):
    ''' 
    takes in dataframe and returns a heatmap based on correlation
    '''
    plt.figure(figsize=(12, 6))
    kwargs = {'alpha':1,
            'linewidth':5, 
            'linestyle':'--',
            'linecolor':'white'}

    sns.heatmap(    train.corr(),
                    #map="YlGnBu", 
                    cmap="Spectral",
                    mask=(np.triu(np.ones_like(train.corr(),dtype=bool))),
                    annot=True,
                    vmin=-1, 
                    vmax=1, 
                    #annot=True,
                    **kwargs
                    )
    plt.title("Correlation value for features")
    plt.show()










def cat_and_num_explore_plot(train,cat,num):
    ''' 
    takes in dataframe and string values of the categorical and numerical columns 
    to run a means TTest on and plot a visualization
    '''

    alpha = .05

    for cat_1 in train[cat].unique():
        for cat_2 in train[cat].unique():
            if not train[cat].unique().tolist().index(cat_2) >= train[cat].unique().tolist().index(cat_1):
                H0 = f"{num} of {cat}{cat_1} has identical average values to {num} of other {cat}{cat_2}"
                Ha = f"{num} of {cat} is not equal to {num} of other {cat}"
                print("-----------------------------")
                #compare variances to know how to run the test
                stat,pval = stats.levene(train[train[cat] == cat_1][num],train[train[cat] == cat_2][num])
                stat,pval
                if pval > 0.05:
                    equal_var_flag = True
                    print(f"we can accept that there are equal variance in these two groups with {round(pval,2)} certainty Flag=T",'stat=%.5f, p=%.5f' % (stat,pval))
                else:
                    equal_var_flag = False
                    print(f"we can reject that there are equal variance in these two groups with {round((1-pval),2)} certainty Flag=F",'stat=%.5f, p=%.5f' % (stat,pval))


                t, p = stats.ttest_ind( train[train[cat] == cat_1][num], train[train[cat] == cat_2][num], equal_var = equal_var_flag )

                if p > alpha:
                    print("\n We fail to reject the null hypothesis (",(H0) , ")",'t=%.5f, p=%.5f' % (t,p))
                else:
                    print("\n We reject the null Hypothesis (", '\u0336'.join(H0) + '\u0336' ,")",'t=%.5f, p=%.5f' % (t,p))

    plt.figure(figsize=(12,6))
    plt.title(f"Density of {cat} and {num}")


    plt.ylabel("Density of those who {num}")
    plt.yticks([],[])


    sns.kdeplot(train[train[cat] == train[cat].unique()[0]][num],label=f"{train[cat].unique()[0].astype(int)}")
    sns.kdeplot(train[train[cat] == train[cat].unique()[1]][num],label=f"{train[cat].unique()[1].astype(int)}")
    sns.kdeplot(train[train[cat] == train[cat].unique()[2]][num],label=f"{train[cat].unique()[2].astype(int)}")

    plt.axvline(train[train[cat] == train[cat].unique()[0]][num].mean(),
                color="blue",
                ls=":",
                label=f"mean for {train[cat].unique()[0].astype(int)}")
    plt.axvline(train[train[cat] == train[cat].unique()[1]][num].mean(),
                color="red",
                ls="-",
                label=f"mean for {train[cat].unique()[1].astype(int)}")
    plt.axvline(train[train[cat] == train[cat].unique()[2]][num].mean(),
                color="green",
                ls="--",
                label=f"mean for {train[cat].unique()[2].astype(int)}")

    plt.xlabel(f"num")
    plt.legend()
    plt.show()








def pearsonr_corr_explore_plot(train,num1,num2):
    ## putting tax value and taxes yearly into a pearsonr and then graphing it for a visual result as a result
    ## of the heat map above highlighting a good possibility of a relation


    H0 = f"That the distributions underlying the samples of {num1} and {num2} are unrelated"
    Ha = f"That the distributions underlying the samples of {num2} and {num2} are related"
    alpha = .05

    r, p = stats.pearsonr(train[num1],train[num2])

    plt.figure(figsize=(10,6))
    plt.scatter( train[num1], train[num2])
    b, a = np.polyfit(train[num1], train[num2], deg=1)
    plt.plot(train[num1], a + b * train[num1], color="k", lw=2.5,label="Regression Line")
    plt.title(f'Correlation value, (r={round(r,1)})', size=16)
    plt.legend()
    plt.show()

    print('r =', r)

    if p > alpha:
        print("\n We fail to reject the null hypothesis (",(H0) , ")",'p=%.5f' % (p))
    else:
        print("\n We reject the null Hypothesis (", '\u0336'.join(H0) + '\u0336' ,")", 'p=%.5f' % (p))
