# A function to execute a t-test and print results

def my_ttest(data1='input1',data2='input2',equalvar=True,alpha=0.05,show_pivot=False,\
             group1_var='none',group2_var='none',\
             group1_val='none',group2_val='none',\
             test1_var='none',test2_var='none'):

    from scipy import stats
    from scipy.stats import norm
    from scipy.stats import t

    # Basic error checking, require names of data objects are in quotes (they should be strings)
    if (type(data1) != str) | (type(data2) != str):
        print("The data1 and data2 input parameters need to be in quotes")
        return

    # Create a copy of the input data and name them data1IN and data2IN
    code_to_create_data1=f'data1IN = {data1}.copy()'
    exec(code_to_create_data1,globals())

    code_to_create_data2=f'data2IN = {data2}.copy()'
    exec(code_to_create_data2,globals())

    # We can only do the pivot table analysis if data1 and data2 are both data frames, and the SAME data frame
    if isinstance(data1IN, pd.DataFrame) & isinstance(data2IN, pd.DataFrame):    
        # Run the pivot table and save output to a data frame named df_byregion
        # This can ONLY be done if data1 == data2 and group1_var == group2_var and test1_var==test2_var
        if (data1==data2) & (group1_var==group2_var) & (test1_var==test2_var):
            df_pivot = data1IN.pivot_table(index=group1_var, values=[test1_var], aggfunc=['mean','std','sem','count'],margins=True)
            # Rename pivot table 
            df_pivot.columns = ['mean','std','sem','count']
            df_pivot['tstat'] = df_pivot['mean']/df_pivot['sem']

            if show_pivot == True:
                display(df_pivot)

        # Create group1_data and group2_data objects, either the pooled sample if group1_var=none or subsets
        if (group1_var == 'none') & (test1_var != test2_var):
            # Use the whole sample, this occurs when test1_var and test2_var are different and group1_var is 'none'
            group1_data = pd.Series(data1IN[test1_var])
            group2_data = pd.Series(data2IN[test2_var])

            # set values of group1_val and group2_val if not specified for output, use input data names
            group1_val=test1_var
            group2_val=test2_var
        else:
            # Create subsets of data for group1 and group2 meeting value criteria
            group1_data = data1IN[(data1IN[group1_var] == group1_val)][test1_var]
            group2_data = data2IN[(data2IN[group2_var] == group2_val)][test2_var]

    else:
        # If input data are lists, convert to series
        if isinstance(data1IN, list):
            group1_data = pd.Series(data1IN)
        elif isinstance(data1IN, pd.Series):
            group1_data = data1IN.copy()
        else:
            print("Check the data types of your input data.  The types can be two data frames, two lists, or two series.")
            return

        if isinstance(data2IN, list):
            group2_data = pd.Series(data2IN)
        elif isinstance(data2IN, pd.Series):
            group2_data = data2IN.copy()
        else:
            print("Check the data types of your input data.  The types can be two data frames, two lists, or two series.")
            return
        # set values of group1_val and group2_val if not specified for output, use input data names
        group1_val=data1
        group2_val=data2
        # if inputs are series or lists, there is no need for: group1_val, group2_val, test1_var, or test2_var

    # calculate means, standard errors, number of observations
    mean1=group1_data.mean()
    mean2=group2_data.mean()
    sem1=group1_data.sem()
    sem2=group2_data.sem()
    n1=group1_data.count()
    n2=group2_data.count()

    # Calculate degrees of freedom
    # under equal variance assumption
    degf_same = n1+n2-2
    # under unequal or different variance assumption use welch's adjusted degrees of freedom
    degf_diff = (sem1**2+sem2**2)**2/((sem1**4/(n1-1))+(sem2**4/(n2-1)))

    # conduct the t-test
    res = stats.ttest_ind(group1_data, group2_data, equal_var=equalvar)

    # The first element returned is the t-statistic, the second is the p-value of the test
    tstat = res[0]
    pval  = res[1]

    # Because tstat = (mean1-mean2)/std_err, we can calculate the std_err as (mean1-mean2)/tstat
    # We are not provided the std error in the ttest_ind method, so calculate it ourselves
    std_err = (mean1-mean2)/tstat

    critical_z2T = norm.ppf(1-alpha/2,loc=0,scale=1)
    critical_z1T = norm.ppf(1-alpha,loc=0,scale=1)    
    
    if equalvar == True:
        critical_t2T = t.ppf(1-alpha/2,degf_same,loc=0,scale=1)
        critical_t1T = t.ppf(1-alpha,degf_same,loc=0,scale=1)    
    else:
        critical_t2T = t.ppf(1-alpha/2,degf_diff,loc=0,scale=1)
        critical_t1T = t.ppf(1-alpha,degf_diff,loc=0,scale=1)    
        
    # Print output for the user
    print(f"---------------------------------------------------------------------------\n\
    In the comparison between:\n\
    \tgroup 1: {group1_val} (mean={mean1:0.4f}, n={n1}, std err={sem1:0.04f})\n\
    \tgroup 2: {group2_val} (mean={mean2:0.4f}, n={n2}, std err={sem2:0.04f})\n\n\
    The results using the t-distribution are*:\n\
    \t\tdiff in means:            {mean1-mean2:0.4f}\n\
    \t\tstd err:                  {std_err:0.04f}\n\
    \t\tt-statistic:              {tstat:0.3f}\n\
    \t\tp-value (one tail):       {pval/2:0.3f}\n\
    \t\tp-value (two tail):       {pval:0.3f}\n\n\
    Critical values for alpha = {alpha}, {(1-alpha)*100:0.0f}% Confidence Level:\n\
    \t\tone tail t critical value: {critical_t1T:0.3f}\n\
    \t\ttwo tail t critical value: {critical_t2T:0.3f}\n\
    \t\tone tail z critical value: {critical_z1T:0.3f}\n\
    \t\ttwo tail z critical value: {critical_z2T:0.3f}\n\
    ")

    # Print the assumption of the variance equality and corresponding degrees of freedom
    if equalvar == True:
        print(f"\t*These results assume equal variances with {degf_same} degrees of freedom")
    else:
        print(f"\t*These results assume unequal variances with {degf_diff:0.0f} degrees of freedom calculated with Welch's method")

    print('---------------------------------------------------------------------------\n\n')
