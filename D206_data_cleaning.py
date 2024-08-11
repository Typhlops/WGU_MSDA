########################################################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import missingno as msno
import seaborn as sns
import fancyimpute
import copy
import pdb
from sklearn import linear_model
# from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
from sklearn.decomposition import PCA
from imblearn.ensemble import BalancedRandomForestClassifier
from itertools import combinations
from datasketch import MinHash, MinHashLSH

'''
Note: The data cleaning and modification starts at line 825. Everything above that line consists of the data assessment 
methods and calls used in part C., many of which are used in the cleaning procedures. Adjust 'show_all_output' below to
values 7, 8, or 9 for the most relevant terminal output. The creation of variant 'Job' columns and methods used is
between lines 410 to 558.
'''

# If True, runs entire file with all output. Set to integer between 1 and 9 inclusive to run corresponding section with
# significant output and suppress non-critical output from other sections with significant output.
show_all_output = 9
# Suppresses breakpoints. Comment out to move one section at a time.
pdb.set_trace = lambda: None
# Adjust or comment out if displaying all columns is unwieldy
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 16)

df_churn = pd.read_csv('D206-churn-files/churn_raw_data.csv')
df_backup = df_churn.copy(deep=True)
df_churn_numerical = df_churn.select_dtypes(exclude=['object'])
df_churn_categorical = df_churn.select_dtypes(['object'])

all_columns = ['Unnamed: 0', 'CaseOrder', 'Customer_id', 'Interaction', 'City', 'State', 'County', 'Zip', 'Lat', 'Lng',
               'Population', 'Area', 'Timezone', 'Job', 'Children', 'Age', 'Education', 'Employment', 'Income',
               'Marital', 'Gender',
               'Churn', 'Outage_sec_perweek', 'Email', 'Contacts', 'Yearly_equip_failure', 'Techie', 'Contract',
               'Port_modem',
               'Tablet', 'InternetService', 'Phone', 'Multiple', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
               'TechSupport',
               'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'PaymentMethod', 'Tenure', 'MonthlyCharge',
               'Bandwidth_GB_Year',
               'item1', 'item2', 'item3', 'item4', 'item5', 'item6', 'item7', 'item8']

# Found through print(list(df_churn.columns[df_churn.isna().sum() > 0]))
contain_nulls_columns = ['Children', 'Age', 'Income', 'Techie', 'InternetService', 'Phone', 'TechSupport', 'Tenure',
                         'Bandwidth_GB_Year']

identifier_columns = ['Unnamed: 0', 'CaseOrder', 'Customer_id', 'Interaction']

large_cat_columns = ['City', 'County', 'Zip', 'Job']
large_cat_columns_hierarch = ['City', 'County', 'Zip', 'Job_hierarch']

small_cat_columns = ['State', 'Area', 'Timezone', 'Education', 'Employment', 'Marital', 'Gender', 'Contract',
                     'InternetService', 'PaymentMethod']

yes_no_columns = ['Churn', 'Techie', 'Port_modem', 'Tablet', 'Phone', 'Multiple', 'OnlineSecurity', 'OnlineBackup',
                  'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling']

item1_to_8_columns = ['item1', 'item2', 'item3', 'item4', 'item5', 'item6', 'item7', 'item8']

small_int_columns = ['Children', 'Age', 'Email', 'Contacts', 'Yearly_equip_failure']

continuous_columns = ['Lat', 'Lng', 'Population', 'Income', 'Outage_sec_perweek', 'Tenure', 'MonthlyCharge',
                      'Bandwidth_GB_Year']

pdb.set_trace()


########################################################################################################################

# Adapted from DataCamp's "Dealing with Missing Data in Python" ch. 2 (Suraj Donthi).
# https://campus.datacamp.com/courses/dealing-with-missing-data-in-python/does-missingness-have-a-pattern?ex=7.
# Accessed 2024.
# Creates dummy values for null values in each column. The dummy values are shifted away from the distribution of data
# to allow for fast visual inspection of scatter plots vs other variables.
def fill_dummy_values(df, scaling_factor):
    df_dummy = df.copy(deep=True)
    for col_name in df_dummy.columns:
        col = df_dummy[col_name]
        col_null = col.isnull()
        num_nulls = col_null.sum()
        if num_nulls > 0:
            col_max = col.max()
            col_min = col.min()
            col_range = col_max - col_min
            dummy_values = (np.random.rand(num_nulls) - 2) * scaling_factor * col_range + col_min
            df_dummy.loc[col_null, col_name] = dummy_values
    return df_dummy


# Creates a scatter plot with non-null data and dummy values between two variables.
def adjusted_null_scatter(df, scaling_factor, col_x, col_y):
    df_dummy = fill_dummy_values(df, scaling_factor)
    nullity = df[col_x].isnull() | df[col_y].isnull()
    df_dummy.plot(x=col_x, y=col_y, kind='scatter', alpha=0.5, c=nullity, cmap='rainbow')
    plt.show()


# For each column in columns, prints the number of unique values, percent null values (if nonzero), and value counts.
def display_value_counts(df, columns, verbose=True):
    expected_total = df.__len__()
    for col in columns:
        null_count = df[col].isnull().sum()
        counts_w_nulls = df[col].value_counts(dropna=False)
        print("----------------------")
        print(f"{col} contains {len(counts_w_nulls)} unique values")
        if null_count > 0:
            print(
                f"{col} has {null_count} null values, which is {round(null_count / expected_total * 100, 3)}% of {expected_total}.")
        print(counts_w_nulls) if verbose else None
        print("----------------------\n")
    print("\n")


# Searches for outliers using IQR method. Prints results and returns a dictionary {'col': outlier points} of outliers.
def outlier_search(df, columns):
    df_outliers_dict = {}
    for col in columns:
        col_stats = df[col].describe()
        q25 = col_stats['25%']
        q75 = col_stats['75%']
        lower_bound = q25 - 1.5 * (q75 - q25)
        upper_bound = q75 + 1.5 * (q75 - q25)
        print("----------------------")
        print(col_stats)
        print(f"\nIQR test for outliers has a lower bound of {round(lower_bound, 3)} and an upper bound"
              f" of {round(upper_bound, 3)}")
        df_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        if not df_outliers.empty:
            print(f"There are {len(df_outliers)} outliers.\n")
            print(df_outliers[col])
        else:
            print(f"There are no outliers in the column {col}.")
        df_outliers_dict[col] = df_outliers
        print("----------------------\n")
    return df_outliers_dict


pdb.set_trace()


########################################################################################################################


def display_all_values(mode=True, df=df_churn):
    display_value_counts(df, identifier_columns, verbose=False)
    print("Verifying values in 'CaseOrder' are all unique ('False' indicates not a duplicate):")
    print(df.duplicated(subset=['CaseOrder'], keep=False).value_counts())
    print(f"\nNumber of values in agreement between columns 'Unnamed: 0' and 'CaseOrder': "
          f"{(df['Unnamed: 0'] == df['CaseOrder']).sum()}\n")

    if mode:
        display_value_counts(df, large_cat_columns)
        print(df[large_cat_columns].describe())
        print(df[['City', 'County', 'Job']].describe())
    else:
        display_value_counts(df, large_cat_columns_hierarch)
        print(df[large_cat_columns_hierarch].describe())

    display_value_counts(df, small_cat_columns)
    print(df[small_cat_columns].describe())

    display_value_counts(df, yes_no_columns)
    print(df[yes_no_columns].describe())

    display_value_counts(df, item1_to_8_columns)
    print(f"Mean values for item1 through item8:\n{df[item1_to_8_columns].mean()}\n")
    item1_to_8_outliers = outlier_search(df, item1_to_8_columns)

    display_value_counts(df, small_int_columns)
    small_int_outliers = outlier_search(df, small_int_columns)

    display_value_counts(df, continuous_columns)
    continuous_outliers = outlier_search(df, continuous_columns)

    return item1_to_8_outliers, small_int_outliers, continuous_outliers


def info_print():
    print("Dataframe info:")
    df_churn.info()
    print(f"\nInspecting number of nulls in columns:\n{df_churn.isna().sum()[df_churn.isna().sum() > 0]}\n")
    print(f"Looking for duplicates: {df_churn.duplicated().sum()}\n")


info_print()
item1_to_8_outliers, small_int_outliers, continuous_outliers = display_all_values()

pdb.set_trace()
########################################################################################################################


# 'InternetService' has a valid entry of 'None'. The number of null 'InternetService' values after importing the csv is
# precisely equal to the number of 'None' values in Excel for that column. This will be adjusted later in the section
# that adjusts the data.

print(f"Verifying number of entries in 'Zip' with number of digits other than 5: "
      f"{len(df_churn.loc[(df_churn['Zip'] >= 100000) + (df_churn['Zip'] < 10000), 'Zip'])}")
print(f"Verifying number of entries in 'Zip' with number of digits exactly 5: "
      f"{len(df_churn.loc[(df_churn['Zip'] >= 10000) * (df_churn['Zip'] < 100000), 'Zip'])}")
print(f"Verifying outage values less than 0: "
      f"{len(df_churn['Outage_sec_perweek'][df_churn['Outage_sec_perweek'] < 0.01])}")

pdb.set_trace()


########################################################################################################################


# Plots a histogram of z-scores for each column in columns.
def zscore_hists(df, columns):
    for col in columns:
        new_label = col + "_z_score"
        df[new_label] = stats.zscore(df[col][~df[col].isnull()])
        plt.hist(df[new_label])
        plt.xlabel(col)
        plt.show()
        df.drop(columns=[new_label], inplace=True)


# Creates plots across variables in 'columns'. Mode can be histogram, pair plot, or scatter plot.
def plots_by_column(columns, df=df_churn, mode='hist', y_scatter='Age'):
    if mode == 'hist':
        for coln in columns:
            nonnull_df = df.loc[~df[coln].isna()]
            plt.hist(nonnull_df[coln])
            plt.xlabel(coln)
            plt.show()
    elif mode == 'pairplot':
        sns.pairplot(df[columns], diag_kind='hist')
    elif mode == 'scatter':
        for coln in columns:
            plt.scatter(x=df[coln], y=df[y_scatter])
            plt.xlabel(coln)
            plt.show()
    else:
        print(f"Invalid mode selection")


# Creates a series of histograms for each column in 'columns' for visual comparison when the null values in a separate
# column are either isolated or excluded. For instance, a histogram of 'Yes' vs 'No' in 'Phone' on rows where 'Income'
# is not null then the same histogram on rows where 'Income' is null.
def null_histogram_gen(columns, null_columns=contain_nulls_columns, df=df_churn):
    for coln in columns:
        plt.ylabel('Count')
        for null_coln in null_columns:
            nonnull_df = df.loc[(~df[null_coln].isna()) * (~df[coln].isna())]
            plt.hist(nonnull_df[coln])
            plt.xlabel((f"{coln} with nulls removed from {null_coln}"))
            plt.show()
            if null_coln is not coln:
                null_df = df.loc[df[null_coln].isna() * (~df[coln].isna())]
                plt.hist(null_df[coln])
                plt.xlabel((f"{coln} with only nulls from {null_coln}"))
                plt.show()


# Convenience function
def outliers_and_plots(df, columns, mode='hist', y_scatter='Age'):
    for column in columns:
        outlier_search(df, [column])
        pdb.set_trace()
        plots_by_column([column], df, mode, y_scatter)
        pdb.set_trace()
        zscore_hists(df, [column])
        pdb.set_trace()


pdb.set_trace()
########################################################################################################################


if (show_all_output == True) or (show_all_output == 1):
    null_histogram_gen(['InternetService', 'Phone'])
    pdb.set_trace()
    outliers_and_plots(df_churn, continuous_columns)
    pdb.set_trace()
    outliers_and_plots(df_churn, small_int_columns)
    pdb.set_trace()
    outliers_and_plots(df_churn, item1_to_8_columns)
    pdb.set_trace()

pdb.set_trace()


########################################################################################################################


# Searches for duplicates across n variables pulled from 'columns'. Variables in 'columns' that should always be
# present when searching for duplicates can be placed in the 'constants' array which defaults to None. Prints any
# duplicates found across all (len(columns) choose n) combinations. To improve inspection of possible duplicate
# entries, inspect_cols adds identifying columns such as 'City', 'Job', and others when printing out duplicate
# candidates. mode=0 prints all duplicates when verbose=True, including trivial matches on n null values.
# mode=1 requires at least one non-null match, mode=2 requires at least two, etc.
def n_column_duplicate_tester(columns, n, constants=None, mode=2, df=df_churn, verbose=True):
    pd.set_option('display.max_rows', 200)
    columns_copy = copy.deepcopy(columns)
    inspect_cols = ['City', 'State', 'Lat', 'Population', 'Age', 'Income', 'Job', 'Education', 'Employment', 'Marital',
                    'Gender', 'MonthlyCharge', 'Bandwidth_GB_Year']
    sub_default = []
    if constants is not None:
        for elem in constants:
            sub_default.append(elem)
            columns_copy.remove(elem)
    total_columns = len(columns_copy)
    combos = combinations(range(total_columns), n)  # Choose n variables from total_columns without replacement
    for comb in combos:
        sub = copy.deepcopy(sub_default)
        for idx in comb:
            sub.append(columns_copy[idx])
        duplicates = df.duplicated(subset=sub, keep=False)
        if duplicates.any():
            display_cols = copy.deepcopy(inspect_cols)
            for item in sub:
                if item not in inspect_cols:
                    display_cols.append(item)
            df_results_raw = df.loc[duplicates, display_cols]
            print(f"Found {duplicates.value_counts().iloc[1]} duplicates between {sub}.")
            if mode > 0:
                upper_bound = n + 1 - mode + (len(constants) if constants is not None else 0)
                null_mask = df.loc[duplicates, sub].isnull().sum(axis=1) < upper_bound
                df_results = df_results_raw[null_mask]
                print(f"{len(df_results_raw) - len(df_results)} of them contain null values in {upper_bound} or more "
                      f"of the above {upper_bound - 1 + mode} columns.")
                print(f"{len(df_results)} of them contain non-null values in at least {mode} of the above "
                      f"{upper_bound - 1 + mode} columns.\n")
            else:
                if mode != 0:
                    print(f"Invalid mode selected. Please use a nonnegative integer <= {n}. Proceeding with mode=0.")
                    mode = 0
                df_results = df_results_raw
            if verbose and (len(df_results) > 0):
                output_lt = f"Displaying results with at least {mode} non-null matches in the above columns:"
                output_all = f"Displaying results with non-null matches in all of the above columns:"
                output_str = output_lt if mode < (n + (len(constants) if constants is not None else 0)) else output_all
                print(f"{output_str}\n{df_results.sort_values(by=inspect_cols[0])}\n")
    pd.set_option('display.max_rows', 10)
    print(f"Completed searching for n = {n}\n")


duplicate_candidates = ['City', 'Zip', 'Job', 'Age', 'Income', 'Outage_sec_perweek', 'Tenure', 'MonthlyCharge',
                        'Bandwidth_GB_Year']
print(f"\nUsing duplicate_candidates = {duplicate_candidates}\n")
print("Trying n = 3:\n")
n_column_duplicate_tester(duplicate_candidates, 3, None, 3)
print("Trying n = 4:\n")
n_column_duplicate_tester(duplicate_candidates, 4, ['City'], 4)
print("Trying n = 5:\n")
n_column_duplicate_tester(duplicate_candidates, 5, ['City'], 5)

pdb.set_trace()


########################################################################################################################


# For each unique value in col1, prints tuples of unique values in col2 that occur for that value of col1 if there's
# more than one such tuple. For example, col1='Zip' and col2='Timezone' would ideally print nothing outside of unusual
# circumstances where a zip code is split between timezones.
def cross_validator(col1, col2, df=df_churn):
    for value in df[col1].unique():
        unique_values = df[df[col1] == value][col2].unique()
        if len(unique_values) > 1:
            print(f"{value} {col2} values: {unique_values}")
    print("complete")


if (show_all_output == True) or (show_all_output == 2):
    cross_validator('Zip', 'Timezone')
    cross_validator('City', 'Zip')

pdb.set_trace()


########################################################################################################################


# Searches for anomalies by grouping the data according to group_columns then calculating the aggregate mean and
# standard deviation on agg_columns. Returns an aggregate dataframe where the standard deviation exceeds a threshold
# 'thresh'. For example, grouping by 'County' and 'State' should refer to a relatively limited geographical region in
# terms of latitude and longitude. Input errors and other outliers, which might be valid in very large counties,
# are returned if the threshold is sensitive enough.
def agg_validator(df, agg_columns, group_columns, thresh):
    df_agg = (df[(agg_columns + group_columns)].groupby(by=group_columns).agg(['mean', 'std']))
    # consider using if thresh is None: thresh = 0.7 * df_agg[(agg_columns[i], 'mean')]
    agg_expr = (df_agg[(agg_columns[0], 'std')] > thresh)
    for i in range(1, len(agg_columns)):
        agg_expr += (df_agg[(agg_columns[i], 'std')] > thresh)
    return df_agg[agg_expr]


geo_validation = agg_validator(df_churn, ['Lat', 'Lng'], ['State', 'County'], 1.0)
print(geo_validation)

pdb.set_trace()


########################################################################################################################


# Uses MinHashing to detect potential duplicates in strings.
def string_match_lsh(str_list, thresh=0.5, verbose=True):
    min_hashes = []
    lsh = MinHashLSH(threshold=thresh, num_perm=128)
    near_duplicates = set()

    for string in str_list:
        mh = MinHash()
        for word in string.split():
            mh.update(word.encode('utf8'))
        min_hashes.append(mh)

    for i, m in enumerate(min_hashes):
        lsh.insert(i, m)

    for i, m in enumerate(min_hashes):
        matches = lsh.query(m)
        for j in matches:
            if i != j:
                near_duplicates.add(tuple(sorted((str_list[i], str_list[j]))))

    near_duplicates_list = sorted(list(near_duplicates))
    if verbose:
        print(f"There are {len(near_duplicates_list)} potential duplicate strings:\n{near_duplicates_list}\n")
    return near_duplicates_list


# For a string (primarily in 'Job') containing a comma, returns a new string with no comma, the first word at the
# end, and adjusted capitalization. For instance, 'Accountant, public' becomes 'Public accountant'. Creates entries in
# a dictionary for its inverse.
def str_swap_format(s, dict):
    reversed_s = s
    if ', ' in s:
        words = s.split(', ')
        words[0] = words[0].lower()
        words[1] = words[1].capitalize()
        reversed_s = ' '.join(reversed(words))
        dict[reversed_s] = s
    return reversed_s


# Inverts str_swap_format
def inverse_str_swap_format(s, dict):
    original_s = s
    if s in dict:
        original_s = dict[s]
    return original_s


# Returns a new string that better reflects its categorization and hierarchy when it ends with a string found in
# substr_list. substr_list might be a list of common job fields like 'Engineer' and 'Manager'. A string such as
# 'Mechanical engineer' outputs 'Engineer, mechanical' if 'Engineer' is found in substr_list.
def hierarchical_substr(s, substr_list):
    new_s = s
    for item in substr_list:
        if (s.endswith((" " + item.lower())) and (',' not in s)):
            words = s.split(' ')
            first_term = item.lower().split()[0]
            cut_index = words.index(first_term)
            words[0] = words[0].lower()
            new_s = item + ', ' + ' '.join(words[:cut_index])
            break
    return new_s


# Uses string transformation methods and MinHashing above to look for duplicates in 'Job'. Creates new columns that
# consolidate near-duplicates (e.g. 'Engineer, mechanical' and 'Mechanical engineer') with different formatting.
# The new columns are dropped as function terminates (drop=True by default), set drop=False to retain them.
def job_inspect(job_col='Job', df=df_churn, drop=True, verbose=True):
    job_list = sorted(df['Job'].unique())
    frequent_job_fields = sorted(set(job.split(', ')[0] for job in job_list if ', ' in job))
    if verbose:
        print(f"\nUnique values in {job_col}: {job_list}")
        print(f"\nFrequently occurring broad categories in {job_col}: {frequent_job_fields}")
    job_dict = {}
    job_filt = job_col + '_filt'
    job_filt_inv = job_col + '_filt_inv'
    job_hierarch = job_col + '_hierarch'
    df[job_filt] = df[job_col].apply(lambda s: str_swap_format(s, job_dict))
    df[job_filt_inv] = df[job_filt].apply(lambda s: inverse_str_swap_format(s, job_dict))
    df[job_hierarch] = df[job_col].apply(lambda s: hierarchical_substr(s, frequent_job_fields))

    print(f"\nNumber of unique values in {job_col, job_filt, job_filt_inv, job_hierarch}: "
          f"{[len(df[job_col].unique()), len(df[job_filt].unique()), len(df[job_filt_inv].unique()), len(df[job_hierarch].unique())]}\n")
    set1 = set(df[job_filt_inv].unique())
    set2 = set(df[job_hierarch].unique())
    set3 = set1.difference(set2)
    set4 = set2.difference(set1)
    if verbose:
        print(f"Set({job_filt_inv} unique values) - Set({job_hierarch} unique values):\n{sorted(set3)}\n")
        print(f"Set({job_hierarch} unique values) - Set({job_filt_inv} unique values):\n{sorted(set4)}\n")
        print(f"Set({job_filt_inv} unique values):\n{sorted(set1)}\n")
        print(f"Set({job_hierarch} unique values):\n{sorted(set2)}\n")

    if verbose:
        print(f"\nMinHash duplicate search on unique values of {job_col}:")
        string_match_lsh(df[job_col].unique(), 0.6)
        print(f"\nMinHash duplicate search on unique values of {job_filt}:")
        string_match_lsh(df[job_filt].unique(), 0.6)
        print(f"\nMinHash duplicate search on unique values of {job_filt_inv}:")
        string_match_lsh(df[job_filt_inv].unique(), 0.6)
        print(f"\nMinHash duplicate search on unique values of {job_hierarch}:")
        string_match_lsh(df[job_hierarch].unique(), 0.6)

    if drop:
        df.drop(columns=[job_filt], inplace=True)
        df.drop(columns=[job_filt_inv], inplace=True)
        df.drop(columns=[job_hierarch], inplace=True)

    return job_list, frequent_job_fields


# Prints lists for each broad category (career field) in str_list (e.g. 'Engineer') with matches in unique_str_list
# that could belong to that category as its child. 'Engineer' in str_list will display subfields such as 'Engineer,
# civil' and 'Mechanical engineer' found in unique_str_list.
def show_frequent_career_cats(str_list, unique_str_list):
    for field in str_list:
        lower_field = [job for job in unique_str_list if job.endswith((" " + field.lower()))]
        upper_field = [job for job in unique_str_list if job.startswith(field.capitalize())]
        combined_field = [field, lower_field, upper_field]
        if combined_field is not None:
            print(combined_field)
    print("\n")


pdb.set_trace()
########################################################################################################################


if (show_all_output == True) or (show_all_output == 3):
    job_list_0, frequent_job_fields_0 = job_inspect('Job', df_churn, False, True)
    show_frequent_career_cats(frequent_job_fields_0, job_list_0)
    pdb.set_trace()

    new_job_list = sorted(df_churn['Job_hierarch'].unique())
    new_frequent_job_fields = sorted(set(job.split(', ')[0] for job in new_job_list if ', ' in job))
    show_frequent_career_cats(new_frequent_job_fields, new_job_list)

    print(agg_validator(df_churn, ['Income'], ['Job_hierarch'], 10000))
else:
    job_list_0, frequent_job_fields_0 = job_inspect('Job', df_churn, False, False)

pdb.set_trace()


########################################################################################################################


# Produces summary statistics between columns containing null values and numerical comparison_columns within the
# dataframe. Differences between measures in stat_columns on rows containing null values in a given null_column vs
# rows with non-null values in that null_column are recorded in a summary dataframe df_stats. Differences above the
# corresponding thresh_values[i] are flagged as a potential noteworthy correlation and stored in a dictionary.
# Verbose = 0 is minimal printed output, 1 is moderate, 2 is extensive. Returns a dictionary of potential
# correlations with each null_column as a key. Also returns df_stats.
def null_inspector(df, null_columns, comparison_columns, thresh_values=[0.02, 0.02], stat_columns=['mean', 'std'],
                   verbose=0):
    stats_multi_index = pd.MultiIndex.from_product([comparison_columns, stat_columns])
    df_stats = pd.DataFrame(index=stats_multi_index, columns=null_columns)
    potential_correlations_master = {}
    for null_column in null_columns:
        potential_correlations = {}
        print(f"Starting '{null_column}'")
        null_stats = df[comparison_columns][df[null_column].isna()].describe()
        non_null_stats = df[comparison_columns][~df[null_column].isna()].describe()
        for compare_column in comparison_columns:
            if verbose == 2:
                print("\n------------------\n")
                print(f"Statistics for {compare_column} for nulls in {null_column}:")
                print(null_stats[compare_column])
                print(f"\nStatistics for {compare_column} for non-nulls in {null_column}:")
                print(non_null_stats[compare_column])
                print("\n")
            null_compare_dict = {}
            non_null_compare_dict = {}
            diff_compare_dict = {}
            for idx, stat_term in enumerate(stat_columns):
                current_thresh = thresh_values[idx]
                null_value = null_stats[compare_column][stat_term]
                non_null_value = non_null_stats[compare_column][stat_term]
                diff_value = 1 - (1.0 * null_value / non_null_value)
                null_str = f"{compare_column} {stat_term} for nulls in {null_column}"
                non_null_str = f"{compare_column} {stat_term} for non-nulls in {null_column}"
                diff_str = f"{compare_column} {stat_term} relative difference for nulls in {null_column}"
                null_compare_dict[null_str] = round(null_value, 3)
                non_null_compare_dict[non_null_str] = round(non_null_value, 3)
                diff_compare_dict[diff_str] = round(diff_value, 3)
                df_stats.loc[(compare_column, stat_term), null_column] = round(diff_value, 3)
                if abs(diff_value) >= current_thresh:
                    potential_str = (f"Relative difference above threshold {current_thresh} among nulls in"
                                     f" {null_column} for {compare_column} {stat_term}")
                    potential_correlations[potential_str] = diff_compare_dict[diff_str]
            if verbose > 0:
                print(null_compare_dict)
                print(non_null_compare_dict)
            print(diff_compare_dict)
        potential_correlations_master[null_column] = potential_correlations
        print(f"\nSummary of potential correlations in '{null_column}':\n{potential_correlations}")
        print("\n-----------------------------\n")
    # print(f"Complete summary of potential correlations in all {null_columns}:\n{potential_correlations_master}\n")
    print(f"Complete summary of potential correlations in all {null_columns}:\n")
    for k, v in potential_correlations_master.items():
        print(k, ':', v)
    print("\n")
    print(df_stats)
    return potential_correlations_master, df_stats


pdb.set_trace()
########################################################################################################################


if (show_all_output == True) or (show_all_output == 4):
    null_inspector(df_churn, contain_nulls_columns, continuous_columns, [0.04, 0.04], ['mean', 'std'], 1)
    null_inspector(df_churn, contain_nulls_columns, small_int_columns, [0.04, 0.04], ['mean', 'std'], 1)
    null_inspector(df_churn, contain_nulls_columns, item1_to_8_columns, [0.04, 0.04], ['mean', 'std'], 2)
else:
    null_inspector(df_churn, contain_nulls_columns, continuous_columns, [0.04, 0.04], ['mean', 'std'], 0)
    null_inspector(df_churn, contain_nulls_columns, small_int_columns, [0.04, 0.04], ['mean', 'std'], 0)
    null_inspector(df_churn, contain_nulls_columns, item1_to_8_columns, [0.04, 0.04], ['mean', 'std'], 0)

pdb.set_trace()


########################################################################################################################


def msno_plots(columns, df=df_churn, df_numerical=df_churn_numerical):
    msno.matrix(df[columns], labels=True)
    plt.show()
    msno.dendrogram(df[columns])
    plt.show()
    msno.heatmap(df_numerical, labels=True, cmap='RdBu', fontsize=12, vmin=-0.1, vmax=0.1)
    plt.show()


def quick_corr_search():
    print(f"Correlation matrix on numerical columns:\n{df_churn_numerical.corr()}\n")
    plots_by_column(continuous_columns, df_churn, 'pairplot')
    plots_by_column(['Tenure'], df=df_churn, mode='scatter', y_scatter='Bandwidth_GB_Year')


def sample_plots():
    adjusted_null_scatter(df_churn_numerical, 0.075, 'Bandwidth_GB_Year', 'Age')
    adjusted_null_scatter(df_churn_numerical, 0.075, 'Income', 'Age')
    adjusted_null_scatter(df_churn_numerical, 0.075, 'Tenure', 'Bandwidth_GB_Year')


if (show_all_output == True) or (show_all_output == 5):
    msno_plots(continuous_columns)
    pdb.set_trace()
    msno_plots(all_columns)
    pdb.set_trace()
    quick_corr_search()
    sample_plots()

pdb.set_trace()


########################################################################################################################


def cat_yn_num_compare(yes_no_column, numer_column, df=df_churn):
    print(df.loc[df_churn[yes_no_column] == 'Yes', numer_column].describe())
    print(df.loc[df_churn[yes_no_column] == 'No', numer_column].describe())
    print(df.loc[df_churn[yes_no_column].isna(), numer_column].describe())


def kde_plotter(column, thresh, df=df_churn):
    sns.kdeplot(df.loc[df[column] < thresh, column], color='blue', label='default', linestyle='--')
    plt.show()


def area_inspector():
    df_urban = df_churn.loc[df_churn['Area'] == 'Urban']
    df_suburban = df_churn.loc[df_churn['Area'] == 'Suburban']
    df_rural = df_churn.loc[df_churn['Area'] == 'Rural']

    print(f"'Population' values when 'Area' = 'Urban':\n{df_urban['Population'].describe()}\n")
    print(f"'Population' values when 'Area' = 'Suburban':\n{df_suburban['Population'].describe()}\n")
    print(f"'Population' values when 'Area' = 'Rural':\n{df_rural['Population'].describe()}\n")

    null_inspector(df_rural, contain_nulls_columns, continuous_columns, [0.04, 0.04], ['mean', 'std'], 0)
    sns.scatterplot(df_churn, x='Bandwidth_GB_Year', y='Outage_sec_perweek', hue='Area')
    plt.show()

    plt.hist(df_churn['Population'])
    plt.xlabel('Population')
    plt.show()

    df_low_pop = df_churn[df_churn['Population'] < 20]
    print(f"Low population dataframe:\n{df_low_pop}\n")

    plt.hist(df_low_pop['Population'])
    plt.xlabel('Low population')
    plt.show()

    plt.hist(df_low_pop['Area'])
    plt.xlabel('Low population by Area type')
    plt.show()


def pop_income_charge_oddities(pop_thresh=20, income_thresh=1000, charge_thresh=0.5):
    print("\nPopulation values under 20:")
    print(df_churn.loc[df_churn['Population'] < pop_thresh, ['Population', 'City', 'Area']].sort_values(by='Population'))
    print("\nIncome values under 1000:")
    print(df_churn.loc[df_churn['Income'] < income_thresh, ['Job', 'Income', 'Employment', 'City']].sort_values(by='Income'))
    print("\nInstances where the annual charge is more than half of the customer's income:")
    print(df_churn.loc[charge_thresh * df_churn['Income'] < 12 * df_churn['MonthlyCharge'], ['Income', 'MonthlyCharge']].sort_values(by='Income'))


def pop_variance_agg(thresh=0.9):
    agg_df = df_churn[(['Population', 'City', 'State'])].groupby(by=['City', 'State']).agg(
        ['mean', 'std', 'max', 'min', 'count'])
    filter_condition = (1 - agg_df[('Population', 'min')] / agg_df[('Population', 'max')]) > thresh
    filt_agg_df_inspect = agg_df[filter_condition]
    print("\nAggregated dataframe:")
    print(filt_agg_df_inspect)


def band_w_no_internet():
    df_nullinternet = df_churn.loc[df_churn['InternetService'].isna()]
    df_nonullinternet = df_churn.loc[df_churn['InternetService'].notna()]
    print(f"Number of 'None' values for 'InternetService': {len(df_nullinternet)}\n")
    print("'Bandwidth_GB_Year' values for 'InternetService' == 'None':\n")
    print(df_nullinternet['Bandwidth_GB_Year'].describe())
    print("'Bandwidth_GB_Year' values for 'InternetService' != 'None':\n")
    print(df_nonullinternet['Bandwidth_GB_Year'].describe())


if (show_all_output == True) or (show_all_output == 6):
    # Higher 'MonthlyCharge' on 'yes' for 'TechSupport'
    cat_yn_num_compare('TechSupport', 'MonthlyCharge')
    area_inspector()
    pdb.set_trace()
    print(df_churn[['Income', 'Job', 'Employment']][(df_churn['Income'] < 30000) & (df_churn['Job'] == 'Psychiatrist')])
    # 1. Low population values are likely in error and will be modified in the following section that alters the data.
    # 2. The very low income values and seemingly inconsistent relationships to Employment status and Job title
    # requires further followup with customers. Presumably those listed as 'Retired' retain their previous career
    # under 'Job'. Widely varying and unexpected incomes, such as full time doctors and engineers with incomes under
    # 10000, and retired customers having high incomes, could potentially be explained by a customer reporting
    # individual vs household income, failing to update, under- or over-reporting, error in data entry, geographic
    # location, pensions, and other reasons. The overall distribution of incomes skews low, but doesn't show signs of
    # widespread systematic error. Investigating its accuracy would require external datasets or additional contact
    # with customers.
    # 3. When the annual charge is 50% or more of the income, it's possible the income was recorded incorrectly.
    # However, spouses and family members could be paying the bills. Additionally, the incomes generally skew low,
    # and further investigation with customers would be needed. Modifying incomes at this point would be reckless.
    pop_income_charge_oddities(20, 1000, 0.5)
    kde_plotter('Income', 2000)
    pop_variance_agg()
    # Customers with 'None' for 'InternetService' have 'Bandwidth_GB_Year' values very similar to customers with
    # internet. Presumably this isn't in error and the bandwidth utilized comes from a different internet service
    # provider.
    band_w_no_internet()

pdb.set_trace()


########################################################################################################################


# From inspection of linear fits on 'Tenure' and 'Bandwidth_GB_Year', the fit is noisier and less accurate below values
# of 1000 for 'Bandwidth_GB_Year'.
cutoff_bandwidth = 1000


# Linear regression fit on 'Bandwidth_GB_Year' vs 'Tenure' subject to a cutoff value of cutoff_bandwidth. Default
# value of gt_condition=True applies the fit to bandwidth values above the cutoff. Produces summary MSE and r^2 for
# the fit as well as its coefficient and intercept parameters. Also displays a scatter plot with the fitted line.
# Returns the model's coefficient and intercept for later imputation.
def tenure_band_lin_fit(cutoff_bandwidth, gt_condition=True, verbose=False):
    if gt_condition:
        df_linear = df_churn.loc[
            df_churn['Bandwidth_GB_Year'] >= cutoff_bandwidth, ['Tenure', 'Bandwidth_GB_Year']].copy(deep=True)
    else:
        df_linear = df_churn.loc[
            df_churn['Bandwidth_GB_Year'] < cutoff_bandwidth, ['Tenure', 'Bandwidth_GB_Year']].copy(deep=True)
    df_linear = df_linear.dropna()
    X_linear = df_linear[['Tenure']]
    y_linear = df_linear[['Bandwidth_GB_Year']]
    print(f"Number of points being fit: {len(X_linear)}")
    X_train, X_test, y_train, y_test = train_test_split(X_linear, y_linear, test_size=0.3)
    lm = linear_model.LinearRegression()
    lm.fit(X_train, y_train)
    lm_fit_coef = lm.coef_[0][0]
    lm_fit_intercept = lm.intercept_[0]
    y_pred = lm.predict(X_test)
    if verbose:
        print(f"Fitting {'above' if gt_condition else 'below'} 'Bandwidth_GB_Year' values of {cutoff_bandwidth}")
        print(f"fit coefficient: {lm_fit_coef}, fit intercept: {lm_fit_intercept}")
        print(f"MSE: {mean_squared_error(y_test, y_pred)}")
        print(f"r^2: {r2_score(y_test, y_pred)}")
        plt.scatter(X_test, y_test, color='black')
        plt.plot(X_test, y_pred, color='blue', linewidth=3)
        plt.show()
    return lm_fit_coef, lm_fit_intercept


if (show_all_output == True) or (show_all_output == 7):
    lm_coef, lm_intercept = tenure_band_lin_fit(cutoff_bandwidth, True, True)
    pdb.set_trace()
    # Trying linear fit below cutoff_bandwidth = 1000
    lm_coef_lt_cutoff, lm_intercept_lt_cutoff = tenure_band_lin_fit(cutoff_bandwidth, False, True)
else:
    lm_coef, lm_intercept = tenure_band_lin_fit(cutoff_bandwidth, True, False)
    pdb.set_trace()
    # Trying linear fit below cutoff_bandwidth = 1000
    lm_coef_lt_cutoff, lm_intercept_lt_cutoff = tenure_band_lin_fit(cutoff_bandwidth, False, False)

pdb.set_trace()


########################################################################################################################
'''
Data cleaning and modification starts here, see lines 410 to 558 for creation of alternate 'Job' columns.
'''


# 'InternetService' has a valid entry of 'None'. The number of null 'InternetService' values after importing the csv is
# precisely equal to the number of 'None' values in Excel for that column. Replacing nulls with the string 'None'.
df_churn['InternetService'] = df_churn['InternetService'].fillna("None")


# Creates a new column 'Zip_int64' to backup the old 'Zip' values while adjusting current 'Zip' values to strings with
# five digits.
def zip_to_str(df=df_churn, zip_col='Zip'):
    df['Zip_int64'] = df[zip_col]
    df[zip_col] = df[zip_col].astype('str')
    for i in range(5):
        df[zip_col].mask(df[zip_col].str.len() == i, '0' * (5 - i) + df[zip_col], inplace=True)
    print(
        f"Verifying number of entries in 'Zip' with number of digits other than 5: {len(df_churn.loc[df_churn['Zip'].str.len() != 5, 'Zip'])}")
    print(
        f"Verifying number of entries in 'Zip' with number of digits exactly 5: {len(df_churn.loc[df_churn['Zip'].str.len() == 5, 'Zip'])}")


# Converts negative outage values to their absolute value.
def abs_outage(df=df_churn, outage_col='Outage_sec_perweek', outage_thresh=0.01):
    df.loc[df[outage_col] < outage_thresh, outage_col] = df[df[outage_col] < outage_thresh][outage_col].apply(
        lambda x: abs(x))
    print(df[outage_col].describe())
    print(df[outage_col][df[outage_col] < outage_thresh])


zip_to_str()
abs_outage()

pdb.set_trace()


########################################################################################################################


# Uses the linear regression model coefficients to replace null values in 'Tenure' and 'Bandwidth_GB_Year' when possible
def band_tenure_linear_impute(band_cut=cutoff_bandwidth):
    # Condition for null 'Bandwidth_GB_Year' values and non-null 'Tenure' values
    nullband_notnulltenure = (df_churn['Bandwidth_GB_Year'].isna()) * (df_churn['Tenure'].notna())
    # Condition for null 'Tenure' values and non-null 'Bandwidth_GB_Year' values that are also >= cutoff_bandwidth from
    # the linear fit model
    nulltenure_bandgtcutoff = (df_churn['Tenure'].isna()) * (df_churn['Bandwidth_GB_Year'] >= band_cut)
    # Imputation using linear model
    df_churn.loc[nullband_notnulltenure, 'Bandwidth_GB_Year'] = df_churn[nullband_notnulltenure]['Tenure'].apply(
        lambda t: round(lm_coef * t + lm_intercept, 7))
    df_churn.loc[nulltenure_bandgtcutoff, 'Tenure'] = df_churn[nulltenure_bandgtcutoff]['Bandwidth_GB_Year'].apply(
        lambda b: round((b - lm_intercept) / lm_coef, 7))


# Verifies and prints the number of remaining null values in 'Tenure' and 'Bandwidth_GB_Year'
def band_tenure_post_impute_verify():
    print(
        f"Number of remaining rows with non-null tenure and null bandwidth: {len(df_churn[(df_churn['Bandwidth_GB_Year'].isna()) * (df_churn['Tenure'].notna())])}\n")
    print(
        f"Number of remaining rows with null tenure and bandwidth >= {cutoff_bandwidth}: {len(df_churn[(df_churn['Tenure'].isna()) * (df_churn['Bandwidth_GB_Year'] >= cutoff_bandwidth)])}\n")
    print(
        f"Number of remaining rows with null tenure and non-null bandwidth: {len(df_churn[(df_churn['Tenure'].isna()) * (df_churn['Bandwidth_GB_Year'].notna())])}\n")
    print(
        f"Number of remaining rows with null tenure and null bandwidth: {len(df_churn[(df_churn['Tenure'].isna()) * (df_churn['Bandwidth_GB_Year'].isna())])}\n")
    print(
        f"Number of remaining rows with non-null tenure and non-null bandwidth: {len(df_churn[(df_churn['Tenure'].notna()) * (df_churn['Bandwidth_GB_Year'].notna())])}\n")
    print(df_churn['Bandwidth_GB_Year'].describe())
    print(df_churn['Tenure'].describe())


band_tenure_linear_impute(cutoff_bandwidth)
band_tenure_post_impute_verify()

pdb.set_trace()


########################################################################################################################


# MICE imputation
def mice_impute(impute_columns, min_vals, df=df_churn):
    df_mice = df[impute_columns].copy(deep=True)
    mice_imputer = IterativeImputer(imputation_order='roman', min_value=min_vals)
    df_mice.iloc[:, :] = mice_imputer.fit_transform(df_mice)
    print("MICE imputation results:\n")
    print(df_mice.describe())
    print(df[impute_columns].describe())
    return df_mice


# KNN imputation
def knn_impute(impute_columns, df=df_churn):
    df_knn = df[impute_columns].copy(deep=True)
    knn_imputer = KNNImputer()
    df_knn.iloc[:, :] = knn_imputer.fit_transform(df_knn)
    print("KNN imputation results:\n")
    print(df_knn.describe())
    print(df[impute_columns].describe())
    return df_knn


# Overwrites the original dataframe df with the calculated imputed values
def impute_overwrite(impute_columns, df_impute, df=df_churn):
    for impute_coln in impute_columns:
        if impute_coln == 'Age' or impute_coln == 'Children':
            df.loc[df[impute_coln].isna(), impute_coln] = df_impute[impute_coln].astype(int)
        df.loc[df[impute_coln].isna(), impute_coln] = df_impute[impute_coln]


# Overlays density plots of impute_column between the original dataframe df and two imputations A and B
def impute_density_plot_compare(impute_column, df, df_imp_A, df_imp_B):
    sns.kdeplot(df[impute_column], color='blue', label='default', linestyle='--')
    sns.kdeplot(df_imp_A[impute_column], color='red', label='imputed_A', linestyle='-')
    sns.kdeplot(df_imp_B[impute_column], color='green', label='imputed_B', linestyle=':')
    plt.xlabel(impute_column)
    plt.ylabel('Density')
    plt.legend()
    plt.show()


# Shows three scatter plots of y vs x for the original dataframe df, imputation A, then imputation B
def impute_scatter_plot_compare(x_col, y_col, df, df_imp_A, df_imp_B):
    plt.xlabel(x_col)
    plt.scatter(x=df[x_col], y=df[y_col])
    plt.ylabel((y_col + " original"))
    plt.show()
    plt.scatter(x=df_imp_A[x_col], y=df_imp_A[y_col])
    plt.ylabel((y_col + " imputed_A"))
    plt.show()
    plt.scatter(x=df_imp_B[x_col], y=df_imp_B[y_col])
    plt.ylabel((y_col + " imputed_B"))
    plt.show()


pdb.set_trace()
########################################################################################################################


# 'MonthlyCharge' and 'Outage_sec_perweek' have no null values but are included to provide additional degrees for
# MICE and KNN to calculate more accurate imputed values.
numerical_impute_columns = ['Children', 'Age', 'Income', 'Tenure', 'Bandwidth_GB_Year', 'MonthlyCharge',
                            'Outage_sec_perweek']
floor_values = [0, 18, 741.00, 1.00, 155.00, 77.50, 0.00]

df_churn_mice = mice_impute(numerical_impute_columns, floor_values)
df_churn_knn = knn_impute(numerical_impute_columns)

if (show_all_output == True) or (show_all_output == 8):
    impute_density_plot_compare('Children', df_churn, df_churn_mice, df_churn_knn)
    pdb.set_trace()
    impute_density_plot_compare('Age', df_churn, df_churn_mice, df_churn_knn)
    pdb.set_trace()
    impute_density_plot_compare('Income', df_churn, df_churn_mice, df_churn_knn)
    pdb.set_trace()
    impute_scatter_plot_compare('Tenure', 'Bandwidth_GB_Year', df_churn, df_churn_mice, df_churn_knn)
    pdb.set_trace()

# KNN is preferred for better mimicking the original distributions. MICE has a strong preference for the mean and
# causes distortions. Linear imputation must be performed first on 'Tenure' and 'Bandwidth' for best results.
impute_overwrite(numerical_impute_columns, df_churn_knn)
print(df_churn[numerical_impute_columns].describe())

# Verifying MICE imputation overwrite non-integer median didn't involve non-integer imputations
print(df_churn.loc[(df_churn['Age'] > 53.0) * (df_churn['Age'] < 54.0), 'Age'])

# Performing linear fit on bandwidth and tenure again after imputation (still above cutoff_bandwidth)
if (show_all_output == True) or (show_all_output == 7):
    lm_coef_new, lm_intercept_new = tenure_band_lin_fit(cutoff_bandwidth, True, True)
    # tenure_band_lin_fit(0, True, True)
    # plt.scatter(x=df_churn['Tenure'], y=df_churn['Bandwidth_GB_Year'])
    # plt.show()
else:
    lm_coef_new, lm_intercept_new = tenure_band_lin_fit(cutoff_bandwidth, True, False)

pdb.set_trace()


########################################################################################################################


# Helper function for population_adjuster below. When an element (population) is below a cutoff value elem_cutoff and
# is also below a certain threshold percentage of the maximum 'Population' value recorded for other instances of the
# same (city, state), it's considered an incorrect input and replaced with the mean amongst entries with that (city,
# state). If it's a uniquely occurring value it's left alone as the customer might live in a very isolated area.
def low_min_convert_to_mean_adjuster(elem, city, state, elem_cutoff, df_agg, thresh, col_adj='Population'):
    if (city, state) in df_agg.index:
        working_df = df_agg.loc[(city, state)]
        if elem < thresh * working_df[(col_adj, 'max')] and elem < elem_cutoff:
            elem = working_df[(col_adj, 'mean')]
    return elem


# Adjusts 'Population' values below a cutoff (pop_cutoff) when the maximum 'Population' among entries with the same
# (city, state) is significantly higher than the minimum. Values below pop_cutoff (defaults to 20) are considered
# entered in error. An aggregated dataframe on (city, state) is created under the condition that the minimum
# 'Population' entry for that (city, state) is less than (agg_thresh * (max 'Population' in that city and state)).
# The old 'Population' column is saved in a new column 'Population_raw'. Then the 'Population' column has values
# below the cutoff updated to the aggregated mean of other entries with the corresponding (city, state). Instances where
# there is no neighbor to compare with are unaffected as the 'min' and 'max' values are equivalent, so filter_condition
# fails and that entry is passed over. For a 'Population' value to be altered, it must be below pop_threshold,
# less than (pop_spread_thresh * (max population in that city)), and the city's aggregated minimum must be
# less than (agg_thresh * (max population in that city)). To accommodate 'Population' values above the minimum
# (frequently 0 in the aggregated dataframe) but still under pop_cutoff, it's necessary that pop_spread_thresh be
# slightly higher than agg_thresh. Returns the aggregated dataframe satisfying the condition filter_condition.
def population_adjuster(agg_thresh=0.1, pop_spread_thresh=0.15, pop_cutoff=20, pop_col='Population', city_col='City',
                        state_col='State', df=df_churn):
    location_columns = [city_col, state_col]
    agg_columns = [pop_col]
    agg_df = df[(agg_columns + location_columns)].groupby(by=location_columns).agg(
        ['mean', 'std', 'max', 'min', 'count'])
    filter_condition = (1 - agg_df[(pop_col, 'min')] / agg_df[(pop_col, 'max')]) > (1 - agg_thresh)
    filt_agg_df = agg_df[filter_condition]
    print("\nAggregated dataframe:")
    print(filt_agg_df)

    num_low_pop_vals = len(df[df[pop_col] < pop_cutoff])
    old_pop_col = pop_col + '_raw'
    df[old_pop_col] = df[pop_col]
    df[pop_col] = df.apply(lambda row: low_min_convert_to_mean_adjuster(row[pop_col], row[location_columns[0]],
                                                                        row[location_columns[1]], pop_cutoff,
                                                                        filt_agg_df, pop_spread_thresh), axis=1)
    df[pop_col] = df[pop_col].astype('int64')
    updated_num_low_pop_vals = len(df[df[pop_col] < pop_cutoff])
    print(f"\nInitially there were {num_low_pop_vals} outlying population values below {pop_cutoff}.\n"
          f"{num_low_pop_vals - updated_num_low_pop_vals} were adjusted to the means of customers living in the same "
          f"city.\nThe remaining {updated_num_low_pop_vals} had no nearby neighbors to compare with.\n")
    return filt_agg_df


print("\nExample for Albany, NY:")
print(df_churn[(df_churn['City'] == 'Albany') * (df_churn['State'] == 'NY')][['Population', 'Zip', 'Lat', 'Lng']])
filt_agg_df = population_adjuster()

pdb.set_trace()
########################################################################################################################


# The columns in X_cols below had sufficiently high feature importance scores in the balanced random forests for each
# of the three categorical columns with null values ('Phone', 'Techie', 'TechSupport').
# Model parameters in cat_params (n_estimators, max_depth, sampling_strategy) were selected on the basis of providing
# a similar proportion of 'Yes'/'No' to the non-null values in that column and having as high an accuracy score as
# possible on the test split data.

X_cols = ['Lat', 'Lng', 'Population', 'Outage_sec_perweek', 'MonthlyCharge', 'Age', 'Email', 'Income']
cat_params = {'Phone': [400, 6, 0.84], 'Techie': [600, 10, 0.85], 'TechSupport': [600, 10, 0.87]}


# Uses a balanced random forest to predict categorical variables in the keys of the cat_params dictionary. Uses X_cols
# as the predictor variables for each cat_params key. Prints model feature importances, accuracy scores, and compares
# its predictions to the proportion of 'Yes'/'No' in the non-null data. Then imputes the predictions into the original
# dataframe's respective column.

def cat_null_model(cat_params, X_cols, df=df_churn, verbose=True):
    for class_col, class_params in cat_params.items():
        df_null_class_col = df.loc[df[class_col].isna(), X_cols].dropna()
        df_class = df.copy(deep=True)[X_cols + [class_col]].dropna()
        y_class = df_class[class_col]
        X_class = df_class[X_cols]
        X_train, X_test, y_train, y_test = train_test_split(X_class, y_class, test_size=0.3)

        class_model = BalancedRandomForestClassifier(n_estimators=class_params[0], max_depth=class_params[1],
                                                     replacement=False, sampling_strategy=class_params[2],
                                                     bootstrap=True)
        class_model.fit(X_train, y_train)
        if verbose:
            print(f"Model feature importances for {class_col}:\n{class_model.feature_importances_}\n")
            print(f"Model accuracy on test data:\n{accuracy_score(y_test, class_model.predict(X_test))}\n")
            print(f"Model accuracy on train data:\n{accuracy_score(y_train, class_model.predict(X_train))}\n")

        predictions = class_model.predict(df_null_class_col)
        prediction_stats = sum(1 for t in predictions if t == 'Yes')
        if verbose:
            print(f"{prediction_stats} 'Yes' and {len(predictions) - prediction_stats} 'No', compare to "
                  f"{df[class_col].value_counts()['Yes']} 'Yes' and {df[class_col].value_counts()['No']} 'No'\n")

        # Replacing null values in class_col with the values predicted by its model
        df.loc[df[class_col].isna(), class_col] = predictions


if (show_all_output == True) or (show_all_output == 9):
    cat_null_model(cat_params, X_cols, df_churn, True)
else:
    cat_null_model(cat_params, X_cols, df_churn, False)

pdb.set_trace()
print(df_churn.describe())
print(df_churn.select_dtypes(['object']).describe())
pd.set_option('display.max_rows', 60)
print(f"\nInspecting dataframe for remaining nulls:\n{df_churn.isnull().any()}\n")
pd.set_option('display.max_rows', 16)

if (show_all_output == True) or (show_all_output == 9):
    print("\nReviewing display counts from original dataframe:\n")
    display_all_values(mode=True, df=df_backup)
    zscore_hists(df_churn, continuous_columns)
    zscore_hists(df_churn, small_int_columns)
    print("\nSearching for any lingering outliers or anomalies:\n")
    cleaned_item1_to_8_outliers, cleaned_small_int_outliers, cleaned_continuous_outliers = display_all_values(mode=False)

clean_data_name = 'cleaned_churn_raw_data.csv'
print(f"Saving cleaned data file to '{clean_data_name}'")
df_churn.to_csv(r'D206-churn-files\cleaned_churn_raw_data.csv')

pdb.set_trace()
########################################################################################################################


# Excluding ['Children', 'Email', 'Contacts', 'Yearly_equip_failure'] because they're integers with small ranges.
# Keeping 'Age' and 'Population' since age is an integer for convenience and population ranges from 0 to ~10^5.
pca_columns_churn = ['Age', 'Population', 'Lat', 'Lng', 'Income', 'Outage_sec_perweek', 'Tenure', 'MonthlyCharge',
                     'Bandwidth_GB_Year']


# Adapted from WGU D206 Data Cleaning "Welcome to Getting Started With Principal Component Analysis". Accessed 2024.
# Performs PCA and produces scree plot to determine optimal number of features
def pca_procedure(pca_columns, df=df_churn, verbose=False):
    pca_scaler = StandardScaler()
    df_normalized = pca_scaler.fit_transform(df[pca_columns])
    pca_col_count = df_normalized.shape[1]
    pca = PCA(n_components=pca_col_count)
    pca.fit(df_normalized)
    pca_col_names = ['PC' + f'{i + 1}' for i in range(pca_col_count)]
    df_pca = pd.DataFrame(pca.transform(df_normalized), columns=pca_col_names)
    if verbose:
        pd.set_option('display.max_columns', 5)
        print("\nPCA dataframe:\n")
        print(df_pca)
        pd.set_option('display.max_columns', None)
    pca_loadings = pd.DataFrame(pca.components_.T, columns=pca_col_names, index=pca_columns)
    print("\nPCA loadings:\n")
    print(pca_loadings)
    print("\nPCA explained variance ratio:\n")
    print(pca.explained_variance_ratio_)
    if verbose:
        plt.plot(pca.explained_variance_ratio_)
        plt.xlabel('PCA component index')
        plt.ylabel('Explained variance ratio')
        plt.show()

    pca_cov_matrix = np.dot(df_normalized.T, df_normalized) / df_normalized.shape[0]
    pca_eigenvalues = [np.dot(eigenvector.T, np.dot(pca_cov_matrix, eigenvector)) for eigenvector in pca.components_]
    if verbose:
        plt.plot(pca_eigenvalues)
        plt.xlabel('Number of components')
        plt.ylabel('PCA eigenvalue')
        plt.axhline(y=1, color='red')
        plt.show()


pdb.set_trace()
########################################################################################################################


if (show_all_output == True) or (show_all_output == 9):
    pca_procedure(pca_columns_churn, df_churn, True)
else:
    pca_procedure(pca_columns_churn, df_churn, False)

########################################################################################################################

