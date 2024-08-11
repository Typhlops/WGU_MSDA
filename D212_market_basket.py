import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


df_med = pd.read_csv('data/D212-medical-files/medical_market_basket.csv')
pd.set_option('display.max_columns', 12)

# Every other row is null, so those are removed here
df_med = df_med.dropna(axis=0, how='all')


all_columns = ['Presc01', 'Presc02', 'Presc03', 'Presc04', 'Presc05', 'Presc06', 'Presc07', 'Presc08', 'Presc09',
               'Presc10', 'Presc11', 'Presc12', 'Presc13', 'Presc14', 'Presc15', 'Presc16', 'Presc17', 'Presc18',
               'Presc19', 'Presc20']


# Verifying there are no nulls
def null_col_list(df=df_med):
    null_list = list(df.columns[df.isna().sum() > 0])
    print(f"Checking for columns with null values: {null_list}\n")
    return null_list


# Dataframe description and value counts
def inspect_data(columns, df=df_med):
    for col in columns:
        print(df[col].describe())
        print(df[col].value_counts())
        print("\n")


null_col_list()
inspect_data(df_med.columns)


# Creates a list of lists for market basket analysis
def nested_lists(df=df_med):
    nested = []
    for i in range(df.shape[0]):
        tmp_arr = []
        for j in range(df.shape[1]):
            elem = df.values[i, j]
            if not pd.isnull(elem):
                tmp_arr.append(str(elem))
        nested.append(tmp_arr)
    return nested


# Adapted from Dr. Kesselly Kamara's “Data Mining II – D212 Task 3” lecture video, accessed 2024.
# https://wgu.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=db85c4f1-0da5-4bde-a1a4-b07c0019d46d
# Creates a dataframe from using TransactionEncoder() to transform the 2D array in the parameter rows. Verifies no
# nulls present and saves it to a csv file. Then uses apriori() to create itemsets with a support above min_sup and
# uses it to produce a rule table according to metric 'metrc' above the threshold min_thresh.
def transactions_and_rules(rows, min_sup=0.02, metrc='lift', min_thresh=1.0):
    TE = TransactionEncoder()
    arr_TE = TE.fit_transform(rows)
    df_transactions = pd.DataFrame(arr_TE, columns=TE.columns_)
    print(f"Encoded transactions dataframe:\n{df_transactions}\n")
    print(f"Top 10 most frequent medications:\n{df_transactions.sum().sort_values(ascending=False).head(10)}\n")

    null_col_list(df_transactions)
    print("Saving dataframe of transactions to 'medical_cleaned_market_basket.csv'\n")
    df_transactions.to_csv('medical_cleaned_market_basket.csv')

    itemset_freq = apriori(df_transactions, min_support=min_sup, use_colnames=True)
    rule_table = association_rules(itemset_freq, metric=metrc, min_threshold=min_thresh)
    print(f"Itemsets with a support above {min_sup}:\n{itemset_freq}\n")
    print(f"Association rules with a {metrc} >= {min_thresh}:\n{rule_table}\n")
    return df_transactions, itemset_freq, rule_table


# Provides association rules for a selected medication 'drug' from rules_table. Creates a dataframe df_drug that
# searches for all occurrences of 'drug' in the 'antecedents' and 'consequents' columns of rules_table,
# sorted by sort_coln (defaults to 'lift'). When duplicates aren't of concern, a second dataframe df_drug_reduced is
# created that only retains unique sets (after performing a union on the antecedents and consequents) by selecting
# the one with the maximal value of agg_coln (defaults to 'confidence').
def rules_by_med(drug, rules_table, sort_coln='lift', agg_coln='confidence'):
    condition = ((rules_table['antecedents'].apply(lambda s: drug in s)) |
                 (rules_table['consequents'].apply(lambda s: drug in s)))
    df_drug = rules_table[condition].sort_values(by=sort_coln, ascending=False)
    df_drug['combined_set'] = df_drug.apply(lambda row: row['antecedents'].union(row['consequents']), axis=1)
    print(f"Association rules for all instances that {drug} appears in antecedents or consequents:\n{df_drug}\n")

    idx_conf = df_drug.groupby(by='combined_set')[agg_coln].idxmax()
    df_drug_reduced = df_drug.loc[idx_conf]
    print(f"Association rules for distinct joined sets of antecedents and consequents selected by maximal confidence:"
          f"\n{df_drug_reduced}\n")
    return df_drug_reduced, df_drug


nested_rows = nested_lists(df_med)
transactions, item_supp, tr_rules = transactions_and_rules(nested_rows, 0.015, 'lift', 1.0)


cond = (tr_rules['confidence'] >= 0.25) * (tr_rules['lift'] >= 1.8)
sort_col = 'lift'
print(f"Top rules by {sort_col}:\n{tr_rules[cond].sort_values(by=sort_col, ascending=False)}\n")


df_diaz_red, df_diaz = rules_by_med('diazepam', tr_rules, 'lift', 'confidence')



'''
df_arr = pd.DataFrame(nested_rows)
counts = list(df_arr.count())
new_counts = []
old_val = 0
for r in reversed(counts):
    new_counts.append(r-old_val)
    old_val = r
new_counts.reverse()
print(new_counts)
print(sum(new_counts))
'''


