import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency


df_med = pd.read_csv('D207_medical_data\medical_clean.csv')

cont_table = pd.crosstab(df_med['HighBlood'], df_med['Complication_risk'])
print(cont_table)
chi2, p, dof, expected = chi2_contingency(cont_table)
print(f"\nchi2 is {chi2} with {dof} degrees of freedom, giving p-value: {p}\n")
df_expect = pd.DataFrame(data=expected, index=['No', 'Yes'], columns=['High', 'Low', 'Medium'])
print(f"Expected values were:\n{df_expect}\n")


print(df_med['Services'].value_counts())
print("\n")
plt.title('Distribution of primary service sought by patients')
plt.xlabel('Type of service')
plt.ylabel('Frequency')
plt.hist(df_med['Services'])
plt.show()

print(df_med['Initial_admin'].value_counts())
print("\n")
plt.title('Distribution of patient admission types')
plt.xlabel('Initial admission type')
plt.ylabel('Frequency')
plt.hist(df_med['Initial_admin'])
plt.show()

print(df_med['Initial_days'].describe())
print("\n")
plt.title('Duration of initial hospital stay distribution')
plt.xlabel('Number of days during initial visit')
plt.ylabel('Density')
sns.kdeplot(x=df_med['Initial_days'], bw_adjust=0.3)
plt.show()

print(df_med['Additional_charges'].describe())
print("\n")
plt.title('Additional charges to patients distribution')
plt.xlabel('Additional charges')
plt.ylabel('Density')
sns.kdeplot(x=df_med['Additional_charges'], bw_adjust=0.3)
plt.show()


print(df_med['Overweight'].value_counts())
print("\n")
print(df_med['ReAdmis'].value_counts())
print("\n")
pivot_x = 'Overweight'
pivot_y = 'ReAdmis'
pivot_df = df_med.pivot_table(index=pivot_x, columns=pivot_y, aggfunc='size')
pivot_df.plot(kind='bar', stacked=True, color=['green', 'orange'])
plt.title('Stacked bar chart for patient obesity and readmission')
plt.xlabel('Overweight')
plt.ylabel('Frequency')
plt.legend(title='Readmission')
plt.show()

print(df_med['Income'].describe())
print("\n")
print(df_med['TotalCharge'].describe())
print("\n")
plt.title('Daily charge to patient ("TotalCharge") vs. income scatter plot')
plt.xlabel('Income')
plt.ylabel('Daily charge (averaged)')
plt.scatter(x=df_med['Income'], y=df_med['TotalCharge'])
plt.show()

df_med.plot.hexbin(x='Income', y='TotalCharge', gridsize=25)
plt.title('Hexbin heatmap of daily charge to patient ("TotalCharge") vs. income')
plt.xlabel('Income')
plt.ylabel('Daily charge (averaged)')
plt.show()

plt.title('Daily charge to patient ("TotalCharge") vs. income scatter plot colored by admission type')
plt.xlabel('Income')
plt.ylabel('Daily charge (averaged)')
sns.scatterplot(x=df_med['Income'], y=df_med['TotalCharge'], hue=df_med['Initial_admin'])
plt.show()

plt.title('Daily charge to patient ("TotalCharge") vs. income scatter plot colored by complication risk level')
plt.xlabel('Income')
plt.ylabel('Daily charge (averaged)')
sns.scatterplot(x=df_med['Income'], y=df_med['TotalCharge'], hue=df_med['Complication_risk'])
plt.show()
