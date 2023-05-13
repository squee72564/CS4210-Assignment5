#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #5
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

# Use the command: "pip install mlxtend" on your terminal to install the mlxtend library

# Read the dataset using pandas
df = pd.read_csv('retail_dataset.csv', sep=',')

# Find the unique items all over the data and store them in the set below
itemset = set()
for i in range(0, len(df.columns)):
    items = (df[str(i)].unique())
    itemset = itemset.union(set(items))

# Remove nan (empty) values
itemset.remove(np.nan)

# Convert the dataset for Apriori module
encoded_vals = []
for index, row in df.iterrows():
    labels = {}
    for item in itemset:
        if item in row.values:
            labels[item] = 1
        else:
            labels[item] = 0
    encoded_vals.append(labels)

# Create a dataframe from the populated list with multiple dictionaries
ohe_df = pd.DataFrame(encoded_vals)

# Call the apriori algorithm with specified parameters
freq_items = apriori(ohe_df, min_support=0.2, use_colnames=True, verbose=1)
rules = association_rules(freq_items, metric="confidence", min_threshold=0.6)

# Iterate over the rules dataframe and print the results
for index, rule in rules.iterrows():
    antecedents = ', '.join(list(rule['antecedents']))
    consequents = ', '.join(list(rule['consequents']))
    support = rule['support']
    confidence = rule['confidence']

    supportCount = sum(ohe_df[consequents] == 1)
    prior = supportCount / len(encoded_vals)
    gain = 100 * (confidence - prior) / prior

    print(f"{antecedents} -> {consequents}")
    print(f"Support: {support}")
    print(f"Confidence: {confidence}")
    print(f"Prior: {prior}")
    print(f"Gain in Confidence: {gain}")
    print()

# Plot support x confidence
plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()
