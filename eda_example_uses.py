# some example usages for the user eda functions 

#import modules
import pandas as pd
import eda_functions as eda

# load pre cleaned data
df = pd.read_csv("data/cleaned_candid_data.csv")

# histogram of ages with boxplot
eda.visualise_numeric(df, "age", show_boxplot=True, bins = int(df["age"].max() - df["age"].min()), color = "red")

# barchart of candidate numbers per department (top 10) 
eda.visualise_categorical(df, "department_name", top_n = 10, horizontal = True, color = "red")

# plot of average age per department
eda.grouped_avgs(df, "age", "department_name", stat = "mean", color = "red")
