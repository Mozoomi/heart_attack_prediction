import pandas as pd

df = pd.read_csv('data/data.csv')

print(df.isna())
df = df.dropna() #Drop rows with null values
df = df.drop_duplicates() #drop duplicate rows

df.to_csv('data/clean_data.csv', index=False) #export

print("Data cleaned")