import pandas as pd

df = pd.read_csv(
    "Dataset/dataset project.csv",
    encoding="latin1",
    low_memory=False
)

print("Dataset Shape:", df.shape)

print("\nAll Column Names:")
print(df.columns.tolist())

print("\nFirst 5 Rows (First 20 Columns):")
print(df.iloc[:5, :20].to_string())

print("\nDuplicate Rows:", df.duplicated().sum())

print("\nNon-empty values in Unnamed columns:")
unnamed = df.loc[:, df.columns.str.contains("^Unnamed")]
print(unnamed.notna().sum().head(20))