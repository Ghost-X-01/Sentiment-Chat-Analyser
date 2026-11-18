import pandas as pd

df = pd.read_csv(
    "Dataset/dataset project.csv",
    encoding="latin1",
    low_memory=False
)

print("Dataset Shape:", df.shape)

print("\nMissing Values Per Column:")
print(df.isnull().sum())

print("\nMissing Value Percentage:")
print((df.isnull().sum() / len(df)) * 100)

print("\nTotal Missing Values:")
print(df.isnull().sum().sum())