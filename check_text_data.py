import pandas as pd

df = pd.read_csv(
    "Dataset/dataset project.csv",
    encoding="latin1",
    low_memory=False
)

print("TOTAL ROWS:", len(df))

print("\nSUBJECT:")
print("Non-empty:", df["subject"].notna().sum())
print("Empty:", df["subject"].isna().sum())

print("\nBODY:")
print("Non-empty:", df["body"].notna().sum())
print("Empty:", df["body"].isna().sum())

print("\nBOTH SUBJECT AND BODY EMPTY:")
print(
    (
        df["subject"].isna() & df["body"].isna()
    ).sum()
)

print("\nDUPLICATE ROWS:", df.duplicated().sum())

print("\nDUPLICATE EMAIL BODIES:", df["body"].duplicated().sum())