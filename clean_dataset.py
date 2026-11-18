import pandas as pd

# Load dataset
df = pd.read_csv(
    "Dataset/dataset project.csv",
    encoding="latin1"
)

print("Original Shape:", df.shape)

# 1. Remove completely empty columns
df.dropna(axis=1, how="all", inplace=True)

# 2. Remove unnecessary unnamed columns
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

# 3. Remove duplicate rows
df.drop_duplicates(inplace=True)

# 4. Fill missing subject values
if "subject" in df.columns:
    df["subject"] = df["subject"].fillna("")

# 5. Fill missing body values
if "body" in df.columns:
    df["body"] = df["body"].fillna("")

# 6. Remove rows where both subject and body are empty
if "subject" in df.columns and "body" in df.columns:
    df = df[
        (df["subject"].str.strip() != "") |
        (df["body"].str.strip() != "")
    ]

# Save cleaned dataset
df.to_csv("Dataset/cleaned_dataset.csv", index=False)

print("\nCleaned Shape:", df.shape)
print("\nRemaining Missing Values:")
print(df.isnull().sum().sort_values(ascending=False).head(15))

print("\nCleaned dataset saved successfully!")