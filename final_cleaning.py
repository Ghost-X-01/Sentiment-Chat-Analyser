import pandas as pd

# Load original dataset
df = pd.read_csv(
    "Dataset/dataset project.csv",
    encoding="latin1",
    low_memory=False
)

print("Original shape:", df.shape)

# Keep only useful columns
df = df[
    ["sender_name", "sender_email", "receiver_name",
     "receiver_email", "subject", "body", "date"]
].copy()

# Remove rows where both subject and body are missing
df = df[
    df["subject"].notna() | df["body"].notna()
].copy()

# Fill missing text values
df["subject"] = df["subject"].fillna("").astype(str).str.strip()
df["body"] = df["body"].fillna("").astype(str).str.strip()

# Remove rows where both fields are empty after cleaning
df = df[
    (df["subject"] != "") | (df["body"] != "")
].copy()

# Create combined text
df["combined_text"] = (
    df["subject"] + " " + df["body"]
).str.strip()

# Remove duplicate combined text
df = df.drop_duplicates(
    subset=["combined_text"]
).copy()

# Save cleaned dataset
df.to_csv(
    "Dataset/final_cleaned_dataset.csv",
    index=False,
    encoding="utf-8"
)

print("\nFinal dataset shape:", df.shape)
print("Missing subject:", df["subject"].isna().sum())
print("Missing body:", df["body"].isna().sum())
print("Duplicate text:", df["combined_text"].duplicated().sum())

print("\nFinal cleaned dataset saved successfully!")