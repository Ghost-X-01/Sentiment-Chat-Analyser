import pandas as pd

df = pd.read_csv(
    "Dataset/dataset project.csv",
    encoding="latin1",
    low_memory=False
)

# Keep rows containing subject or body
df = df[
    df["subject"].notna() | df["body"].notna()
].copy()

# Convert missing values to empty strings
df["subject"] = df["subject"].fillna("").astype(str).str.strip()
df["body"] = df["body"].fillna("").astype(str).str.strip()

# Combine subject and body
df["combined_text"] = (
    df["subject"] + " " + df["body"]
).str.strip()

print("Rows containing text:", len(df))

print(
    "Rows with both subject and body:",
    ((df["subject"] != "") & (df["body"] != "")).sum()
)

print(
    "Rows with only subject:",
    ((df["subject"] != "") & (df["body"] == "")).sum()
)

print(
    "Rows with only body:",
    ((df["subject"] == "") & (df["body"] != "")).sum()
)

print(
    "Duplicate combined texts:",
    df["combined_text"].duplicated().sum()
)

print(
    "Unique combined texts:",
    df["combined_text"].nunique()
)