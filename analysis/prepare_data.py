"""
Prepare a clean, analysis-ready Ghana subset from Afrobarometer Round 9 (2022).
Source: MafiAtUN/Afrobarometer-Data-Explorer (public GitHub mirror of Afrobarometer R9 merged data)
Focus: trust, perceived corruption, and perceived performance of traditional leaders
       vs. elected local government councils ("dual allegiance" in local governance).
"""
import pandas as pd
import numpy as np

SRC = "repo/data/processed/afrobarometer_processed_data_gzip.parquet"
OUT = "data/ghana_r9_clean.csv"

df = pd.read_parquet(SRC)
gh = df[df["Country_Name"] == "Ghana"].copy()

COLS = {
    "Q37D": "Q37d. Trust your elected local government council (Q37D)",
    "Q37J": "Q37j. Trust traditional leaders (Q37J)",
    "Q38D": "Q38d. Corruption: local government council lors (Q38D)",
    "Q38H": "Q38h. Corruption: traditional leaders (Q38H)",
    "Q47C": "Q47c. Performance: local government council lor (Q47C)",
    "Q11B": "Q11b. Contact local government council lor (Q11B)",
    "Q11E": "Q11e. Contact traditional leader (Q11E)",
    "URBRUR": "Urban or Rural Primary Sampling Unit (URBRUR)",
    "REGION": "Province or region (REGION)",
    "AGE": "Q1. Age (Q1)",
    "EDUC": "Q94. Education of respondent (Q94)",
    "GENDER": "Q100. Gender of respondent (Q100)",
    "POVERTY": "LivedPoverty",
}

sub = gh[list(COLS.values())].copy()
sub.columns = list(COLS.keys())

# --- Recode ordinal trust / corruption / performance scales to numeric 0-3 ---
trust_map = {
    "Not at all": 0, "Just a little": 1, "Somewhat": 2, "A lot": 3,
}
corr_map = {
    "None": 0, "Some of them": 1, "Most of them": 2, "All of them": 3,
}
perf_map = {
    "Strongly disapprove": 0, "Disapprove": 1, "Approve": 2, "Strongly approve": 3,
}
contact_map = {
    "Never": 0, "Only once": 1, "A few times": 2, "Often": 3,
}

def recode(series, mapping):
    return series.map(mapping)

sub["trust_local_gov"] = recode(sub["Q37D"], trust_map)
sub["trust_traditional"] = recode(sub["Q37J"], trust_map)
sub["corrupt_local_gov"] = recode(sub["Q38D"], corr_map)
sub["corrupt_traditional"] = recode(sub["Q38H"], corr_map)
sub["perf_local_gov"] = recode(sub["Q47C"], perf_map)

sub["contact_freq_local_gov"] = recode(sub["Q11B"], contact_map)
sub["contact_freq_traditional"] = recode(sub["Q11E"], contact_map)
sub["contacted_local_gov"] = (sub["contact_freq_local_gov"] > 0).astype("Int64")
sub["contacted_traditional"] = (sub["contact_freq_traditional"] > 0).astype("Int64")
sub.loc[sub["contact_freq_local_gov"].isna(), "contacted_local_gov"] = pd.NA
sub.loc[sub["contact_freq_traditional"].isna(), "contacted_traditional"] = pd.NA

# --- Demographics ---
sub["urban"] = (sub["URBRUR"] == "Urban").astype("Int64")
sub["region"] = sub["REGION"].astype(str).str.title()
sub["age"] = pd.to_numeric(sub["AGE"], errors="coerce")
sub["female"] = (sub["GENDER"] == "Woman").astype("Int64")
sub["lived_poverty"] = pd.to_numeric(sub["POVERTY"], errors="coerce")

edu_order = [
    "No formal schooling", "Informal schooling only (including Koranic schooling)",
    "Some primary schooling", "Primary school completed",
    "Intermediate school or some secondary school/high school",
    "Secondary school/high school completed",
    "Post -secondary qu alifications other than university",
    "University completed", "Post - graduate",
]
edu_map = {v: i for i, v in enumerate(edu_order)}
sub["education_level"] = sub["EDUC"].map(edu_map)

# --- Derived outcome: trust gap (positive => leans traditional, negative => leans local gov) ---
sub["trust_gap"] = sub["trust_traditional"] - sub["trust_local_gov"]
sub["leans_traditional"] = (sub["trust_gap"] > 0).astype("Int64")
sub.loc[sub["trust_gap"].isna(), "leans_traditional"] = pd.NA

keep = [
    "trust_local_gov", "trust_traditional", "trust_gap", "leans_traditional",
    "corrupt_local_gov", "corrupt_traditional", "perf_local_gov",
    "contact_freq_local_gov", "contact_freq_traditional",
    "contacted_local_gov", "contacted_traditional",
    "urban", "region", "age", "female", "education_level", "lived_poverty",
]
clean = sub[keep].copy()
clean.to_csv(OUT, index=False)
print("Saved:", OUT, clean.shape)
print(clean.isna().mean().round(3))
