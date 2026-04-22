import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset")

FEATURES = [
    "total_funding", "num_rounds", "avg_round_size", "max_round_size",
    "days_to_first_funding", "funding_duration_days", "founded_year",
    "country_code_enc", "category_code_enc",
]

def load_and_build(data_path=DATA_PATH):
    objects        = pd.read_csv(f"{data_path}/objects.csv",        encoding="latin-1", low_memory=False)
    funding_rounds = pd.read_csv(f"{data_path}/funding_rounds.csv", encoding="latin-1", low_memory=False)
    acquisitions   = pd.read_csv(f"{data_path}/acquisitions.csv",   encoding="latin-1", low_memory=False)
    ipos           = pd.read_csv(f"{data_path}/ipos.csv",           encoding="latin-1", low_memory=False)

    df = objects[objects["entity_type"] == "Company"].copy()
    df = df[df["status"].isin(["operating", "acquired", "closed", "ipo"])]
    df["survived"]        = (df["status"] != "closed").astype(int)
    df["had_acquisition"] = df["id"].isin(acquisitions["acquired_object_id"].dropna()).astype(int)
    df["had_ipo"]         = df["id"].isin(ipos["object_id"].dropna()).astype(int)

    funding_rounds["funded_at"] = pd.to_datetime(funding_rounds["funded_at"], errors="coerce")
    fagg = funding_rounds.groupby("object_id").agg(
        total_funding  =("raised_amount_usd", "sum"),
        num_rounds     =("id", "count"),
        avg_round_size =("raised_amount_usd", "mean"),
        max_round_size =("raised_amount_usd", "max"),
        first_funded   =("funded_at", "min"),
        last_funded    =("funded_at", "max"),
    ).reset_index()

    df = df.merge(fagg, left_on="id", right_on="object_id", how="left")

    df["founded_at"]           = pd.to_datetime(df["founded_at"], errors="coerce")
    df["days_to_first_funding"] = (df["first_funded"] - df["founded_at"]).dt.days
    df["funding_duration_days"] = (df["last_funded"]  - df["first_funded"]).dt.days
    df["founded_year"]          = df["founded_at"].dt.year

    encoders = {}
    for col, n in [("country_code", 10), ("category_code", 15)]:
        top = df[col].value_counts().nlargest(n).index
        df[col] = df[col].where(df[col].isin(top), other="OTHER").fillna("UNKNOWN")
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col])
        encoders[col] = le

    return df, encoders


def get_xy(df):
    data = df[FEATURES + ["survived"]].dropna(subset=["survived"])
    return data[FEATURES], data["survived"]


def encode_single(input_dict, encoders):
    for col in ["country_code", "category_code"]:
        le = encoders[col]
        val = input_dict.get(col, "UNKNOWN")
        if val not in le.classes_:
            val = "OTHER"
        input_dict[col + "_enc"] = int(le.transform([val])[0])
    return input_dict


def save_encoders(encoders, path="models/encoders.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(encoders, f)


def load_encoders(path="models/encoders.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)
