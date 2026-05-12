import requests
import pandas as pd
from io import StringIO
from pathlib import Path


BASE_URL = "https://rest.uniprot.org/uniprotkb/search"


def fetch_uniprot(query: str, size: int = 300) -> pd.DataFrame:
    params = {
        "query": query,
        "format": "tsv",
        "fields": "accession,protein_name,sequence,length",
        "size": size,
    }

    response = requests.get(BASE_URL, params=params, timeout=60)
    response.raise_for_status()

    return pd.read_csv(StringIO(response.text), sep="\t")


def clean_df(df: pd.DataFrame, label: int) -> pd.DataFrame:
    df = df.rename(
        columns={
            "Entry": "id",
            "Protein names": "protein_name",
            "Sequence": "sequence",
            "Length": "length",
        }
    )

    df = df.dropna(subset=["sequence"])
    df = df[(df["length"] >= 50) & (df["length"] <= 800)]
    df["label"] = label

    return df[["id", "protein_name", "sequence", "length", "label"]]


def main():
    Path("data").mkdir(exist_ok=True)

    membrane_query = (
        "reviewed:true AND keyword:Transmembrane "
        "AND length:[50 TO 800]"
    )

    non_membrane_query = (
        "reviewed:true NOT keyword:Transmembrane "
        "AND length:[50 TO 800]"
    )

    membrane = fetch_uniprot(membrane_query, size=400)
    non_membrane = fetch_uniprot(non_membrane_query, size=400)

    membrane = clean_df(membrane, label=1)
    non_membrane = clean_df(non_membrane, label=0)

    df = pd.concat([membrane, non_membrane], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    output_path = "data/protein_sequences.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df)} sequences to {output_path}")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()