import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, AutoModel


MODEL_NAME = "facebook/esm2_t6_8M_UR50D"


def mean_pool_embedding(sequence, tokenizer, model, device):
    inputs = tokenizer(
        sequence,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    token_embeddings = outputs.last_hidden_state[0]

    # Remove special tokens: first and last token
    residue_embeddings = token_embeddings[1:-1]

    protein_embedding = residue_embeddings.mean(dim=0)

    return protein_embedding.cpu().numpy()


def main():
    Path("results").mkdir(exist_ok=True)

    df = pd.read_csv("data/protein_sequences.csv")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    embeddings = []

    for sequence in tqdm(df["sequence"], desc="Embedding sequences"):
        emb = mean_pool_embedding(sequence, tokenizer, model, device)
        embeddings.append(emb)

    X = np.vstack(embeddings)
    y = df["label"].values

    np.save("results/esm2_embeddings.npy", X)
    np.save("results/labels.npy", y)

    df.to_csv("results/metadata.csv", index=False)

    print("Saved embeddings to results/esm2_embeddings.npy")
    print("Embedding matrix shape:", X.shape)


if __name__ == "__main__":
    main()