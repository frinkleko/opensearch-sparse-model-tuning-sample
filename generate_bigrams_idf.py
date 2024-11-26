import numpy as np
import json
from collections import Counter
from tqdm import tqdm
import pandas as pd
from datasets import load_dataset
from transformers import BertTokenizer

def tokenize_function(examples):
    return {"tokenized_text": tokenizer.tokenize(examples["text"])}

def find_df_bigram(df):
    bi_gram = [f"{df[i]} {df[i+1]}" for i in range(len(df) - 1)]
    bi_gram_set = set(bi_gram)
    return bi_gram_set

def calculate_idf_bigrams(bigram_df, output_file):
    """
    Calculates the IDF values for bi-grams in the given DataFrame,
    where each row contains a list of bigrams. Saves the IDF scores
    to a JSON file after processing every 10,000 documents.

    Args:
        bigram_df (pandas.DataFrame): A DataFrame with a column 'df' containing lists of bigrams.
        output_file (str): The path to the output JSON file.

    Returns:
        dict: A dictionary mapping bi-grams to their IDF values.
    """

    # Create a set to store unique bi-grams
    unique_bigrams = set()

    # Initialize variables for document frequency counting
    total_docs = len(bigram_df)
    bigram_doc_freqs = Counter()

    # Initialize IDF scores dictionary
    idf_scores = {}

    # Process documents in batches of 10,000, just to auto save the results
    for batch in tqdm([bigram_df.iloc[i:i+10000] for i in range(0, len(bigram_df), 10000)]):
        # Calculate document frequencies for bi-grams in the batch
        for row in batch['df']:
            unique_bigrams.update(row)
            bigram_doc_freqs.update(row)

        # Calculate IDF scores for bi-grams in the batch
        batch_idf_scores = {bigram: np.log(total_docs / doc_freq) for bigram, doc_freq in bigram_doc_freqs.items()}

        # Update the overall IDF scores dictionary
        idf_scores.update(batch_idf_scores)

        # Save IDF scores to JSON file
        with open(output_file, 'w') as f:
            json.dump(idf_scores, f)

    return idf_scores

if __name__ == "__main__":
    import os

    if os.path.exists("tokenized_ds.feather"):
        tokenized_ds = pd.read_feather("tokenized_ds.feather")
    else:
        ds = load_dataset("BeIR/msmarco", "corpus")
        tokenizer = BertTokenizer.from_pretrained('distilbert/distilbert-base-uncased')
        print("Tokenizing the dataset")
        ds = ds['corpus'].select(range(8841823))
        tokenized_ds = ds.map(tokenize_function, num_proc=60)
        tokenized_ds = tokenized_ds.to_pandas()
        tokenized_ds.to_feather("tokenized_ds.feather")

    if os.path.exists("bigram_df.feather"):
        df_ds = pd.read_feather("bigram_df.feather")
    else:
        df_ds = tokenized_ds.copy()
        df_ds['df'] = df_ds['tokenized_text'].apply(find_df_bigram)
        df_ds.to_feather("bigram_df.feather")

    output_file = "bigrams_idf.json"
    idf_scores = calculate_idf_bigrams(df_ds, output_file)
