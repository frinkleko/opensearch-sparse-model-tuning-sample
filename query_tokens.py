import torch.utils
import torch.utils.data
import yaml
import logging
import os
import json
import random
import sys
import torch
from dataclasses import dataclass, field, asdict
from typing import Optional
from tqdm import tqdm

import pandas as pd
import asyncio

from scripts.model.sparse_encoders import (
    SparseModel,
    SparseEncoder,
    sparse_embedding_to_query,
)
from scripts.data.dataset import BEIRCorpusDataset, KeyValueDataset
from scripts.ingest import ingest
from scripts.utils import get_os_client, batch_search, set_logging, get_model
from scripts.args import ModelArguments, DataTrainingArguments, parse_args

from transformers import (
    set_seed,
)


from accelerate import Accelerator
from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader

logger = logging.getLogger(__name__)

def main():
    model_args, data_args, training_args = parse_args()
    args_dict = {
        "model_args": asdict(model_args),
        "data_args": asdict(data_args),
        "training_args": asdict(training_args),
    }

    set_seed(training_args.seed)

    model = get_model(model_args)

    accelerator = Accelerator(mixed_precision="fp16")
    accelerator.wait_for_everyone()

    datasets = data_args.beir_datasets.split(",")

    max_length = 512
    encoder = SparseEncoder(
        sparse_model=model,
        max_length=max_length,
        do_count=True,
    )
    for dataset in datasets:
        stat = dict()
        # if the dataset wasn't download before, only download it on main process
        if not os.path.exists(os.path.join(data_args.beir_dir, dataset)):
            url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
            try:
                data_path = util.download_and_unzip(url, data_args.beir_dir)
            except Exception as e:
                continue
        accelerator.wait_for_everyone()
        data_path = os.path.join(data_args.beir_dir, dataset)
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(
            split="test"
        )

        dataset_warp = BEIRCorpusDataset(corpus=corpus)
        dataloader = torch.utils.data.DataLoader(
            dataset_warp, batch_size=1, shuffle=False, num_workers=0
        )
        for ids, texts in tqdm(dataloader):
            encoded_doc = encoder.encode(texts)
            if None in encoded_doc[0].values():
                print(encoded_doc)
                break
            stat[ids] = encoded_doc[0]
            if len(stat) > 100:
                break

        df = pd.DataFrame(stat)
        df.to_feather(f"{data_args.beir_dir}/{dataset}_encoded.feather")
        break


if __name__ == "__main__":
    main()
