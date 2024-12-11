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


async def just_encode_query(
    queries: dict,
    model: SparseModel,
    out_dir: str,
    index_name: str,
    max_length: int = 512,
    batch_size: int = 50,
    result_size: int = 15,
    inf_free: bool = True,
    delete: bool = False,
):
    os.makedirs(out_dir, exist_ok=True)

    queries_dataset = KeyValueDataset(queries)
    dataloader = torch.utils.data.DataLoader(queries_dataset, batch_size=batch_size)

    query_encoder = SparseEncoder(
        sparse_model=model,
        max_length=max_length,
        do_count=True,
    )

    encoded_queries = []
    for ids, texts in tqdm(dataloader):
        queries_encoded = query_encoder.encode(texts, inf_free=inf_free)
        encoded_queries.extend(queries_encoded)

    if delete:
        client = get_os_client()
        client.indices.delete(index_name, params={"timeout": 1000})

    return encoded_queries

def main():
    model_args, data_args, training_args = parse_args()

    args_dict = {
        "model_args": asdict(model_args),
        "data_args": asdict(data_args),
        "training_args": asdict(training_args),
    }
    beir_eval_dir = os.path.join(training_args.output_dir, "beir_eval")
    os.makedirs(beir_eval_dir, exist_ok=True)
    with open(
        os.path.join(training_args.output_dir, "beir_eval", "config.yaml"), "w"
    ) as file:
        yaml.dump(args_dict, file, sort_keys=False)

    set_logging(training_args, "eval_beir.log")
    set_seed(training_args.seed)

    model = get_model(model_args)

    accelerator = Accelerator(mixed_precision="fp16")
    accelerator.wait_for_everyone()

    datasets = data_args.beir_datasets.split(",")

    for dataset in datasets[0]:
        # if the dataset wasn't download before, only download it on main process
        print('dataset:', dataset)
        if accelerator.is_local_main_process and not os.path.exists(
            os.path.join(data_args.beir_dir, dataset)
        ):
            url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
            data_path = util.download_and_unzip(url, data_args.beir_dir)
        accelerator.wait_for_everyone()
        data_path = os.path.join(data_args.beir_dir, dataset)
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(
            split="test"
        )

        asyncio.run(
            ingest(
                dataset=BEIRCorpusDataset(corpus=corpus),
                model=model,
                out_dir=beir_eval_dir,
                index_name=dataset,
                accelerator=accelerator,
                max_length=data_args.max_seq_length,
                batch_size=training_args.per_device_eval_batch_size,
            )
        )

        # search is only run on main process
        if accelerator.is_local_main_process:
            encoded_quires = asyncio.run(
                just_encode_query(
                    queries=queries,
                    model=model,
                    out_dir=beir_eval_dir,
                    index_name=dataset,
                    max_length=data_args.max_seq_length,
                    batch_size=training_args.per_device_eval_batch_size,
                    inf_free=model_args.inf_free,
                )
            )

        # save the encoded queries
        if accelerator.is_local_main_process:
            with open(
                os.path.join(beir_eval_dir, f"{dataset}_encoded_queries.json"), "w"
            ) as f:
                json.dump(encoded_quires, f)

        accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
