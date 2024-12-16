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

# async def ingest(
#     dataset: Dataset,
#     model: SparseModel,
#     out_dir: str,
#     index_name: str,
#     accelerator: Accelerator,
#     max_length: int = 512,
#     batch_size: int = 50,
# ):
#     os_client = get_os_client()
#     os.makedirs(out_dir, exist_ok=True)
#     if isinstance(dataset, DDPDatasetWithRank):
#         logger.error("Input dataset can not be DDPDatasetWithRank.")
#         raise RuntimeError("Input dataset can not be DDPDatasetWithRank.")
#     ddp_dataset = DDPDatasetWithRank(
#         dataset, accelerator.local_process_index, accelerator.num_processes
#     )
#     logger.info(
#         f"Local rank: {accelerator.local_process_index}, index_name: {index_name}, sample number: {len(ddp_dataset)}"
#     )
#     dataloader = DataLoader(ddp_dataset, batch_size=batch_size)

#     accelerator.prepare(model)
#     sparse_encoder = SparseEncoder(
#         sparse_model=model,
#         max_length=max_length,
#         do_count=True,
#     )

#     # do model encoding and ingestion
#     # use async io so we don't need to wait every bulk return
#     # we send out 20 bulk request, then wait all of them return
#     tasks = []
#     timeout = ClientTimeout(total=600)
#     async with aiohttp.ClientSession(timeout=timeout) as session:
#         for ids, texts in tqdm(dataloader):
#             output = sparse_encoder.encode(texts)

#         await asyncio.gather(*tasks)

#     sparse_encoder.count_tensor = sparse_encoder.count_tensor.reshape(1, -1)
#     accelerator.wait_for_everyone()
#     all_count_tensor = accelerator.gather(sparse_encoder.count_tensor)


def main():
    model_args, data_args, training_args = parse_args()
    stat = dict()
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
        # if the dataset wasn't download before, only download it on main process
        if not os.path.exists(
        os.path.join(data_args.beir_dir, dataset)
        ):
            url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
            data_path = util.download_and_unzip(url, data_args.beir_dir)
        accelerator.wait_for_everyone()
        data_path = os.path.join(data_args.beir_dir, dataset)
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(
            split="test"
        )

        
        for ids, values in tqdm(corpus.items()):
            text = values['text']
            encoded_doc = encoder.encode(text)
            stat[ids] = encoded_doc
            if len(stat) > 10:
                break
        
        df_dict = {
            "id": list(stat.keys()),
            "value": list(stat.values())
        }
        df = pd.DataFrame(df_dict)
        df.to_feather(f"{data_args.beir_dir}/{dataset}_encoded.feather")
        break




if __name__ == "__main__":
    main()
