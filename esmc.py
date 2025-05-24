from esm.sdk import client
from getpass import getpass
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence
import os
from tqdm import tqdm
from time import sleep
import pickle
import torch


from esm.sdk.api import (
    ESM3InferenceClient,
    ESMProtein,
    ESMProteinError,
    LogitsConfig,
    LogitsOutput,
    ProteinType,
)




def read_seq(seqfilepath):
    with open(seqfilepath, "r") as f:
        lines = f.readlines()
        # 跳过第一行（通常是描述行）
        seq = "".join(line.strip() for line in lines[1:])
    return seq


def embed_sequence(model: ESM3InferenceClient, protein_id: str, sequence: str) -> LogitsOutput:
    protein = ESMProtein(sequence=sequence)
    while True:
        protein_tensor = model.encode(protein)
        if isinstance(protein_tensor, ESMProteinError):
            print(protein_tensor)
            sleep(1)
            continue
        break
    while True:
        logits_output = model.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
        if isinstance(logits_output, ESMProteinError):
            print(logits_output)
            sleep(1)
            continue
        break
    # 去掉批量维度
    embeddings = logits_output.embeddings.squeeze(0)[1:-1]
    return protein_id, embeddings

def batch_embed(model: ESM3InferenceClient, inputs, embedding_dir):
    error_list = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [
            executor.submit(embed_sequence, model, protein_id, inputs[protein_id]) for protein_id in inputs.keys()
        ]
        all = len(futures)
        for i, future in enumerate(futures):
            try:
                protein_id, emb = future.result()
                print(f"Embedding shape for {protein_id}: {emb.shape}")
                # 创建目标目录
                target_dir = os.path.join(embedding_dir, protein_id)
                os.makedirs(target_dir, exist_ok=True)
                # 保存文件
                with open(os.path.join(target_dir, "origin_seq_emb_6b.pkl"), "wb") as f:
                    pickle.dump(emb, f)
                print(i, "/", all, " Success ", protein_id)
            except Exception as e:
                print(i, "/", all, f" Error: {e}")
                error_list.append(protein_id)
    return error_list

token = getpass("Token from Forge console: ")
model = client(model="esmc-6b-2024-12", url="https://forge.evolutionaryscale.ai", token=token)

seq_dict = {}
data_dir = r'D:\MsgaBpred\data\Data_245\fasta'
embedding_dir = r"D:\MsgaBpred\data\Data_245\esmc"
for protein_id in tqdm(os.listdir(data_dir)):
    '''seq_path = os.path.join(data_dir, protein_id, "seq.fasta")'''
    seq_path = os.path.join(data_dir, protein_id)
    if not os.path.isfile(seq_path):
        continue

    target_file = os.path.join(embedding_dir, protein_id, "origin_seq_emb_6b.pkl")
    if os.path.exists(target_file):
        print(f"Skipping {protein_id} as embedding already exists.")
        continue
    seq = read_seq(seq_path)
    if len(seq) > 2048:
        continue
    if os.path.exists(os.path.join(embedding_dir, protein_id, "origin_seq_emb_6b.pkl")):
        continue
    seq_dict[protein_id] = read_seq(seq_path)

error_list = batch_embed(model, seq_dict, embedding_dir)

import json

with open("error_list.json", "w") as f:
    json.dump(error_list, f, indent=4)