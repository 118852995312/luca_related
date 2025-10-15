#!/usr/bin/env python
# encoding: utf-8

import os
import sys
import torch
from esm import pretrained,BatchConverter
sys.path.append(".")
sys.path.append("..")
sys.path.append("../../")
sys.path.append("../../../")
sys.path.append("../../../src")
from src.utils import clean_seq

global_model,global_alphabet,global_version,global_layer_size = None,None,None,None


def predict_embedding_esm(sample,trunc_type,embedding_type,repr_layers=[-1],truncation_seq_length = 409,
                      device=None,version="3B",matrix_add_special_roken = False):
    '''
    :param sample:
    :param trunc_type:
    :param embedding_type:
    :param repr_layers:
    :param truncation_seq_length:
    :param device:
    :param version:
    :param matrix_add_special_roken:
    :return:
    '''
    global global_model,global_alphabet,global_version,global_layer_size
    assert "bos" in embedding_type or "representations" in embedding_type or \
            "matrix" in embedding_type or "vector" in embedding_type or "contacts" in embedding_type
    if len(sample) > 2:
        protein_id,protein_seq = sample[0],sample[2]
    else:
        protein_id, protein_seq = sample[0], sample[1]

    protein_seq = clean_seq(protein_id,protein_seq)

    if len(protein_seq) > truncation_seq_length:
        if trunc_type == "left":
            protein_seq = protein_seq[-truncation_seq_length:]
        else:
            protein_seq = protein_seq[:truncation_seq_length]


    if global_model is None or global_alphabet is None or global_version is None or global_version != version or global_layer_size is None:
        if version == "15B":
            llm_name = "esm2_t48_15B_UR50D"
            global_layer_size = 48
            global_model,global_alphabet = pretrained.load_model_and_alphabet(llm_name)
        elif version == "3B":
            llm_name = "esm2_t48_3B_UR50D"
            global_layer_size = 36
            global_model, global_alphabet = pretrained.load_model_and_alphabet(llm_name)
        elif version == "650M":
            llm_name = "esm2_t48_650M_UR50D"
            global_layer_size = 33
            global_model, global_alphabet = pretrained.load_model_and_alphabet(llm_name)
        elif version == "150M":
            llm_name = "esm2_t48_150M_UR50D"
            global_layer_size = 30
            global_model, global_alphabet = pretrained.load_model_and_alphabet(llm_name)
        else:
            raise Exception("not support this version=%s" % version)

        global_version = version

    if device is None:
        device = next(global_model.parameters()).device
    else:
        model_device = next(global_model.parameters()).device

        if device != model_device:
            global_model = global_model.to(device)

    assert all(-(global_model.num_layers + 1) <= i <= global_model.num_layers for i in repr_layers)
    repr_layers = [(i + global_model.num_layers + 1) % (global_model.num_layers + 1) for i in repr_layers]
    global_model.eval()

    converter = BatchConverter(global_alphabet,truncation_seq_length)

    protein_ids,raw_seqs,tokens = converter([[protein_id,protein_seq]])

    embeddings = {}
    with torch.no_grad():
        tokens = tokens.to(device = device,non_blocking=True)
        try:
            out = global_model(tokens,repr_layers = repr_layers,return_contacts = False)
            truncate_len = min(truncation_seq_length,len(raw_seqs[0]))
            processed_seq_len = truncate_len + 2
            if "representations" in embedding_type or "matrix" in embedding_type:
                if matrix_add_special_roken:
                    embedding = out["representations"][global_layer_size].to(device="cpu")[0, 0: truncate_len + 2].clone().numpy()
                else:
                    embedding = out["representations"][global_layer_size].to(device="cpu")[0, 1: truncate_len + 1].clone().numpy()
                embeddings['representations'] = embedding

            if "bos" in embedding_type or "vector" in embedding_type:
                embedding = out["representations"][global_layer_size].to(device="cpu")[0, 0].clone().numpy()
                embeddings["bos_representations"] = embedding
            if "contacts" in embedding_type:
                embedding = out["contacts"][global_layer_size].to(device="cpu")[0, :, :].clone().numpy()
                embeddings["contacts"] = embedding
            if len(embeddings) > 1:
                return embeddings,processed_seq_len
            elif len(embeddings) == 1:
                return list(embeddings.items())[0][1], processed_seq_len
            else:
                return None, None
        except RuntimeError as e:
            if e.args[0].startswith("CUDA out of memory"):
                print(f"Failed (CUDA out of memory) on sequence {protein_id} of length {len(protein_seq)}.")
                print("Please reduce the 'truncation_seq_length'")
            raise Exception(e)
    return None, None