#!/usr/bin/env python
# encoding: utf-8
import numpy as np
import torch
import os
from embeddings import predict_embedding_esm
class Encoder(object):
    def __init__(self,
                 input_type,
                 trunc_type,
                 seq_max_length,
                 llm_type,
                 device,
                 prepend_bos=True,
                 append_eos=True,
                 ):


        self.device = device
        self.input_type = input_type
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos
        self.seq_max_length = seq_max_length
        self.llm_type = llm_type
        self.trunc_type = trunc_type



    def encoder_single(self,seq_id,seq_type,seq,vector_filename = None,
                       matrix_filename = None,label = None):
        seq_type = seq_type.strip().lower()

        vector = None
        if self.input_type in ["vector","seq_vector"]:
            if vector_filename is None:
                if seq is None:
                    raise Exception("seq is none and vector_filename is none")
                elif seq_type not in ['protein',"prot","gene"]:
                    raise Exception("now not support embedding of the seq_type=%s" % seq_type)
                else:
                    vector = self.__get_embedding__(seq_id,seq_type,seq,"vector")

            elif isinstance(vector_filename,str):
                vector = torch.load(os.path.join(self.vector_dirpath, vector_filename))

            elif isinstance(vector_filename,np.ndarray):
                vector = vector_filename
            else:
                raise Exception("vector is not filepath-str and np.ndarray")

        matrix = None
        if self.input_type in ["matrix", "seq_matrix"]:
            if matrix_filename is None:
                if seq is None:
                    raise Exception("seq is none and matrix_filename is none")
                elif seq_type not in ["protein", "prot", "gene"]:
                    raise Exception("now not support embedding of the seq_type=%s" % seq_type)
                else:
                    matrix = self.__get_embedding__(seq_id, seq_type, seq, "matrix")
            elif isinstance(matrix_filename, str):
                matrix = torch.load(os.path.join(self.matrix_dirpath, matrix_filename))
            elif isinstance(matrix_filename, np.ndarray):
                matrix = matrix_filename
            else:
                raise Exception("matrix is not filepath-str and np.ndarray")

        seq = seq.upper()
        return {
            "seq_id": seq_id,
            "seq": seq,
            "seq_type": seq_type,
            "vector": vector,
            "matrix": matrix,
            "label": label
        }



    def __get_embedding__(self,seq_id,seq_type,seq,embedding_type):
        seq_type = seq_type.strip().lower()
        if "prot" not in seq_type and "gene" not in seq_type:
            raise Exception("Not support this seq_type=%s" % seq_type)
        embedding_info = None
        truncation_seq_length = self.seq_max_length - int(self.prepend_bos) - int(self.append_eos)
        truncation_seq_length = min(len(seq), truncation_seq_length)
        if self.llm_type == "esm":
            embedding_info,processed_seq_len = predict_embedding_esm(
                [seq_id,seq],
            self.trunc_type,
            embedding_type,repr_layers=[-1],
            truncation_seq_length = truncation_seq_length,
            device=self.device)
        else:
            raise Exception("Not support the llm_type=%s" % self.llm_type)

        return embedding_info



