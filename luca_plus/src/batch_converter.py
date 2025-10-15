import sys
import torch
from typing import Sequence


class BatchConverter(object):
    def __init__(self,
                 label_size,

                 seq_subword,
                 seq_tokenizer,
                 no_position_embeddings,
                 np_token_type_embeddings,
                 truncation_seq_length:int = None,
                 truncation_matrix_length:int = None,
                 padding_idx: int = 0,
                 unk_idx:int = 1,
                 cls_idx:int = 2,
                 eos_idx: int = 3,
                 mask_idx: int = 4,
                 prepend_bos = None,
                 append_eos = None

                 ):
        self.seq_subword = seq_subword
        self.label_size = label_size
        self.truncation_seq_length = truncation_seq_length
        self.seq_tokenizer = seq_tokenizer
        self.no_position_embeddings = no_position_embeddings
        self.padding_idx = padding_idx
        self.unk_idx = unk_idx
        self.cls_idx = cls_idx
        self.eos_idx = eos_idx
        self.mask_idx = mask_idx

        self.truncation_matrix_length = truncation_matrix_length
        self.no_token_type_embeddings = np_token_type_embeddings
        if prepend_bos is None:
            if seq_subword is not None:
                self.prepend_bos = True
            else:
                self.prepend_bos = False

        if append_eos is None:
            if seq_subword is not None:
                self.append_eos = True
            else:
                self.append_eos = False

        if self.seq_tokenizer is None:
            self.append_len = 0
        else:
            if hasattr(self.seq_tokenizer,"prepend_bos") and self.seq_tokenizer.prepend_bos is not None:
                self.seq_prepend_nos = self.seq_tokenizer.prepend_bos
            if hasattr(self.seq_tokenizer,"prepend_eos") and self.seq_tokenizer.seq_prepend_nos is not None:
                self.seq_append_eos = self.seq_tokenizer.append_eos

            if hasattr(self.seq_tokenizer,"padding_idx") and self.seq_tokenizer.padding_idx is not None:
                self.padding_idx = self.seq_tokenizer.padding_idx

            if hasattr(self.seq_tokenizer,"unk_idx") and self.seq_tokenizer.unk_idx is not None:
                self.unk_idx = self.seq_tokenizer.unk_idx

            if hasattr(self.seq_tokenizer,"cls_idx") and self.seq_tokenizer.cls_idx is not None:
                self.cls_idx = self.seq_tokenizer.cls_idx


            if hasattr(self.seq_tokenizer,"mask_idx") and self.seq_tokenizer.mask_idx is not None:
                self.mask_idx = self.seq_tokenizer.mask_idx

            if hasattr(self.seq_tokenizer, "all_special_token_idx_list"):
                self.all_special_token_idx_list = self.seq_tokenizer.all_special_token_idx_list
            else:
                self.all_special_token_idx_list = [self.padding_idx, self.unk_idx, self.cls_idx, self.eos_idx,
                                                   self.mask_idx]
            self.append_len = int(self.prepend_bos) + int(self.append_eos)


        self.truncation_seq_length -= self.append_len
        self.truncation_matrix_length -= self.append_len







    def __seq_encode__(self,batch_size,seqs):
        seq_encoded_list = []
        if self.seq_subword:

            for seq_str in seqs:
                seq_to_list = self.seq_subword.process_line(seq_str.upper()).split(" ")
                seq = " ".join(seq_to_list)
                inputs = self.seq_tokenizer.encode_plus(
                    seq,
                    None,
                    add_special_tokens=False,
                    max_length=self.truncation_seq_length,
                    truncation=True
                )
                seq_encoded_list.append(inputs['input_ids'])
        else:
            seq_encoded_list = [self.seq_tokenizer.encode(seq_str.upper()) for seq_str in seqs]
            if self.truncation_seq_length:
                seq_encoded_list = [encoded[:self.truncation_seq_length] for encoded in seq_encoded_list]

        max_len = max(len(seq_encoded) for seq_encoded in seq_encoded_list)
        max_len = max_len + int(self.prepend_bos) + int(self.append_eos)

        input_ids = torch.empty(
            (batch_size,
             max_len,
             ),
            dtype = torch.int64,
        )

        position_ids = None
        if not self.no_position_embeddings:
            position_ids = torch.empty(
                (
                    batch_size,
                    max_len,
                ),
                dtype = torch.int64
            )
            position_ids.fill_(self.padding_idx)

        token_type_ids = None
        if not self.no_position_embeddings:
            token_type_ids = torch.empty(
                (
                    batch_size,
                    max_len,
                ),
                dtype = torch.int64
            )
            token_type_ids.fill_(self.padding_idx)

        attention_masks = torch.empty(
            (
                batch_size,
                max_len,
            ),
            dtype = torch.int64
        )
        attention_masks.fill_(0)

        return seq_encoded_list, input_ids, position_ids, token_type_ids, attention_masks, max_len




    def __parse_label__(self,label_size,label):
        new_label = [0]*label_size
        if label is not None:
            new_label[int(label)] = 1
        return new_label


    def __vector_encode__(self,batch_size,vectors):
        embedding_vector_dim = vectors[0].shape[0]
        filled_vectors = torch.empty(
            (
                batch_size,
                embedding_vector_dim
            ),
            dtype = torch.float32,
        )
        filled_vectors.fill_(0.0)
        return filled_vectors, 1


    def __matrix_encode__(self,batch_size,matrices):
        max_len = max(matrix.shape[0] for matrix in matrices)
        if self.truncation_matrix_length:
            max_len = min(max_len,self.truncation_matrix_length)

        max_len = max_len + int(self.prepend_bos) + int(self.append_eos)

        embedding_vector_dim = matrices[0].shape[1]

        filled_matrices = torch.empty(
            (
                batch_size,
                max_len,
                embedding_vector_dim
            ),
            dtype=torch.float32,
        )
        filled_matrices.fill_(0.0)
        attention_masks = torch.empty(
            (
                batch_size,
                max_len,
            ),
            dtype=torch.int64,
        )
        attention_masks.fill_(0)
        return filled_matrices, attention_masks, max_len




    def __call_single__(self,batch_size,seq_types,seqs,vectors,matrices,labels):
        max_length = sys.maxsize
        input_ids,position_ids,token_type_ids,seq_attention_masks = None,None, None, None
        seq_part_if_input = False
        if seqs:
            new_seqs = []
            for seq_idx,seq_type in enumerate(seq_types):
                new_seqs.append(seqs[seq_idx].upper())

            seq_encoded_list,input_ids,position_ids,token_type_ids,seq_attention_masks,seq_max_length = self.__seq_encode__(batch_size = batch_size,seqs = new_seqs)
            max_length = min(max_length,seq_max_length)
            seq_part_if_input = True
        else:
            seq_encoded_list = None

        encoded_vectors = None
        vector_part_of_input = False
        if vectors:
            encoded_vectors,vector_max_length = self.__vector_encode__(batch_size = batch_size,vectors = vectors)
            vector_part_of_input = True

        encoded_matrices,matrix_attention_masks = None,None
        matrix_part_of_input = False
        if matrices:
            encoded_matrices,matrix_attention_masks,matrix_max_length = self.__matrix_encode__(
                batch_size = batch_size,matrices = matrices
            )
            max_length = min(max_length,matrix_max_length)
            matrix_part_of_input = True


        has_label = False
        if labels:
            has_label = True

        new_labels = []
        for sample_idx in range(batch_size):
            if seq_part_if_input:
                if self.prepend_bos:
                    input_ids[sample_idx,0] = self.cls_idx
                seq_encoded = seq_encoded_list[sample_idx]
                real_seq_len = len(seq_encoded)

                seq_tensor = torch.tensor(seq_encoded,dtype=torch.int64)
                input_ids[sample_idx,int(self.prepend_bos):real_seq_len + int(self.prepend_bos)] = seq_tensor

                if self.append_eos:
                    input_ids[sample_idx,real_seq_len + int(self.prepend_bos)] = self.eos_idx
                cur_len = int(self.prepend_bos) + real_seq_len + int(self.append_eos)

                if not self.no_position_embeddings:
                    for pos_idx in range(0,cur_len):
                        position_ids[sample_idx,pos_idx] = pos_idx

                if not self.no_token_type_embeddings:
                    for pos_idx in range(0, cur_len):
                        token_type_ids[sample_idx, pos_idx] = 1

                seq_attention_masks[sample_idx,0:cur_len] = 1

            if vector_part_of_input:
                encoded_vectors[sample_idx,:] = torch.tensor(vectors[sample_idx],dtype = torch.float32)

            if matrix_part_of_input:
                matrix_encoded = matrices[sample_idx]
                real_seq_len = matrix_encoded.shape[0]

                real_seq_len = min(real_seq_len,self.truncation_matrix_length)
                matrix = torch.tensor(matrix_encoded,dtype=torch.float32)
                encoded_matrices[sample_idx,int(self.prepend_bos):real_seq_len + int(self.prepend_bos)] = matrix[0:real_seq_len]
                matrix_attention_masks[sample_idx,int(self.prepend_bos):real_seq_len + int(self.prepend_bos)] = 1


            if has_label:
                new_labels.append(self.__parse_label__(self.label_size,labels[sample_idx]))

        if new_labels is not None and new_labels:
            labels = torch.tensor(new_labels,dtype = torch.int64)

        else:
            labels = None



        return input_ids,position_ids,token_type_ids,seq_attention_masks,encoded_vectors,encoded_matrices,matrix_attention_masks,labels




    def __call__(self,raw_batch:Sequence[dict]):
        batch_size = len(raw_batch)
        res = {}
        seq_types = []
        seqs = []
        vectors = []
        matrices = []
        labels = []
        for item in raw_batch:
            seq_types.append(item['seq_type'])
            if item["seq"] is not None:
                seqs.append(item["seq"])
            if item["vector"] is not None:
                vectors.append(item["vector"])
            if item["matrix"] is not None:
                matrices.append(item["matrix"])
            if item["label"] is not None:
                labels.append(item["label"])

        input_ids,position_ids,token_type_ids,seq_attention_masks,encoded_vectors,encoded_matrices,matrix_attention_masks,labels \
        = self.__call_single__(batch_size,seq_types,seqs,vectors,matrices,labels)

        res.update({
            "input_ids": input_ids,
            "position_ids": position_ids,
            "token_type_ids": token_type_ids,
            "seq_attention_masks": seq_attention_masks,
            "vectors": encoded_vectors,
            "matrices": encoded_matrices,
            "matrix_attention_masks": matrix_attention_masks,
            "labels": labels if labels is not None and len(labels) > 0 else None
        })
        return res
