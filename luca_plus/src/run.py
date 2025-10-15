import torch
from models import Encoder,load_model
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data.dataloader import DataLoader
from batch_converter import BatchConverter
import torch.distributed as dist
import logging
from trainer import train
import os
from types import SimpleNamespace
from tester import test
logger = logging.getLogger(__name__)
from transformers.configuration_utils import PretrainedConfig
def create_device(args):
    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device("cpu")
        args.n_gpu = 0

    else:
        args.n_gpu = torch.cuda.device_count()
        if args.n_gpu > 1:
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda",args.local_rank)
        else:
            device = torch.device("cuda")
    return device



class get_args:
    def __init__(self):
        pass




def load_train_data(args,parse_row_func):
    train_dataset = load_dataset('csv',
                                 data_dir=args.train_data_dir,
                                 split='train',
                                 streaming=True)
    train_dataset = train_dataset.map(
        lambda x: parse_row_func(
            x["seq_id"],
            x["seq_type"] if "seq_type" in x else "prot",
            x["seq"],
            x["vector_filename"] if "vector_filename" in x else None,
            x["matrix_filename"] if "matrix_filename" in x else None,
            x["label"] if "label" in x else None,
        ),
        batched=False
    )
    train_dataset = split_dataset_by_node(train_dataset, rank=args.local_rank, world_size=dist.get_world_size()) \
        .shuffle(buffer_size=args.buffer_size, seed=args.seed)

    train_dataset = train_dataset.with_format("torch")

    return train_dataset



def load_valid_data(args,parse_row_func):
    valid_dataset = load_dataset('csv',
                                 data_dir=args.valid_data_dir,
                                 split='train',
                                 streaming=True)
    valid_dataset = valid_dataset.map(
        lambda x: parse_row_func(
            x["seq_id"],
            x["seq_type"] if "seq_type" in x else "prot",
            x["seq"],
            x["vector_filename"] if "vector_filename" in x else None,
            x["matrix_filename"] if "matrix_filename" in x else None,
            x["label"] if "label" in x else None,
        ),
        batched=False
    )
    valid_dataset = split_dataset_by_node(valid_dataset, rank=args.local_rank, world_size=dist.get_world_size()) \
        .shuffle(buffer_size=args.buffer_size, seed=args.seed)

    valid_dataset = valid_dataset.with_format("torch")

    return valid_dataset


def load_test_data(args,parse_row_func):
    test_dataset = load_dataset('csv',
                                 data_dir=args.test_data_dir,
                                 split='train',
                                 streaming=True)
    test_dataset = test_dataset.map(
        lambda x: parse_row_func(
            x["seq_id"],
            x["seq_type"] if "seq_type" in x else "prot",
            x["seq"],
            x["vector_filename"] if "vector_filename" in x else None,
            x["matrix_filename"] if "matrix_filename" in x else None,
            x["label"] if "label" in x else None,
        ),
        batched=False
    )
    test_dataset = split_dataset_by_node(test_dataset, rank=args.local_rank, world_size=dist.get_world_size()) \
        .shuffle(buffer_size=args.buffer_size, seed=args.seed)

    test_dataset = test_dataset.with_format("torch")

    return test_dataset



config_path = {
    "num_labels":2,
    "input_type":"seq_matrix",
    "hidden_size":128,
    "num_attention_heads":12,
    "max_seq_len":512,
    "num_layers":12,
    "device":torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"),
    "hidden_dropout_prob":0.1,
    "feedforward_dropout_prob":0.1,
    "no_token_embeddings":False,
    "no_position_embeddings":False,
    "max_position_embeddings":512,
    "no_token_type_embeddings":False,
    "no_task_type_embeddings":False,
    "task_vocab_size":4,
    "seq_fc_size":[128*4],
    "activate_func":"sigmoid",
    "cnn_dropout":0.1,
    "embedding_input_size":128,
    "embedding_fc_size":[128*4],
    "seq_weight":-9999,
"embedding_weight":-9999,
    "truncation_seq_length":1000

}


args_path = {
    "llm_type":"esm",
    "input_type":"matrix",
    "trunc_type":"right",
    "seq_max_length":512,
    "buffer_size":10000,
    "train_data_dir":"",
    "valid_data_dir":"",
    "test_data_dir":"",
    "local_rank":-1,
    "seed":1234,
    "not_prepend_bos":True,
    "not_append_eos":True,
    "per_gpu_train_batch_size":128,
    "per_gpu_valid_batch_size":128,
    "worker_num":1,
    "do_train":True,
    "global_step_prefix":1000,
    "output_dir":"",
    "logging_steps":1000,
    "max_steps":10000,
    "weight_decay":0.1,
    "learning_rate":1e-4,
    "beta1":0.9,
    "beta2":0.98,



    


}



def main():
    ##参数处理
    args = SimpleNamespace(**args_path)

    args.device = create_device(args)
    config = SimpleNamespace(**config_path)
    ##加载模型
    seq_tokenizer,model,seq_subword = load_model(config,args)



    encoder_config = {
        "llm_type": args.llm_type,
        "input_type": args.input_type,
        "trunc_type": args.trunc_type,
        "seq_max_length": args.seq_max_length,
        "prepend_bos": True,
        "append_eos": True,
        "device":config.device
    }



    #数据解析,将数据传入esm模型，并返回原本所有数据以及esm模型的输出
    parse_row_func = Encoder(**encoder_config).encoder_single
    train_dataset = load_train_data(args,parse_row_func)
    valid_dataset = load_valid_data(args, parse_row_func)


    ##data处理
    batch_data_func = BatchConverter(label_size = args.label_size,

                 seq_subword = seq_subword,
                 seq_tokenizer = seq_tokenizer,
                 no_position_embeddings = config.no_position_embeddings,
                 np_token_type_embeddings = config.np_token_type_embeddings,
                 truncation_seq_length = config.truncation_seq_length,
                 truncation_matrix_length = config.truncation_matrix_length,
                 padding_idx = 0,
                 unk_idx = 1,
                 cls_idx = 2,
                 eos_idx = 3,
                 mask_idx = 4,
                 prepend_bos = not args.not_prepend_bos,
                 append_eos = not args.not_append_eos)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.per_gpu_train_batch_size,
        num_workers=args.worker_num,
        pin_memory=True,
        collate_fn=batch_data_func
    )

    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=args.per_gpu_valid_batch_size,
        num_workers=args.worker_num,
        pin_memory=True,
        collate_fn=batch_data_func
    )


    ##training
    if args.do_train:
        logger.info("+++++++Training+++++++++")
        #todo
        global_step, tr_loss, max_metric_model_info = train(args, train_dataloader, valid_dataloader, config, model, seq_tokenizer)
        logger.info("global_step = %s, average loss = %s", global_step, tr_loss)
        args.global_step_prefix = global_step



    if args.do_predict and args.global_step_prefix:
        test_dataset = load_test_data(args, parse_row_func)
        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=args.per_gpu_valid_batch_size,
            num_workers=args.worker_num,
            pin_memory=True,
            collate_fn=batch_data_func
        )
        logger.info("++++++++++++Testing+++++++++++++")
        # global_step = max_metric_model_info["global_step"]
        prefix = "checkpoint-{}".format(args.global_step_prefix)
        checkpoint = os.path.join(args.output_dir, prefix)
        model.to(args.device)
        result = test(args, model, test_dataloader)
        result = dict(("evaluation_" + k + "_{}".format(args.global_step_prefix), v) for k, v in result.items())


