from .new_bert_cnn import NewBertCNN
import torch
from transformers import get_scheduler,BertTokenizer,AutoTokenizer
import codecs
from subword_nmt.apply_bpe import BPE



def load_model(config,args):
    # tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)
    tokenizer = BertTokenizer(args.vocab_path,do_lower_case=args.do_lower_case)
    config.vocab_size = tokenizer.vocab_size
    config.pad_token_id = tokenizer.pad_token_id
    model = NewBertCNN(config,args)
    bpe_codes = codecs.open(args.codes_file)
    seq_subword = BPE(bpe_codes, merges=-1, separator='')
    model = model.to(args.device)
    if len(args.model_path) > 0:
        model.load_state_dict(torch.load(args.model_path,map_location='auto'))

    return tokenizer,model,seq_subword






