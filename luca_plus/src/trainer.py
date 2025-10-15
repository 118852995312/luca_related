
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import time
import torch.distributed as dist
from evaluator import evaluate
import os
def reduce_tensor(tensor,world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= world_size
    return rt


def get_lr(optimizer):
    '''
    get learning rate
    :param optimizer:
    :return:
    '''
    for p in optimizer.param_groups:
        if "lr" in p:
            return p["lr"]



def train(args,train_dataloader,dev_dataloader,model_config,model,seq_tokenizer):
    train_batch_total_num = len(train_dataloader)

    if args.logging_steps <= 0:
            args.logging_steps = (train_batch_total_num + args.gradient_accumulation_steps - 1) // args.gradient_accumulation_steps
        # if args.save_steps <= 0:
        #     args.save_steps = args.logging_steps

    t_total = args.num_train_epochs * (train_batch_total_num + args.gradient_accumulation_steps - 1) // args.gradient_accumulation_steps

    if args.max_steps < t_total:
        args.max_steps = t_total

    no_decay = ["bias","rmsnorm.weight","rms_norm.weight","rms.norm.weight"]
    print("-"*50)

    optimizer_grouped_parameters = [
        {
            "params":[p for n,p in model.named_parameters() if not any(nd in n.lower() for nd in no_decay)],
            "weight_decay":args.weight_decay
        },
        {
            "params":[p for n,p in model.named_parameters() if any(nd in n.lower() for nd in no_decay)],
            "weight_decay":0.0
        }
    ]

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr = args.learning_rate,
                      betas = [args.beta1 if args.beta1 > 0 else 0.9, args.beta2 if args.beta2 > 0 else 0.98],
                      eps = args.adam_epsilon)


    if args.warmup_steps > 0:
        scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps = args.warmup_steps,
                                                    num_training_steps = args.max_steps)

    else:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                           gamma=args.lr_decay_rate if args.lr_decay_rate > 0 else 0.9)


    if args.n_gpu > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids = [args.local_rank],
            output_device = args.local_rank,
            find_unused_parameters= True
        )
    optimizer.zero_grad()
    total_use_time = 0
    cur_epoch_total_loss = 0.0
    real_epoch = 0
    total_loss,logging_loss = 0.0,0.0
    run_begin_time = time.time()
    best_metric_model_info = {}
    global_step = 0
    best_metric_type = args.best_metric_type
    best_metric_flag = True
    if "loss" in best_metric_type:  # argmin
        best_metric_value = 10000000.0
        best_metric_flag = False
    else:  # argmax
        best_metric_value = 0.0



    for epoch in range(args.num_train_epochs):
        model.train()
        cur_epoch_step = 0
        cur_epoch_loss = 0.0
        cur_epoch_time = 0.0
        no_grad_gradient_accumulation_step = False
        for step,batch in enumerate(train_dataloader):
            begin_time = time.time()

            loss, logits, output = model(**batch)
            if args.gradient_accumulation_steps > 1:
                loss /= args.gradient_accumulation_steps

            loss.backward()
            no_grad_gradient_accumulation_step = False

            if args.n_gpu > 1:
                reduced_loss = reduce_tensor(loss.data,dist.get_world_size())
            else:
                reduced_loss = loss

            cur_loss = reduced_loss.item()
            end_time = time.time()
            cur_use_time = end_time - begin_time
            total_use_time += cur_use_time
            total_loss += cur_loss
            logging_loss += cur_loss
            cur_epoch_total_loss += cur_loss
            cur_epoch_loss += cur_loss
            cur_epoch_time += cur_use_time
            global_step += 1
            cur_epoch_step += 1
            if global_step % args.gradient_accumulation_steps == 0:
                # print("\rTraining, Epoch: %04d, Batch: %06d, C-L: %.08f, C-A-L: %.08f, G-A-L: %.08f" % (
                # epoch + 1,
                # cur_epoch_step,
                # cur_loss,
                # cur_epoch_total_loss / cur_epoch_step,
                # total_loss / global_step), end="", flush=True)
                if global_step % args.logging_steps == 0:
                    print(
                        "Training, Epoch: %04d, Batch: %06d, Cur Loss: %.08f, Cur Avg Loss: %.08f,  Global Avg Loss: %.08f, Time: %0.4f\n"
                        % (
                            epoch + 1,
                            cur_epoch_step,
                            cur_loss,
                            cur_epoch_total_loss / cur_epoch_step,
                            total_loss / global_step,
                            cur_use_time)
                        )

            if (step + 1) % args.gradient_accumulation_steps == 0:
                ##到gradient_accumulation_steps就更新参数
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                if scheduler and args.lr_update_strategy == "step":
                     # Update learning rate schedule
                    scheduler.step()
                    if global_step % args.logging_steps == 0:
                        updated_lr = get_lr(optimizer)
                        print("\ncur steps: %d,  lr: %f" % (global_step, updated_lr))
                        print("Steps: %d, Updated lr: %f\n" % (global_step, updated_lr))

                optimizer.zero_grad()
                no_grad_gradient_accumulation_step = True


        if not no_grad_gradient_accumulation_step:
            ##如果epoch结束没有到gradient_accumulation_steps，仍强制更新参数
            optimizer.step()
            optimizer.zero_grad()
            print("Has retained gard: rank=%d" % args.local_rank)

        scheduler.step()
        if args.local_rank in [-1, 0]:
            updated_lr = scheduler.get_last_lr()[0]

        acc_val = evaluate(args,dev_dataloader, model)

        print(" valid data acc:%.5f，global_step：%d,epoch:%d " % (acc_val, global_step, epoch))
        if acc_val > best_metric_value:
            best_metric_value = acc_val
            output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
            save_check_point(args, model, model_config, seq_tokenizer, output_dir)

    if args.n_gpu > 1:
        cleanup()
    return global_step, total_loss / global_step, best_metric_model_info



def cleanup():
    dist.destroy_process_group()



def save_check_point(args, model, model_config, seq_tokenizer, output_dir):
    '''
    save checkpoint
    :param args:
    :param model:
    :param seq_tokenizer
    :param model_config
    :param output_dir:
    :return:
    '''

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, "module") else model
    try:
        model_to_save.save_pretrained(output_dir)
    except Exception as e:
        model_config.save_pretrained(output_dir)
        torch.save(model_to_save, os.path.join(output_dir, "pytorch.pt"))
        torch.save(model_to_save.state_dict(), os.path.join(output_dir, "pytorch.pth"))
    # torch.save(model_to_save, os.path.join(output_dir + "model.pth"))
    if seq_tokenizer:
        tokenizer_dir = os.path.join(output_dir, "tokenizer")
        if not os.path.exists(tokenizer_dir):
            os.makedirs(tokenizer_dir)
        seq_tokenizer.save_pretrained(tokenizer_dir)

    torch.save(args, os.path.join(output_dir, "training_args.bin"))
    print("Saving model checkpoint to %s" % output_dir)