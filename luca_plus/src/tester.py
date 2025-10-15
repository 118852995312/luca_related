
import torch
from utils import process_outputs
from metrics import metrics_binary

def test(args,dev_dataloader,model):
    '''
    evaluation
    :param args:
    :param model:
    :param parse_row_func:
    :param batch_data_func:
    :param prefix:
    :param log_fp:
    :return:
    '''

    # save_output_dir = os.path.join(args.output_dir, prefix)
    # print("\nEvaluating information dir: ", save_output_dir)
    # if args.local_rank in [-1, 0] and not os.path.exists(save_output_dir):
    #     os.makedirs(save_output_dir)


    nb_steps = 0
    # truth
    truths = None
    # predicted prob
    preds = None
    eval_loss = 0
    model.eval()
    for step, batch in enumerate(dev_dataloader):
        # eval
        with torch.no_grad():
            output = model(**batch)
            cur_loss, cur_logits, cur_output = output[:3]
            cur_loss = cur_loss.item()
            eval_loss += cur_loss
            nb_steps += 1
            print("\rEval,  Sample Num: %d, Cur Loss: %0.6f, Avg Loss: %0.6f" % (step + 1, cur_loss, eval_loss / nb_steps),
                  end="", flush=True)

            if args.do_metrics and "labels" in batch and batch["labels"] is not None:
                truths, preds = process_outputs(

                    batch["labels"],
                    cur_output,
                    truths,
                    preds,
                    ignore_index=args.ignore_index,
                    keep_seq=False
                )
    avg_loss = eval_loss / nb_steps
    all_result = {
        "avg_loss": round(float(avg_loss), 6),
        "total_loss": round(float(eval_loss), 6)
    }
    if args.do_metrics and truths is not None and len(truths) > 0:
        dev_metrics = metrics_binary(truths,preds)
        all_result.update(
            dev_metrics
        )

    # with open(os.path.join(save_output_dir, "dev_metrics.txt"), "w") as writer:
    #     writer.write("***** Dev results {} *****\n".format(prefix))
    #     writer.write("Test average loss = %0.6f\n" % avg_loss)
    #     writer.write("Test total loss = %0.6f\n" % eval_loss)
    #     for key in sorted(all_result.keys()):
    #         writer.write("%s = %s\n" % (key, str(all_result[key])))
    return all_result