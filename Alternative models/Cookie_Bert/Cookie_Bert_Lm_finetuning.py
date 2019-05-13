from pathlib import Path
import pandas as pd
from examples.run_lm_finetuning import *

#Reading dataset
path = Path('Cookie_Bert')
df_train = pd.read_csv(path/'train_cookie_simb_off.csv', header=0)
df_test = pd.read_csv(path/'test_cookie_simb_off.csv', header=0)
print(df_test.head())

frames=[df_train, df_test]
df_for_lm = pd.concat(frames)
df_for_lm = df_for_lm['text'].str.cat(sep = '\n')

#I write the corpus for the finetuning in a txt file to use the already designed Bert functions
with open("Cookie_Text_for_finetuning.txt", "w") as text_file:
     text_file.write(df_for_lm)

# Hyperparameter and config values
output_dir = path / 'Models'
bert_model = 'bert-base-uncased'
do_lower_case = True
train_file = "Cookie_Text_for_finetuning.txt"
gradient_accumulation_steps = 1
max_seq_length = 128
on_memory = True
train_batch_size = 32
num_train_epochs = 3
learning_rate = 3e-5
local_rank = -1
fp16 = True
loss_scale = 0
warmup_proportion = 0.1
do_train = True
n_gpu = torch.cuda.device_count()


if os.path.exists(output_dir) and os.listdir(output_dir):
    raise ValueError("Output directory ({}) already exists and is not empty.".format(output_dir))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)

# train_examples = None
num_train_optimization_steps = None

train_dataset = BERTDataset(train_file, tokenizer, seq_len=max_seq_length, corpus_lines=None, on_memory=on_memory)
num_train_optimization_steps = int(
    len(train_dataset) / train_batch_size / gradient_accumulation_steps) * num_train_epochs
if local_rank != -1:
    num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

# Prepare model
model = BertForPreTraining.from_pretrained(bert_model)
if fp16:
    model.half()
model.to("cuda")
if local_rank != -1:
    try:
        from apex.parallel import DistributedDataParallel as DDP
    except ImportError:
        raise ImportError(
            "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
    model = DDP(model)
elif n_gpu > 1:
    model = torch.nn.DataParallel(model)

# Prepare optimizer
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

if fp16:
    try:
        from apex.optimizers import FP16_Optimizer
        from apex.optimizers import FusedAdam
    except ImportError:
        raise ImportError(
            "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

    optimizer = FusedAdam(optimizer_grouped_parameters,
                          lr=learning_rate,
                          bias_correction=False,
                          max_grad_norm=1.0)
    if loss_scale == 0:
        optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
    else:
        optimizer = FP16_Optimizer(optimizer, static_loss_scale=loss_scale)

else:
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=learning_rate,
                         warmup=warmup_proportion,
                         t_total=num_train_optimization_steps)

global_step = 0
if do_train:
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Batch size = %d", train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)

    if local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
    else:
        # TODO: check if this works with current data generator from disk that relies on next(file)
        # (it doesn't return item back by index)
        train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

    model.train()
    for _ in trange(int(num_train_epochs), desc="Epoch"):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to('cuda') for t in batch)
            input_ids, input_mask, segment_ids, lm_label_ids, is_next = batch
            loss = model(input_ids, segment_ids, input_mask, lm_label_ids, is_next)
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            if fp16:
                optimizer.backward(loss)
            else:
                loss.backward()
            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % gradient_accumulation_steps == 0:
                if fp16:
                    # modify learning rate with special warm up BERT uses
                    # if args.fp16 is False, BertAdam is used that handles this automatically
                    lr_this_step = learning_rate * warmup_linear(global_step / num_train_optimization_steps,
                                                                 warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

    # Save a trained model
    logger.info("** ** * Saving fine - tuned model ** ** * ")
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
    if do_train:
        torch.save(model_to_save.state_dict(), output_model_file)