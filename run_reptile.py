import os
import numpy as np
import torch
import higher

from transformers import BartTokenizer, BartConfig
from transformers import AdamW, get_linear_schedule_with_warmup

from dataloader.fewshot_gym_reptile import NLPFewshotGymReptileData

from bart import MyBart
from utils import freeze_embeds, trim_batch, get_tasks_list

from tqdm import tqdm

def run(args, logger):
    tokenizer = BartTokenizer.from_pretrained(args.model)

    train_tasks = get_tasks_list(args.custom_tasks_splits, "train")
    dev_tasks = get_tasks_list(args.custom_tasks_splits, "dev")
    logger.info("Training on the following tasks: {}".format(train_tasks))

    train_data = NLPFewshotGymReptileData(logger, args, args.train_dir, tasks=train_tasks, data_type="train", is_training=True)
    train_data.load_dataset(tokenizer)
    train_data.load_dataloader()

    if args.no_dev:
        dev_data = None
    else:
        dev_data = NLPFewshotGymReptileData(logger, args, args.train_dir, tasks=dev_tasks, data_type="dev", is_training=False)
        dev_data.load_dataset(tokenizer)
        dev_data.load_dataloader()

    if args.do_train:
        if args.checkpoint is not None:
            def convert_to_single_gpu(state_dict):
                def _convert(key):
                    if key.startswith('module.'):
                        return key[7:]
                    return key
                return {_convert(key):value for key, value in state_dict.items()}
            model = MyBart.from_pretrained(args.model,
                                           state_dict=convert_to_single_gpu(torch.load(args.checkpoint)))
        else:
            model = MyBart.from_pretrained(args.model)

        if args.freeze_embeds:
            logger.info("Freezing embeddings")
            freeze_embeds(model)

        if args.n_gpu>1:
            model = torch.nn.DataParallel(model)

        if torch.cuda.is_available():
            model.to(torch.device("cuda"))

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler =  get_linear_schedule_with_warmup(optimizer,
                                        num_warmup_steps=args.warmup_steps,
                                        num_training_steps=args.total_steps)
        train(args, logger, model, train_data, dev_data, optimizer, scheduler)

def train(args, logger, model, train_data, dev_data, optimizer, scheduler):
    torch.set_printoptions(profile="full")

    model.train()
    global_batch = 0
    global_step = 0
    train_losses = []
    dev_losses = []
    best_dev_loss = 1e10
    stop_training=False
    pad_token_id = train_data.tokenizer.pad_token_id

    logger.info("Starting training!")
    for epoch in range(int(args.num_train_epochs)):
        for task in tqdm(train_data.dataloader, desc="Train Epoch {}".format(epoch)):

            global_batch += 1

            task = task[:args.inner_step]  

            inner_opt = torch.optim.SGD(model.parameters(), lr=args.inner_lr)
            with higher.innerloop_ctx(
                model, 
                inner_opt,
                track_higher_grads=False,
                copy_initial_weights=False
            ) as (fnet, diffopt):

                for bidx, batch in enumerate(task):
                    if torch.cuda.is_available():
                        batch = [b.to(torch.device("cuda")) for b in batch]

                    batch[0], batch[1] = trim_batch(batch[0], pad_token_id, batch[1])
                    batch[2], batch[3] = trim_batch(batch[2], pad_token_id, batch[3])

                    train_loss = fnet(input_ids=batch[0], attention_mask=batch[1],
                                decoder_input_ids=batch[2], decoder_attention_mask=batch[3],
                                is_training=True)

                    train_losses.append(train_loss.detach().cpu())
                    diffopt.step(train_loss)

                for p, q in zip(model.parameters(), fnet.parameters()):
                    delta = p - q # !!! the optimizer is doing gradient _descent_, so you have to flip the direction here
                    if p.grad is None:
                        p.grad = delta.detach()
                    else:
                        p.grad += delta.detach()

            if global_batch % args.gradient_accumulation_steps == 0:
                global_step += 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

                if global_step % args.eval_period == 0:
                    logger.info("".format(epoch, np.mean(train_losses)))
                    logger.info("Epoch {} Train loss: {}".format(epoch, np.mean(train_losses)))
                    train_losses = []

                    if not args.no_dev:
                        curr_dev_loss = inference(args, model, dev_data, epoch)
                        logger.info("Epoch {} Dev loss: {}".format(epoch, curr_dev_loss))
                        
                        if curr_dev_loss < best_dev_loss:
                            model_state_dict = {k:v.cpu() for (k, v) in model.state_dict().items()}
                            torch.save(model_state_dict, os.path.join(args.output_dir, "best-model.pt"))
                            logger.info("Dev loss: {:.2f} --> {:.2f}. Saving the best checkpoint... ".format(best_dev_loss, curr_dev_loss))
                            best_dev_loss = curr_dev_loss

            if global_step >= args.total_steps:
                stop_training = True
                break

        if stop_training:
            break

    model_state_dict = {k:v.cpu() for (k, v) in model.state_dict().items()}
    torch.save(model_state_dict, os.path.join(args.output_dir, "last-model.pt"))

def inference(args, model, dev_data, epoch):
    # dev at the end of epoch
    dev_losses = []
    pad_token_id = dev_data.tokenizer.pad_token_id
    for task in tqdm(dev_data.dataloader, desc="Dev Epoch {}".format(epoch)):
        task = task[:args.inner_step]
        inner_opt = torch.optim.SGD(model.parameters(), lr=args.inner_lr)
        with higher.innerloop_ctx(
            model, 
            inner_opt,
            track_higher_grads=False,
            copy_initial_weights=False
        ) as (fnet, diffopt):
            for bidx, batch in enumerate(task):
                if torch.cuda.is_available():
                    batch = [b.to(torch.device("cuda")) for b in batch]
                batch[0], batch[1] = trim_batch(batch[0], pad_token_id, batch[1])
                batch[2], batch[3] = trim_batch(batch[2], pad_token_id, batch[3])
                    
                train_loss = fnet(input_ids=batch[0], attention_mask=batch[1],
                            decoder_input_ids=batch[2], decoder_attention_mask=batch[3],
                            is_training=True)
                diffopt.step(train_loss)
                
            # use the loss in the last optimization step as dev metric
            dev_losses.append(train_loss.detach().cpu())
    
    return np.mean(dev_losses)
