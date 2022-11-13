# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for l2p implementation
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------
"""
Train and eval functions used in main.py
"""
import math
import sys
import os
import datetime
import json
from typing import Iterable
from pathlib import Path
# from black import out

import torch
import torch.nn.functional as F
import numpy as np

from timm.utils import accuracy
from timm.optim import create_optimizer

import utils
import matplotlib.pyplot as plt


def l2_normalize(x, dim=None, epsilon=1e-12):
    """Normalizes a given vector or matrix."""
    square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
    x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon)))
    return x * x_inv_norm

# todo Key Contrastive로 수정!


def key_loss_fn(out):
    #! 집에가서 코드 마무리하기! 목표는 기존 81.59 -> over 83.0
    # todo 1. token값 필요
    # todo 2. Prompt당 token 수 필요 -> Label 생성, matrix 압축
    logit = out['key_logits']
    targets = out['key_targets']

    loss = 0.2*F.cross_entropy(logit, targets)

    return loss
def prompt_loss_fn(out):
    cur_prompt = out['task_prompts'][-1]
    prev_prompts = out['task_prompts'][:-1]
    num=len(prev_prompts)
    cur_mean = cur_prompt.mean(dim=1).mean(dim=0).unsqueeze(0)
    loss =0.
    for prev_p in prev_prompts:
        prev_mean = prev_p.mean(dim=1).mean(dim=0).unsqueeze(0)
        loss = loss+ torch.cosine_similarity(cur_mean,prev_mean)
    loss = loss/num
    
    return loss


def train_one_epoch(model: torch.nn.Module, original_model: torch.nn.Module,
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None, args=None,):

    model.train(set_training_mode)
    original_model.eval()

    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Lr', utils.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('Loss', utils.SmoothedValue(
        window_size=1, fmt='{value:.4f}'))
    header = f'Train: Epoch[{epoch+1:{int(math.log10(args.epochs))+1}}/{args.epochs}]'

    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if original_model is not None:
            output = original_model(input)
            cls_features = output['pre_logits']
            # cls_features = output['img_query']
        else:
            cls_features = None
        if task_id==0: #first task
            threshold=(epoch)*0.05
        else:
            threshold=0.2  
        output = model(input, task_id=task_id, cls_features=cls_features, test=-1, threshold=threshold,epoch=epoch)
        logits = output['logits']
        # key_logits =output['T_prompt_logits']

        # here is the trick to mask out classes of non-current tasks
        if args.train_mask and class_mask is not None:
            mask = class_mask[task_id]
            not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
            logits = logits.index_fill(
                dim=1, index=not_mask, value=float('-inf'))

            # key_logits = key_logits.index_fill(
            # dim=1, index=not_mask, value=float('-inf'))

        # base criterion (CrossEntropyLoss)
        main_loss = criterion(logits, target)
        key_loss = key_loss_fn(output)
        loss = main_loss + key_loss- args.pull_constraint_coeff * output['reduce_sim']

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        torch.cuda.synchronize()
        metric_logger.update(Loss=loss.item())
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
        metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    print("train prompt selection:\n", output['train_prompt_selection'])
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, original_model: torch.nn.Module, data_loader,
             device, task_id=-1, class_mask=None, args=None,
             test=-1):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test: [Task {}]'.format(task_id + 1)

    # switch to evaluation mode
    model.eval()
    original_model.eval()

    # key_correct = 0.
    # total_size = 0.
    with torch.no_grad():
        for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output

            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
            else:
                cls_features = None
            threshold=0.2
            output = model(input, task_id=task_id,
                           cls_features=cls_features, test=test, threshold=threshold)
            logits = output['logits']

            if args.task_inc and class_mask is not None:
                # adding mask to output logits
                mask = class_mask[task_id]
                logits_mask = torch.ones_like(
                    logits, device=device) * float('-inf')
                logits_mask = logits_mask.index_fill(1, mask, 0.0)
                logits = logits + logits_mask

            loss = criterion(logits, target)

            acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            metric_logger.meters['Loss'].update(loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

            # correct = output['prompt_idx'] == task_id
            # key_correct += correct.sum().item()
            # total_size += target.size(0)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'], losses=metric_logger.meters['Loss']))
    # print('Task{} Key Predict ACC: {:.2f}%'.format(
    #     task_id, key_correct/total_size*100))
    print("test prompt selection:\n", output['test_prompt_selection'])
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_till_now(model: torch.nn.Module, original_model: torch.nn.Module, data_loader,
                      device, task_id=-1, class_mask=None, acc_matrix=None, args=None,):
    stat_matrix = np.zeros((3, args.num_tasks))  # 3 for Acc@1, Acc@5, Loss

    for i in range(task_id+1):
        test_stats = evaluate(model=model, original_model=original_model, data_loader=data_loader[i]['val'],
                              device=device, task_id=i, class_mask=class_mask, args=args,
                              test=task_id)

        stat_matrix[0, i] = test_stats['Acc@1']
        stat_matrix[1, i] = test_stats['Acc@5']
        stat_matrix[2, i] = test_stats['Loss']

        acc_matrix[i, task_id] = test_stats['Acc@1']

    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id+1)

    diagonal = np.diag(acc_matrix)

    result_str = "[Average accuracy till task{}]\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}".format(
        task_id+1, avg_stat[0], avg_stat[1], avg_stat[2])
    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) -
                              acc_matrix[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(
            forgetting, backward)
    print(result_str)

    return test_stats


def train_and_evaluate(model: torch.nn.Module, model_without_ddp: torch.nn.Module, original_model: torch.nn.Module,
                       criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer, lr_scheduler, device: torch.device,
                       class_mask=None, args=None,):

    # create matrix to save end-of-task accuracies
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))

    for task_id in range(args.num_tasks):
        # Create new optimizer for each task to clear optimizer status
        model.module.before_train(task_id)
        optimizer = create_optimizer(args, model)
        # if task_id > 0 and args.reinit_optimizer:
            # if task_id > 0:
            # optimizer = create_optimizer(args, model)

        for epoch in range(args.epochs):
            train_stats = train_one_epoch(model=model, original_model=original_model, criterion=criterion,
                                          data_loader=data_loader[task_id]['train'], optimizer=optimizer,
                                          device=device, epoch=epoch, max_norm=args.clip_grad,
                                          set_training_mode=True, task_id=task_id, class_mask=class_mask, args=args,)

            if lr_scheduler:
                lr_scheduler.step(epoch)
        # model.module.after_train()

        # model.module.after_train(task_id)

        test_stats = evaluate_till_now(model=model, original_model=original_model, data_loader=data_loader, device=device,
                                       task_id=task_id, class_mask=class_mask, acc_matrix=acc_matrix, args=args)
        if args.output_dir and utils.is_main_process():
            Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(
                parents=True, exist_ok=True)

            checkpoint_path = os.path.join(
                args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id+1))
            state_dict = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args,
            }
            if args.sched is not None and args.sched != 'constant':
                state_dict['lr_scheduler'] = lr_scheduler.state_dict()

            utils.save_on_master(state_dict, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch, }

        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, '{}_stats.txt'.format(datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M'))), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('='*50)
    print('Number of params:', n_parameters)
