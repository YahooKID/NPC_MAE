import argparse
import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

from npc_cls import npc_cls
from dataset import NPCCls_Dataset

import copy
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '5678'

def get_args():
    opt = argparse.ArgumentParser()
    opt.add_argument('--epochs', help="Num of epochs", default=100, type=int)
    opt.add_argument('--bs', help="Batch Size", default=64, type=int)
    opt.add_argument('--ckpt_dir', help="where to save checkpoints", default="checkpoints/clsckpt2")
    opt.add_argument('--local_rank', default=0, type=int)
    return opt.parse_args()
    
    
def train(model, critetion, optimizer, train_loader, epoch, device):
    with tqdm(total=len(train_loader)) as pb:
        for _, sample_batched in enumerate(train_loader):
            featmap, input_ids, attention_mask, gt = sample_batched
            featmap = featmap.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            gt = gt.to(device)
            
            logits = model(featmap, input_ids, attention_mask)
            conf = gt[:, 0:1]
            sigt = gt[:, 1:]
            # print(logits.shape, conf.shape, sigt.shape)
            total_loss = critetion(conf * logits, sigt)
            # print(total_loss)
            
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            str_loss = f"{total_loss.cpu().data.numpy():.4f}"
            pb.update(1)
            pb.set_postfix(epoch=epoch, loss=str_loss)
            
def eval(model, test_loader, device):
    preds = []
    acts = []
    
    for _, sample_batched in  enumerate(test_loader):
        featmap, input_ids, attention_mask, gt = sample_batched
        featmap = featmap.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        gt = gt.to(device)
        
        logits = model(featmap, input_ids, attention_mask)
        logits = logits.cpu().data.numpy()
        gt = gt.cpu().data.numpy()
        preds.extend(logits)
        acts.extend(gt)
    
    preds, acts = np.asarray(preds), np.asarray(acts)
    p = accuracy1(torch.from_numpy(preds), torch.from_numpy(acts))
    return p
            
def accuracy(pred, act):
    p = pred[:, 0] * pred[:, 1]
    a = act[:, 0] * act[:, 1]
    p0 = copy.deepcopy(p)
    p0[p0>0.3] = 1
    p0[p0<=0.3] = 0
    q = torch.sum(p0)
    p0 = torch.abs(p0 - a)
    p0 = float(torch.sum(p0) / a.shape[0])
    p = torch.abs(p - a)
    p = float(torch.sum(p) / a.shape[0])
    print("Average bias: {:.3f} | Accuracy: {:.3f} | total: {} | positive: {} | cal: {}".format(p, p0, a.shape[0], torch.sum(a), q))
    return p

def accuracy1(pred, act):
    p = pred[:, 0]
    a = act[:, 1]
    p0 = copy.deepcopy(p)
    p0[p0>0.3] = 1
    p0[p0<=0.3] = 0
    q = torch.sum(p0)
    p0 = torch.abs(p0 - a)
    p0 = float(torch.sum(p0) / a.shape[0])
    p = torch.abs(p - a)
    p = float(torch.sum(p) / a.shape[0])
    print("Average bias: {:.3f} | Accuracy: {:.3f} | total: {} | positive: {} | cal: {}".format(p, p0, a.shape[0], torch.sum(a), q))
    return p

def auc(pred, act):
    p = pred[:, 0]
    a = act[:, 1]
    c = act[:, 0]
    p = p[c!=0]
    a = a[c!=0]
    pos = p[a==1]
    neg = p[a==0]
    q = pos.shape
    
    
def save_checkpoint(state, ckpt_dir, is_best):
    filename = '%s/ckpt.pth' % (ckpt_dir)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth', 'best.pth'))

def is_parallel(model):
    '''Return True if model's type is DP or DDP, else False.'''
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
    
def main(args):
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', rank = 0, world_size = 1)
    device = torch.device("cuda", local_rank)
    if dist.get_rank() == 0:
        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)
            
    print("Building Engine...")
    model = npc_cls(32*768, device=device)
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)    
    print("Build Done!")

    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCELoss()
    criterion = FocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=1.5e-7, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.03)

    train_dataset = NPCCls_Dataset()
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.bs,
                              num_workers=4,
                              sampler=train_sampler)
    
    test_dataset = NPCCls_Dataset(mission="eval")
    test_loader = DataLoader(test_dataset,
                             batch_size=args.bs,
                             num_workers=4,
                             shuffle=False)

    
    best_prec1 = 0.
    for epoch in range(args.epochs):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        train(model, criterion, optimizer, train_loader, epoch, device)
        
        model.eval()
        p = eval(model, test_loader, device)
        if dist.get_rank() == 0:
            print('Epoch: {}, Accuracy: {:.3f}'.format(epoch, 1 - p))

            is_best = (1 - p) > best_prec1
            best_prec1 = max((1 - p), best_prec1)
            state_dict = model.module.state_dict() if is_parallel(model) else model.state_dict()
            save_checkpoint({
                    'epoch': epoch,
                    'state_dict': state_dict,
                    'optimizer': optimizer.state_dict(),
                    'best_prec1': best_prec1,
                }, args.ckpt_dir, is_best)
        # end of one epoch
    if dist.get_rank() == 0:
        print(f"Finish Training.")
        print(f"Best Prec: {best_prec1:.3f}")
        print(f"Best ckpt save in: {args.ckpt_dir+'/ckpt.best.pth'}")

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()
        
if __name__ == '__main__':
    args = get_args()
    print(args)
    main(args)
