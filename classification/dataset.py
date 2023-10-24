from torch.utils.data import Dataset
import os
import numpy as np
import torch
from transformers import BertTokenizer
from tqdm import tqdm
import logging 
import copy

logger = logging.getLogger(__name__)

class NPCCls_Dataset(Dataset):
    def __init__(self, 
                 confpath="conf", 
                 gtpath="gt", 
                 vicunapath="vicuna_res", 
                 featpath="clsset",
                 mission="train",
                 max_txt_len=128
                 ):
        super().__init__()
        self.tokenizer = self.init_tokenizer()
        self.confpath = confpath
        self.gtpath = gtpath
        self.vicunapath = vicunapath
        self.featpath = featpath
        self.max_txt_len = max_txt_len
        vid = os.listdir(self.vicunapath)
        vvid = set([i[:-4] for i in vid])
        tid = os.listdir(confpath)
        ttid = set([i.replace(i.split("_")[-1],"")[:-1] for i in tid])
        self.id = sorted(list(set.intersection(vvid, ttid)))
        self.featlist = []
        self.confs = []
        self.gts = []
        self.rids = []
        self.inputiddict = {}
        self.attention_maskdict={}
        if mission=="train":
            self.id = self.id[:int(len(self.id) / 10 * 9)]
        else:
            self.id = self.id[int(len(self.id) / 10 * 9):]
        for i in tid:
            ii = i.split("_")[0] + "_" + i.split("_")[1]
            if ii in self.id:
                self.featlist.append(i[:-4])
                
        logger.info("Dataset: Preapre the conf and ground-truth of {} samples.".format(len(self.featlist)))
        for featname in tqdm(self.featlist):
            featid = featname.split("_")[0] + "_" + featname.split("_")[1]
            self.rids.append(featid)
            conffile = os.path.join(self.confpath, "{}.txt".format(featname))
            with open(conffile, "r") as f:
                self.confs.append(int(f.read()))
            gtfile = os.path.join(self.gtpath, "{}.txt".format(featid))
            with open(gtfile, "r") as f:
                self.gts.append(int(f.read()))
        
        self.rids = list(set(self.rids))
        
        logger.info("Dataset: Prepare the input_ids and attentionmasks.")
        for featid in tqdm(self.rids):
            vicunafile = os.path.join(self.vicunapath, "{}.txt".format(featid))
            with open(vicunafile, "r") as f:
                vicunatxt = f.read()
                input_ids, attention_mask = self.getInputidsAndAttentionmask(vicunatxt)
            self.inputiddict[featid] = input_ids[0]
            self.attention_maskdict[featid] = attention_mask[0]

        
                
        
                
    def __len__(self):
        return len(self.featlist)
    
    def __getitem__(self, index):
        featname = self.featlist[index]
        featid = featname.split("_")[0] + "_" + featname.split("_")[1]
        featfile = os.path.join(self.featpath, "{}.npy".format(featname))
        featmap = np.load(featfile)
        featmap = torch.from_numpy(featmap)
        input_ids = copy.deepcopy(self.inputiddict[featid])
        attention_mask = copy.deepcopy(self.attention_maskdict[featid])
        gt = torch.Tensor([self.confs[index], self.confs[index]*self.gts[index]])
        return featmap, input_ids, attention_mask, gt
    
    def getInputidsAndAttentionmask(self, txt):
        text_Qformer = self.tokenizer(
                txt,
                padding='max_length',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            )
        return text_Qformer.input_ids, text_Qformer.attention_mask
        
        
        
    @staticmethod
    def init_tokenizer(truncation_side="left"):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side=truncation_side)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer
    
if __name__ == "__main__":
    n = NPCCls_Dataset()
    _=0
        