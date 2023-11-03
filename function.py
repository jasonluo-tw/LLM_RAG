import torch
import torch.nn as nn

from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights
from torchvision.models import resnet50, ResNet50_Weights

#from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import (
    BertTokenizerFast,
    AutoModel
)
import numpy as np

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import logging
logging.set_verbosity_error()

class LM():
    def __init__(self):
        self.model = AutoModel.from_pretrained('ckiplab/bert-base-chinese')

        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
        self.device = 'cuda'

    def get_embs(self, texts):
        encode = self.tokenizer(texts, add_special_tokens=True, truncation=True,
                                padding="max_length", max_length=20, 
                                return_attention_mask=True, return_tensors="pt")

        self.model.eval()
        self.model = self.model.to(self.device)

        with torch.no_grad():
            ## The first output is last hidden state
            ## The second output is the output of first token([CLS]) after fully-connected and tanh
            outputs = self.model(
                input_ids=encode['input_ids'].to(self.device),
                token_type_ids=encode['token_type_ids'].to(self.device),
                attention_mask=encode['attention_mask'].to(self.device)
            )
            #print(model.pooler(outputs.last_hidden_state)[0, :5])
            #print(outputs.pooler_output[0, :5])

        ## get the last hidden output
        embs = outputs[0].to('cpu')

        #print(encode['attention_mask'].shape)
        #print(embs.size())
        input_mask_expanded = encode['attention_mask'].unsqueeze(-1).expand(embs.size()).float()
        #print(input_mask_expanded.shape)

        out_embs = torch.sum(embs * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return out_embs.tolist()

class ImageExtractor():
    def __init__(self, kind='resnet50', device='cpu'):
        if kind == 'resnet50':
            weights = ResNet50_Weights.DEFAULT
            preprocess = weights.transforms(antialias=True)
            model = resnet50(weights=weights)
        elif kind == 'efficientnet_b7':
            weights = EfficientNet_B7_Weights.DEFAULT
            preprocess = weights.transforms(antialias=True)
            model = efficientnet_b7(weights=weights)

        modules = list(model.children())[:-1]
        model = nn.Sequential(*modules)
        for p in model.parameters():
            p.requires_grad = False

        model.eval()

        self.device = device
        self.model = model
        self.preprocess = preprocess

        self.model = self.model.to(self.device)

    def get_emb(self, imgs):
        img_embs = []
        for ii, img in enumerate(imgs):
            input_img = self.preprocess(img).unsqueeze(0).to(self.device)
            img_emb = self.model(input_img).cpu().numpy()

            img_embs.append(np.squeeze(img_emb))
            print('Process:', (ii+1), '/', len(imgs))

        img_embs = np.stack(img_embs, axis=0)

        return img_embs

    def get_batch_embs(self, imgs):
        batch = torch.stack([self.preprocess(img) for img in imgs], dim=0)
        img_embs = torch.squeeze(self.model(batch.to(self.device))).cpu().numpy()

        return img_embs

