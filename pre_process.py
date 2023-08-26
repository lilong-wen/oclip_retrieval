import torch
from PIL import Image
import numpy as np
import os
import math
import csv
import pandas as pd
import pickle
from pathlib import Path

class ExtractFeature:

    def __init__(self,file_path, save_dir, batch_size, preprocess_fn, model, device):

        self.file_path = file_path
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.preprocess_fn = preprocess_fn
        self.model = model
        
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        img_key = 'filepath'
        text_key = 'title'
        df = pd.read_csv(file_path, sep='\t', quoting=csv.QUOTE_NONE)
        self.images = df[img_key].tolist()
        self.texts = df[text_key].tolist()

        self.batches = math.ceil(len(self.images) / batch_size)
        # self.save_dir = os.path.dirname(file_path)
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
            
    def query_texts(self):
        return self.texts

    def compute_oclip_feature(self, batch_file):
        
        photos = [Image.open(photo_file) for photo_file in batch_file]
        photo_preprocessed = torch.vstack([self.preprocess_fn(photo)[0] for photo in photos]).to(self.device)
        # mask_preprocessed = torch.stack([self.preprocess_fn(photo)[1] for photo in photos]).to(self.device)

        with torch.no_grad():
            photo_feature = self.model.encode_image(photo_preprocessed)[0]
            photo_feature /= photo_feature.norm(dim=-1, keepdim=True)
            photo_feature = photo_feature.permute(1,0,2)
        return photo_feature.cpu().numpy()

    def merge_batches(self):

        features_list = [np.load(features_file) for features_file in sorted(Path(self.save_dir).glob("*.npy"))]
        features = np.concatenate(features_list)
        np.save(os.path.join(self.save_dir, "features.npy"), features)

        ids = pd.concat([pd.read_csv(ids_file) for ids_file in sorted(Path(self.save_dir).glob("*.csv"))])
        ids.to_csv(os.path.join(self.save_dir, "ids.csv"), index=False)
        
        print(f"merge features complete")

    def extract_features(self):
        
        for i in range(self.batches):
            print(f"processing batch {i+1}/{self.batches}")

            batch_id_path = os.path.join(self.save_dir,f"{i:010d}.csv")
            batch_feature_path = os.path.join(self.save_dir,f"{i:010d}.npy")

            if not os.path.isfile(batch_feature_path):
                try:
                    batch_file = self.images[i*self.batch_size: (i+1)*self.batch_size]
                    batch_feature = self.compute_oclip_feature(batch_file)
                    np.save(batch_feature_path, batch_feature)

                    photo_ids = [name.split('/')[-1] for name in batch_file]
                    photo_ids_data = pd.DataFrame(photo_ids, columns=['photo_id'])
                    photo_ids_data.to_csv(batch_id_path, index=False)

                except:
                    print(f"problem with batch: {i}")
            else:
                print(f"feature extraction for batch: {i} already done")
        
        if not os.path.isfile(os.path.join(self.save_dir, "features.npy")):
            self.merge_batches()
        else:
            print(f"merge complete")
                
class Tokenizer():
    def __init__(self, char_dict_pth):
        with open(char_dict_pth, 'rb') as f:
            self.letters = pickle.load(f)
            self.letters = [chr(x) for x in self.letters]

        self.p2idx = {p: idx+1 for idx, p in enumerate(self.letters)}
        self.idx2p = {idx+1: p for idx, p in enumerate(self.letters)}

        self.idx_mask = len(self.letters) + 1
        self.EOS = len(self.letters) + 2
        self.word_len = 25

    def tokenize(self, text):
        token = torch.zeros(self.word_len)
        for i in range(min(len(text), self.word_len)):
            if text[i] == ' ':
                token[i] = self.idx_mask
            else:
                token[i] = self.p2idx[text[i]]
        if len(text) >= self.word_len:
            token[-1] = self.EOS
        else:
            token[len(text)] = self.EOS

        return token

    def char_token(self, all_texts):
        texts = torch.zeros((1, len(all_texts), self.word_len))
        for i in range(len(all_texts)):
            t = self.tokenize(all_texts[i])  
            texts[0, i] += t

        return texts.long()