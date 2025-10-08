import os
import numpy as np
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
import pickle
import random
random.seed(0)

class GoldDataset(Dataset):
    def __init__(
        self,
        data_file,
        D_file,
        O_file,
        path_to_images,
        tokenizer,
        image_transform,
        cfg,
    ):

        self.data = pd.read_csv(data_file, sep="\t", encoding="utf-8")

        with open(D_file, "rb") as f:
            self.D = pickle.load(f)

        with open(O_file, "rb") as f:
            self.O = pickle.load(f)

        self.path_to_images = path_to_images
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.cfg = cfg
    
        def __getitem__(self, idx):

        row = self.data.iloc[idx, :]

        pid = row["pid"]
        caption = row["text"]
        target_of_sarcasm = str(row["target_of_sarcasm"])
        target_of_sarcasm = target_of_sarcasm if target_of_sarcasm.lower() != "nan" else "No target present"
            
        d = self.D.get(pid, "")
        
        max_length = self.cfg.max_len
        
        encoded_dict = self.tokenizer(
            f"{caption} </s> {d}", 
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            add_prefix_space=True,
        )
        input_ids = encoded_dict["input_ids"][0]
        attention_mask = encoded_dict["attention_mask"][0]

        image_path = os.path.join(self.path_to_images, pid + ".jpg")
        img = np.array(Image.open(image_path).convert("RGB"))
        img_inp = self.image_transform(img)

        encoded_dict = self.tokenizer(
            target_of_sarcasm, 
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            add_prefix_space=True,
        ) 
        ts_ids = encoded_dict["input_ids"][0]
        ts_attn_mask = encoded_dict["attention_mask"][0] 
        
        sample = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_image": img_inp,
            "target_ids": ts_ids,
#             "is_sarcastic": is_sarcastic
        }

        return sample

    def __len__(self):
        return self.data.shape[0]
