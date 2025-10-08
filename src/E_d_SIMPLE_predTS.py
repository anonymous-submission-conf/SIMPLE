import os
import re
import numpy as np
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
import pickle


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

    #Returns clean version of text
    def clean_data(self, text):

        #Fix unicode errors
        t = re.sub(r"Œì√á√ñ | Œì√á√ø | Œì√á¬™", "'", text) #Single quotes
        t = re.sub(r"Œì√á¬£ | Œì√á¬•", "\"", t) #Double quotes
        t = re.sub(r"Œì√©¬º", "$", t) #Dollar
        t = re.sub(r"Œì√á√∂ | Œì√á√¥", "-", t) #Hyphen
        t = re.sub(r"Œì√á‚ñì", "`", t) #Backticks
        t = re.sub(r"Œì√©‚ï£", "₹", t) #Rupee
        t = re.sub(r"Œ±‚ñì√°", "ಠ", t) #Eye
        t = re.sub(r"Œì√§√≥", "™", t) #Trademark  
        t = re.sub(r"\s+", " ", t).strip() 

        #Remove quotes from ends of caption
        t = re.sub(r"^(\s*['\"]+\s*)+", "", t)
        t = re.sub(r"(\s*['\"]+\s*)+$", "", t).strip()
        
        #Emoji removal
        t = re.sub(r'emoji_\w*\b', "", t)
        t = re.sub(r"\s+", " ", t).strip() 

        #Website URL removal
        t = re.sub(r"https://[^\s,]*", "", t)
        t = re.sub(r"\s+", " ", t).strip()
    
        #  #Hashtag removal
        #  t = re.sub(r"(\s*#\s*\S*\s*)+$", "", t)
        #  t = re.sub(r"#", "", t)
        #  t = re.sub(r"\s+", " ", t).strip()

        return t
    
    def __getitem__(self, idx):

        row = self.data.iloc[idx, :]

        pid = row["pid"]
        caption = row["text"]
        explanation = row["explanation"]
#         is_sarcastic = row["is_sarcastic"]
#         caption = self.clean_data(caption)

        max_length = self.cfg.max_len
        
        d = self.D.get(pid, "")
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
            explanation,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            add_prefix_space=True,
        )

        explanation_ids = encoded_dict["input_ids"][0]

        sample = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_image": img_inp,
            "target_ids": explanation_ids,
            "caption": caption,
#             "is_sarcastic": is_sarcastic
        }

        return sample

    def __len__(self):
        return self.data.shape[0]
