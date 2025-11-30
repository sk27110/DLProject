import kagglehub
from io import StringIO
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from src.transforms import transform_train, transform_val
from torch.utils.data import Subset


class FlickrDataset(Dataset):
    def __init__(self, df, root_dir, transform=None, tokenizer = None, max_len = 20):
        self.df = df.reset_index(drop=True)
        self.dir = root_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['image']
        caption = row['caption']

        img_path = os.path.join(self.dir, 'Images', img_name)

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        caption_tokens = self.tokenizer(
            text = caption,
            max_length = self.max_len,
            padding = 'max_length',
            truncation = True,
            return_attention_mask = True,
            return_tensors = 'pt'
        )

        return image, caption_tokens['input_ids'].squeeze(0), caption_tokens['attention_mask'].squeeze(0)



def get_datasets():
    path = kagglehub.dataset_download("adityajn105/flickr8k")

    with open(path + '/captions.txt', "r") as f:
        lines = f.readlines()


    data_str = "".join(lines)
    df = pd.read_csv(StringIO(data_str))

    all_images = df['image'].unique()

    train_imgs, tmp_imgs = train_test_split(
    all_images, test_size=0.2, random_state=42
    )

    val_imgs, test_imgs = train_test_split(
        tmp_imgs, test_size=0.5, random_state=42
    )
    
    df_train = df[df['image'].isin(train_imgs)].reset_index(drop=True)[:128]
    df_val   = df[df['image'].isin(val_imgs)].reset_index(drop=True)[:64]
    df_test  = df[df['image'].isin(test_imgs)].reset_index(drop=True)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_dataset = FlickrDataset(df_train, path, transform_train, tokenizer, 20)
    val_dataset = FlickrDataset(df_val, path, transform_val, tokenizer, 20)
    test_dataset = FlickrDataset(df_test, path, transform_val, tokenizer, 20)

    return train_dataset, val_dataset, test_dataset