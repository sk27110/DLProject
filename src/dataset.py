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
import timm
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class ImageDataset(Dataset):
    def __init__(self, df, root, transform):
        self.df=df
        self.root=root
        self.transform=transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root, "Images", row['image'])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        return image



def preprocess_vit_embeddings(df, root_dir, save_path, transform, batch_size):

    if os.path.exists(save_path):
        return pd.read_pickle(save_path)
    
    dataset = ImageDataset(df, root_dir, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


    vit_model = timm.create_model('vit_small_patch16_224', pretrained=True)
    vit_model.head = torch.nn.Identity()
    for p in vit_model.parameters():
        p.requires_grad = False
    vit_model.eval()

    all_embeddings = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vit_model = vit_model.to(device)
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="ViT embedding"):
            batch = batch.to(device)
            emb = vit_model(batch)
            all_embeddings.append(emb.cpu())

    embeddings = torch.cat(all_embeddings)

    df = df.copy()
    df["vit_embedding"] = list(embeddings.numpy())

    df.to_pickle(save_path)

    return df




class FlickrDataset(Dataset):
    def __init__(self, df, tokenizer=None, max_len=20):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # уже готовый embedding
        image_emb = torch.tensor(row["vit_embedding"], dtype=torch.float32)

        caption = row["caption"]
        caption_tokens = self.tokenizer(
            caption,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        return image_emb, caption_tokens["input_ids"].squeeze(0), caption_tokens["attention_mask"].squeeze(0)

    


def get_datasets(batch_size=8):
    path = kagglehub.dataset_download("adityajn105/flickr8k")
    df = pd.read_csv(StringIO(open(path + "/captions.txt").read()))


    all_images = df["image"].unique()

    train_imgs, tmp_imgs = train_test_split(all_images, test_size=0.2, random_state=42)
    val_imgs, test_imgs = train_test_split(tmp_imgs, test_size=0.5, random_state=42)

    df_train = df[df["image"].isin(train_imgs)].reset_index(drop=True)
    df_val   = df[df["image"].isin(val_imgs)].reset_index(drop=True)
    df_test  = df[df["image"].isin(test_imgs)].reset_index(drop=True)

    train_file = os.path.join(path, "df_train_vit.pkl")
    val_file   = os.path.join(path, "df_val_vit.pkl")
    test_file  = os.path.join(path, "df_test_vit.pkl")

    # Обрабатываем только train (обычно достаточно)
    df_train = preprocess_vit_embeddings(df_train, path, train_file, transform_train, batch_size)

    # Если нужно — можно делать тоже для val/test
    df_val  = preprocess_vit_embeddings(df_val,  path, val_file, transform_val, batch_size)
    df_test = preprocess_vit_embeddings(df_test, path, test_file, transform_val, batch_size)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_dataset = FlickrDataset(df_train, tokenizer, 20)
    val_dataset = FlickrDataset(df_val, tokenizer, 20)
    test_dataset = FlickrDataset(df_test, tokenizer, 20)

    return train_dataset, val_dataset, test_dataset
