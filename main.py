from src.dataset import get_datasets
from src.train import Trainer
from src.models import ImageCaptioningModelTransformer
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
import timm
import torch
import argparse


def main(run_name, num_epoch):
    train, val, test = get_datasets()

    vit_model = timm.create_model('vit_small_patch16_224', pretrained=True)
    vit_model.head = torch.nn.Identity()
    for p in vit_model.parameters():
        p.requires_grad = False

    vit_model.eval()

    model = ImageCaptioningModelTransformer(train.tokenizer.vocab_size, vit_model)
    train_loader = DataLoader(train, batch_size=8, shuffle=True)
    val_loader = DataLoader(val, batch_size=8, shuffle=False)

    checkpoint_dir = "./checkpoint"

    criterion = nn.CrossEntropyLoss(ignore_index=train.tokenizer.pad_token_id)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=500,
        num_training_steps=len(train_loader) * num_epoch,
    )
    

    trainer = Trainer(
        model=model, 
        train_loader=train_loader,
        val_loader=val_loader,
        num_epoch=num_epoch, 
        criterion=criterion, 
        optimizer=optimizer,
        checkpoint_dir=checkpoint_dir, 
        scheduler=scheduler,    
        resume=False,
        use_wandb=True,
        project_name="Transformer Image Caption",
        name=run_name,
        last_checkpoint=None
        )

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default=None, help="Example parameter Run 1")
    parser.add_argument("--num_epoch", type=int, default=5)
    args = parser.parse_args()

    main(args.run_name, args.num_epoch)