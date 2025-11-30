from src.dataset import get_datasets
from src.train import Trainer
from src.models import ImageCaptioningModelTransformer
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup


def main():
    train, val, test = get_datasets()

    train_loader = DataLoader(train, batch_size=8, shuffle=True)
    val_loader = DataLoader(val, batch_size=8, shuffle=False)

    model = ImageCaptioningModelTransformer(train.tokenizer.vocab_size)


    num_epochs = 5
    checkpoint_dir = "/checkpoint"

    criterion = nn.CrossEntropyLoss(ignore_index=train.tokenizer.pad_token_id)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=500,
        num_training_steps=len(train_loader) * num_epochs,
    )
    

    trainer = Trainer(
        model=model, 
        train_loader=train_loader,
        val_loader=val_loader,
        num_epoch=num_epochs, criterion=criterion, 
        optimizer=optimizer,
        checkpoint_dir=checkpoint_dir, 
        scheduler=scheduler,    
        resume=False,
        use_wandb=True,
        project_name="Transformer Image Caption"
        )

    trainer.train()


if __name__ == '__main__':
    main()