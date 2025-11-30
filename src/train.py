from tqdm import tqdm
import torch
from src.utils import generate_square_subsequent_mask
import os
import wandb

class Trainer:
    def __init__(self, model, train_loader, val_loader, num_epoch, criterion, optimizer, checkpoint_dir,
                 scheduler=None, resume=False, last_checkpoint=None, use_wandb=False, project_name='Image caption'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.num_epoch = num_epoch
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        self.start_epoch = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.resume = resume
        self.last_checkpoint = last_checkpoint
        self.use_wandb = use_wandb
        self.project_name = project_name
        self.best_val_loss = float("inf")

        if use_wandb:
            wandb.init(
                project=self.project_name,
                config={
                    "epochs": num_epoch,
                    "batch_size": train_loader.batch_size,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "scheduler": type(scheduler).__name__ if scheduler else None
                }
            )
            wandb.watch(self.model, log="all", log_freq=100)

        if self.resume:
            self._load_checkpoint(self.last_checkpoint)

    def _train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0

        for step, (images, captions, attention_mask) in enumerate(tqdm(self.train_loader, desc="Train")):
            images = images.to(self.device)
            captions = captions.to(self.device)

            self.optimizer.zero_grad()
            tgt_mask = generate_square_subsequent_mask(captions.size(1)).to(self.device)

            output = self.model(images, captions, tgt_mask)

            target = captions[:, 1:]
            output = output[:, :-1, :]

            loss = self.criterion(output.reshape(-1, output.size(-1)), target.reshape(-1))
            loss.backward()
            self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()

            total_loss += loss.item()

            if self.use_wandb:
                wandb.log({
                    "train_loss": loss.item(),
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                    "step": epoch * len(self.train_loader) + step
                })

        return total_loss / len(self.train_loader)


    def _validate(self, epoch):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for step, (images, captions, attention_mask) in enumerate(tqdm(self.val_loader, desc="Validate")):
                images = images.to(self.device)
                captions = captions.to(self.device)
                tgt_mask = generate_square_subsequent_mask(captions.size(1)).to(self.device)

                output = self.model(images, captions, tgt_mask)
                target = captions[:, 1:]
                output = output[:, :-1, :]
                loss = self.criterion(output.reshape(-1, output.size(-1)), target.reshape(-1))
                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)

        if self.use_wandb:
            wandb.log({
                "val_loss": avg_loss,
                "epoch": epoch
            })

        return avg_loss


    def train(self):
        for epoch in range(self.start_epoch, self.num_epoch):
            print(f"Epoch {epoch}")
            train_loss = self._train_one_epoch(epoch)
            val_loss = self._validate(epoch)
            print(f"Train_loss: {train_loss:.4f} | Val_loss: {val_loss:.4f}")
            self._save_checkpoint(epoch, val_loss, best=False)


    def _save_checkpoint(self, epoch, val_loss):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        filename = f"checkpoint_epoch{epoch}.pth"

        path = os.path.join(self.checkpoint_dir, filename)

        checkpoint = {
            "epoch": epoch,
            "val_loss": val_loss,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "model_state": self.model.state_dict()
        }

        torch.save(checkpoint, path)

        if self.use_wandb:
            wandb.save(path)

    def _load_checkpoint(self, checkpoint_name):
        path = os.path.join(self.checkpoint_dir, checkpoint_name)
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])

        if self.scheduler and checkpoint['scheduler_state']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])

        self.start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from checkpoint: {checkpoint_name}, starting at epoch {self.start_epoch}")
