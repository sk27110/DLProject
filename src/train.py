from tqdm import tqdm
import torch
from src.utils import generate_square_subsequent_mask
import os
import wandb
import logging
from src.models import ImageCaptioningModelTransformer


class Trainer:
    def __init__(self, name, model, train_loader, val_loader, num_epoch, criterion, optimizer, checkpoint_dir, vit_model,
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
        self.name=name
        self.vit_model=vit_model


        os.makedirs("logs", exist_ok=True)
        log_file = f"logs/{self.name}.log"

        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.INFO)

        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.INFO)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        if not self.logger.handlers:
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

        self.logger.info("Logger initialized.")


        if use_wandb:
            wandb.init(
                project=self.project_name,
                name=self.name,
                config={
                    "epochs": num_epoch,
                    "batch_size": train_loader.batch_size,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "scheduler": type(scheduler).__name__ if scheduler else None
                }
            )
            wandb.watch(self.model, log="all", log_freq=100)

        if self.resume:
            self._load_checkpoint(self.last_checkpoint, self.vit_model)


    def _train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0

        try:
            for step, (images, captions, attention_mask) in enumerate(
                tqdm(self.train_loader, desc="Train")
            ):
                images = images.to(self.device)
                captions = captions.to(self.device)
                attention_mask = attention_mask.to(self.device)

                self.optimizer.zero_grad()


                tgt_mask = generate_square_subsequent_mask(
                    captions.size(1)
                ).to(self.device)

                padding_mask = (attention_mask == 0)   # bool tensor

                output = self.model(
                    images, 
                    captions, 
                    tgt_mask=tgt_mask, 
                    padding_mask=padding_mask
                )

                target = captions[:, 1:]
                output = output[:, :-1, :]

                loss = self.criterion(
                    output.reshape(-1, output.size(-1)),
                    target.reshape(-1)
                )

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

        except Exception:
            self.logger.error(
                f"Error during training at epoch {epoch}",
                exc_info=True
            )
            raise

        return total_loss / len(self.train_loader)



    def _validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for step, (images, captions, attention_mask) in enumerate(tqdm(self.val_loader, desc="Validate")):
                images = images.to(self.device)
                captions = captions.to(self.device)
                tgt_mask = generate_square_subsequent_mask(captions.size(1)).to(self.device)

                padding_mask = (attention_mask == 0)

                output = self.model(
                    images, 
                    captions, 
                    tgt_mask=tgt_mask,
                    padding_mask=padding_mask
                )
                
                target = captions[:, 1:]
                output = output[:, :-1, :]
                loss = self.criterion(output.reshape(-1, output.size(-1)), target.reshape(-1))
                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)

        return avg_loss


    def train(self):
        try:
            for epoch in range(self.start_epoch, self.num_epoch):
                self.logger.info(f"Epoch {epoch}")

                train_loss = self._train_one_epoch(epoch)
                val_loss = self._validate()

                self.logger.info(
                    f"Train_loss: {train_loss:.4f} | Val_loss: {val_loss:.4f}"
                )

                self._save_checkpoint(epoch, val_loss)

                if self.use_wandb:
                    wandb.log({
                        "val_loss epoch": val_loss,
                        "train_loss epoch": train_loss,
                        "epoch": epoch
                    })

            if self.use_wandb:
                wandb.finish()

        except Exception as e:
            self.logger.error("Training failed with exception:", exc_info=True)
            raise e



    def _save_checkpoint(self, epoch, val_loss):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        filename = f"{self.name}_checkpoint_epoch_{epoch}.pth"
        path = os.path.join(self.checkpoint_dir, filename)

        checkpoint = {
            "epoch": epoch,
            "val_loss": val_loss,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "model_state": self.model.state_dict(),
            "model_config": self.model.config
        }

        torch.save(checkpoint, path)
        self.logger.info(f"Saving {self.name}_checkpoint_epoch_{epoch} to {path}")


    def _load_checkpoint(self, checkpoint_name, vit_model):
        path = os.path.join(self.checkpoint_dir, checkpoint_name)
        checkpoint = torch.load(path, map_location=self.device)

        self.model = ImageCaptioningModelTransformer(
            vocab_size=checkpoint['model_config']['vocab_size'],
            vit_model=vit_model,
            decoder_dim=checkpoint['model_config']['decoder_dim'],
            nhead=checkpoint['model_config']['nhead'],
            num_layers=checkpoint['model_config']['num_layers'],
            max_len=checkpoint['model_config']['max_len'],
            dropout=checkpoint['model_config']['dropout']
        )
        
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.to(self.device)

        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if self.scheduler and checkpoint.get("scheduler_state") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])

        self.start_epoch = checkpoint["epoch"] + 1
        self.logger.info(
            f"Resumed from checkpoint {checkpoint_name}, starting at epoch {self.start_epoch}"
        )
        

