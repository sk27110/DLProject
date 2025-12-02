from tqdm import tqdm
import torch
from src.utils import generate_square_subsequent_mask
import os
import wandb
import logging


class Trainer:
    def __init__(self, name, model, train_loader, val_loader, num_epoch, criterion, optimizer, checkpoint_dir,
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


        if self.resume:
            self._load_checkpoint(self.last_checkpoint)


    def _validate(self):
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

        return avg_loss


    def _load_checkpoint(self, checkpoint_name):
        path = os.path.join(self.checkpoint_dir, checkpoint_name)
        checkpoint = torch.load(path, map_location=self.device)

        model_state_dict = self.model.state_dict()
        trained_params = checkpoint["model_state"]
        model_state_dict.update(trained_params)
        self.model.load_state_dict(model_state_dict)

        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if self.scheduler and checkpoint.get("scheduler_state") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])

        self.start_epoch = checkpoint["epoch"] + 1
        self.logger.info(
            f"Resumed from checkpoint {checkpoint_name}, starting at epoch {self.start_epoch}"
        )
        

