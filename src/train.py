from tqdm import tqdm
import torch
from scr.utils import generate_square_subsequent_mask

class Trainer:
    def __init__(self, model, train_loader, val_loader, num_epoch, criterion, optimizer, scheduler):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.num_epoch = num_epoch
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _train_one_epoch(self):
        self.model.train()
        total_loss = 0
        for images, captions, attention_mask in tqdm(self.train_loader, desc="Train"):
            images = images.to(self.device)
            captions = captions.to(self.device)

            self.optimizer.zero_grad()

            tgt_mask = generate_square_subsequent_mask(captions.size(1)).to(self.device)

            output = self.model(images, captions, tgt_mask)  # [batch, max_len, vocab_size]

            # Shift targets на один токен
            target = captions[:, 1:]          # пропускаем [CLS] токен
            output = output[:, :-1, :]        # предсказываем до последнего токена

            # Вычисление loss
            loss = self.criterion(output.reshape(-1, output.size(-1)), target.reshape(-1))
            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def _validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for images, captions, attention_mask in tqdm(self.val_loader, desc="Validate"):
                images = images.to(self.device)
                captions = captions.to(self.device)
                tgt_mask = generate_square_subsequent_mask(captions.size(1)).to(self.device)
                output = self.model(images, captions, tgt_mask)
                target = captions[:, 1:]  
                output = output[:, :-1, :] 
                loss = self.criterion(output.reshape(-1, output.size(-1)), target.reshape(-1))
                total_loss += loss.item()
            
        return total_loss / len(self.val_loader)
    
    def train(self):
        for epoch in range(self.num_epoch):
            print("Epoch:", epoch)
            test_loss = self._train_one_epoch()
            val_loss = self._validate()
            print("Train_loss:", test_loss, "Val_loss:", val_loss)

