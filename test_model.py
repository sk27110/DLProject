import torch
from transformers import BertTokenizer
from src.eval import evaluate_model
from src.models import ImageCaptioningModelTransformer  # твоя модель
import torch.nn as nn
from src.dataset import get_datasets
import timm
from torch.utils.data import DataLoader


def load_model(checkpoint_path, train, device="cuda"):

    vit_model = timm.create_model('vit_small_patch16_224', pretrained=True)
    vit_model.head = torch.nn.Identity()
    for p in vit_model.parameters():
        p.requires_grad = False

    vit_model.eval()

    model = ImageCaptioningModelTransformer(train.tokenizer.vocab_size, vit_model)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint["model_state_dict"]
    model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()

    return model

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = "./checkpoint/run1_checkpoint_epoch_19.pth"
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train, _, test = get_datasets()
    test_loader = DataLoader(test, batch_size=8, shuffle=False)

    model = load_model(checkpoint, train, device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    results = evaluate_model(model, test_loader, criterion, tokenizer, device)

    print("==== Evaluation on test set ====")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
