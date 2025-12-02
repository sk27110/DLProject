import torch
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def evaluate_model(model, dataloader, criterion, tokenizer, device="cuda"):
    model.eval()

    total_loss = 0
    bleu1, bleu2, bleu4 = [], [], []

    smooth = SmoothingFunction().method1

    with torch.no_grad():
        for images, captions, attention_mask in tqdm(dataloader, desc="Testing"):
            images = images.to(device)
            captions = captions.to(device)

            padding_mask = (attention_mask == 0)

            tgt_mask = torch.triu(torch.ones(captions.size(1), captions.size(1)), diagonal=1).bool().to(device)

            outputs = model(
                images, 
                captions, 
                tgt_mask,
                padding_mask=padding_mask
            )

            target = captions[:, 1:]
            outputs = outputs[:, :-1, :]

            loss = criterion(outputs.reshape(-1, outputs.size(-1)), target.reshape(-1))
            total_loss += loss.item()

            # ---- BLEU SCORE ----
            predictions = outputs.argmax(-1).cpu().tolist()
            references = captions[:, 1:].cpu().tolist()

            for pred, ref in zip(predictions, references):
                pred_text = tokenizer.decode(pred, skip_special_tokens=True).split()
                ref_text = tokenizer.decode(ref, skip_special_tokens=True).split()

                bleu1.append(sentence_bleu([ref_text], pred_text, weights=(1, 0, 0, 0), smoothing_function=smooth))
                bleu2.append(sentence_bleu([ref_text], pred_text, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth))
                bleu4.append(sentence_bleu([ref_text], pred_text, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth))

    avg_loss = total_loss / len(dataloader)
    metrics = {
        "loss": avg_loss,
        "bleu1": sum(bleu1) / len(bleu1),
        "bleu2": sum(bleu2) / len(bleu2),
        "bleu4": sum(bleu4) / len(bleu4),
    }

    return metrics
