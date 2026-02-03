from main import POSDataset, parse_dataset, NUM_TAGS, IDX2TAG
from typing import Tuple, List
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AutoModelForTokenClassification, BertModel
from collections import defaultdict

TEST_DATA = [
    "Късият път през планината е заснежен.",
    "Алчната за пари оперна певица купи на продуцента нов бял мерцедес с вертикални врати?",
    "hey, dai si feisbooka",
    "Kvo 6e praim sled daskalo",
    "Заедно со Соња, тој оди во полициската станица и пред Иља Петрович го признава злосторството што го направил. (Не сака тоа да го стори пред Порфириј, зашто го мрази неговиот цинизам и му е смачено целото негово иследување)."
    "6te hodim li, 4e zamruzvam",
]


def predict(text: str, model, tokenizer):
    idx2tag = model.config.id2label
    words = text.split()
    tokens = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
    )

    word_ids = tokens.word_ids()

    res_p = []

    model.to("cpu")
    model.eval()
    with torch.no_grad():
        logits = model(**tokens).logits
        [predictions] = torch.argmax(logits, dim=2)

        for i, p in enumerate(predictions):
            if word_ids[i] is not None and word_ids[i] != word_ids[i - 1]:
                res_p.append(idx2tag[p.item()])

    return list(zip(words, res_p))


def print_prediction(prediction: List[Tuple[str, str]]):
    for p in prediction:
        print(f"{p[0]}\t{p[1]}")


def run_predictions(model, tokenizer):
    for test_sent in TEST_DATA:
        prediction = predict(
            test_sent,
            model,
            tokenizer,
        )
        print_prediction(prediction)
        print("-" * 20)


def eval_total_accuracy(test_dataset, model):
    dataloader = DataLoader(test_dataset, batch_size=32)

    total = 0
    correct = 0

    model.to("cpu")
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            logits = model(**batch).logits
            predictions = torch.argmax(logits, dim=2)
            mask = batch["labels"] != -100
            correct = correct + (predictions == batch["labels"])[mask].sum().item()
            total = total + mask.sum().item()

    return correct / total


def eval_per_tag_accuracy(test_dataset, model):
    dataloader = DataLoader(test_dataset, batch_size=32)

    total = defaultdict(int)
    correct = defaultdict(int)
    confusion_matrix = torch.zeros((NUM_TAGS, NUM_TAGS))

    model.to("cpu")
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            logits = model(**batch).logits
            predictions = torch.argmax(logits, dim=2)
            mask = batch["labels"] != -100
            for pred, gold in zip(predictions[mask], batch["labels"][mask]):
                if pred.item() == gold.item():
                    correct[gold.item()] += 1

                total[gold.item()] += 1

                confusion_matrix[gold.item()][pred.item()] += 1

    # return total, correct, confusion_matrix

    total_labeled = {
        IDX2TAG[k]:v for (k,v) in total.items()
    }
    correct_labeled = {
        IDX2TAG[k]:v for (k,v) in correct.items()
    }

    return total_labeled, correct_labeled, confusion_matrix


def main():
    model_path = "./output"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)

    # run_predictions(model, tokenizer)
    test_dataset = POSDataset(parse_dataset("test"), tokenizer)
    # total_accuracy = eval_total_accuracy(test_dataset, model)
    *_, confusion_matrix = eval_per_tag_accuracy(test_dataset, model)
    print(confusion_matrix)


if __name__ == "__main__":
    main()
