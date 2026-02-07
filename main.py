from typing import Literal, List, Tuple, cast, Dict
from collections import Counter
from transformers import (
    BertTokenizer,
    AutoModelForTokenClassification,
    BatchEncoding,
    TrainingArguments,
    DataCollatorForTokenClassification,
    Trainer,
)
from torch.utils.data import Dataset
import torch
import numpy as np

PrimitiveDataset = List[Tuple[List[str], List[str]]]

NUM_TAGS = 16

TAG2IDX = {
    "ADP": 0,
    "NOUN": 1,
    "PUNCT": 2,
    "VERB": 3,
    "AUX": 4,
    "PRON": 5,
    "ADJ": 6,
    "PART": 7,
    "ADV": 8,
    "INTJ": 9,
    "DET": 10,
    "PROPN": 11,
    "CCONJ": 12,
    "NUM": 13,
    "SCONJ": 14,
    "X": 15,
}
IDX2TAG = {
    0: "ADP",
    1: "NOUN",
    2: "PUNCT",
    3: "VERB",
    4: "AUX",
    5: "PRON",
    6: "ADJ",
    7: "PART",
    8: "ADV",
    9: "INTJ",
    10: "DET",
    11: "PROPN",
    12: "CCONJ",
    13: "NUM",
    14: "SCONJ",
    15: "X",
}


def parse_dataset(
    dataset: Literal["train"] | Literal["dev"] | Literal["test"],
) -> PrimitiveDataset:
    assert dataset in ["train", "dev", "test"]

    tokens = []

    with open(f"./corpus/bg_btb-ud-{dataset}.conllu") as file:
        sents = file.read().split("\n" * 2)
        for sent in sents:
            if not sent:
                continue

            sent_words = []
            sent_pos_types = []

            rows = sent.split("\n")
            for r in rows:
                if r[0] == "#":
                    continue
                _, word, _, pos_type, *_ = r.split("\t")
                sent_words.append(word)
                sent_pos_types.append(pos_type)

            tokens.append((sent_words, sent_pos_types))

    return tokens


def count_tokens(dataset: PrimitiveDataset) -> Counter:
    tokens = [token for (_, sent_tokens) in dataset for token in sent_tokens]
    return Counter(tokens)


def tokenize_and_assign_pos_tag(sent_words, sent_tags, tokenizer) -> Dict:
    T = tokenizer(
        sent_words,
        is_split_into_words=True,
        max_length=128,
        truncation=True,
        padding="max_length",
    )
    # T = cast(BatchEncoding, T)
    word_ids = T.word_ids()
    padded_batch_encoding = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }

    for i, (ii, tti, am) in enumerate(
        zip(
            T["input_ids"],
            T["token_type_ids"],
            T["attention_mask"],
        )
    ):
        padded_batch_encoding["input_ids"].append(ii)
        padded_batch_encoding["attention_mask"].append(am)

        padded_batch_encoding["labels"].append(
            TAG2IDX[sent_tags[word_ids[i]]]
            if word_ids[i] is not None and word_ids[i] != word_ids[i - 1]
            else -100
        )

    return padded_batch_encoding


class POSDataset(Dataset):
    def __init__(self, pds: PrimitiveDataset, tokenizer):
        self.tokenizer = tokenizer
        self.pds = pds
        self.tds = []
        for sent_words, sent_tokens in self.pds:
            pbe = tokenize_and_assign_pos_tag(sent_words, sent_tokens, self.tokenizer)
            self.tds.append(
                {
                    "input_ids": torch.tensor(pbe["input_ids"]),
                    "attention_mask": torch.tensor(pbe["attention_mask"]),
                    "labels": torch.tensor(pbe["labels"]),
                }
            )

    def __len__(self):
        return len(self.tds)

    def __getitem__(self, idx):  # type: ignore (the LSP complains)
        return self.tds[idx]


def test_tokenizer():
    # print(run_tokenizer())
    # print(run_tokenizer().word_ids())
    # test_parse()
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    tokenizer = cast(BertTokenizer, tokenizer)
    dataset = parse_dataset("train")

    r1_sents = dataset[0][0]
    r1_tags = dataset[0][1]
    r1 = tokenize_and_assign_pos_tag(r1_sents, r1_tags, tokenizer)
    r1_words = tokenizer.convert_ids_to_tokens(r1["input_ids"])

    print(r1_sents)
    print(r1_tags)

    print("Token\tLabel\tTag")
    for i in range(len(r1["input_ids"])):
        print(
            f"{r1_words[i]}\t{r1['labels'][i]}\t{IDX2TAG.get(r1['labels'][i], 'IGN')}"
        )


def compute_metrics(eval_prediction):
    (predictions, label_ids) = eval_prediction
    predictions = np.argmax(predictions, axis=2)

    total = 0
    correct = 0

    for i in range(len(predictions)):
        compare_tuples = [t for t in zip(predictions[i], label_ids[i]) if t[1] != -100]
        total = total + len(compare_tuples)
        correct = correct + sum(1 for t in compare_tuples if t[0] == t[1])

    return {"accuracy": correct / total}


def create_trainer(model, train_dataset, test_dataset, tokenizer):
    # https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments
    training_args = TrainingArguments(
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        warmup_ratio=0.1,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )

    return trainer


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    tokenizer = cast(BertTokenizer, tokenizer)
    train_dataset = POSDataset(parse_dataset("train"), tokenizer)
    test_dataset = POSDataset(parse_dataset("test"), tokenizer)
    # print(dataset[0])
    # print(dataset[0]["labels"].size())
    model = AutoModelForTokenClassification.from_pretrained(
        "bert-base-multilingual-cased",
        num_labels=NUM_TAGS,
        id2label=IDX2TAG,
        label2id=TAG2IDX,
    )
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    trainer = create_trainer(model, train_dataset, test_dataset, tokenizer)
    trainer.train()
