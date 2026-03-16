# /// script
# dependencies = [
#     "marimo",
#     "requests==2.32.5",
# ]
# requires-python = ">=3.13"
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App()

with app.setup:
    import re
    from typing import NamedTuple, Iterable
    import math
    from collections import defaultdict, Counter
    from io import BytesIO
    import requests
    import tarfile
    from pathlib import Path
    from machine_learning import split_data
    import random


@app.function
def tokenize(text: str) -> set[str]:
    text = text.lower()
    all_words = re.findall(r"[0-9a-z']+", text)
    return set(all_words)


@app.cell
def _():
    tokenize("Data Science is science")
    return


@app.class_definition
class Message(NamedTuple):
    text: str
    is_spam: bool


@app.class_definition
class NaiveBayesClassifier:
    def __init__(self, k: float = 0.5, min_token_count: int = 5) -> None:
        self.k = k
        self.min_token_count = min_token_count
        self.tokens: set[str] = set()
        self.token_spam_count: dict[str, int] = defaultdict(int)
        self.token_ham_count: dict[str, int] = defaultdict(int)
        self.spam_messages = self.ham_messages = 0

    def token_cleaner(self) -> None:
        removal_set = set()
        for token in self.tokens:
            count = self.token_ham_count[token] + self.token_spam_count[token]
            if count <= self.min_token_count:
                removal_set.add(token)

        self.tokens = self.tokens - removal_set

    def train(self, messages: Iterable[Message]) -> None:
        for message in messages:
            if message.is_spam:
                self.spam_messages += 1
            else:
                self.ham_messages += 1

            for token in tokenize(message.text):
                if token[-1] == 's':
                    token = token[:-1]
                if token.isdigit():
                    token = "has_number"
                self.tokens.add(token)
                if message.is_spam:
                    self.token_spam_count[token] += 1
                else:
                    self.token_ham_count[token] += 1

        self.token_cleaner()

    def _probabilities(self, token: str) -> tuple[float, float]:
        spam = self.token_spam_count[token]
        ham = self.token_ham_count[token]

        p_token_spam = (spam + self.k) / (self.spam_messages + 2 * self.k)
        p_token_ham = (ham + self.k) / (self.ham_messages + 2 * self.k)
        return p_token_spam, p_token_ham

    def predict(self, text: str) -> float:
        text_tokens = tokenize(text)
        log_prob_ham = log_prob_spam = 0

        for token in self.tokens:
            prob_if_spam, prob_if_ham = self._probabilities(token)
            if token in text_tokens:
                log_prob_spam += math.log(prob_if_spam)
                log_prob_ham += math.log(prob_if_ham)
            else:
                log_prob_spam += math.log(1.0 - prob_if_spam)
                log_prob_ham += math.log(1.0 - prob_if_ham)

        prob_if_spam = math.exp(log_prob_spam)
        prob_if_ham = math.exp(log_prob_ham)
        return prob_if_spam / (prob_if_spam + prob_if_ham)


@app.cell
def _():
    messages = [
        Message("spam rules", is_spam=True),
        Message("ham rules", is_spam=False),
        Message("hello ham", is_spam=False),
    ]
    model = NaiveBayesClassifier(min_token_count=0)
    model.train(messages)
    return (model,)


@app.cell
def _(model):
    model.tokens
    return


@app.cell
def _(model):
    text = "Hello spam"
    model.predict(text)
    return


@app.cell
def _():
    BASE_URL = "https://spamassassin.apache.org/old/publiccorpus"
    FILES = [
        "20030228_easy_ham.tar.bz2",
        "20030228_hard_ham.tar.bz2",
        "20030228_spam.tar.bz2",
    ]
    OUTPUT_DIR = "data/spam"
    for filename in FILES:
        content = requests.get(f"{BASE_URL}/{filename}").content
        fin = BytesIO(content)
        with tarfile.open(fileobj=fin, mode="r:bz2") as tf:
            tf.extractall(OUTPUT_DIR)
        print(f"Downloaded: {filename}")
    return (OUTPUT_DIR,)


@app.cell
def _(OUTPUT_DIR):
    data: list[Message] = []

    for file_path in Path(OUTPUT_DIR).glob("*/*"):
        is_spam = "ham" not in file_path.parent.name
        with file_path.open(errors="ignore") as email_file:
            text_data = ""
            for line in email_file:
                if line.startswith("Subject:"):
                    text_data += line.lstrip("Subject: ")
                    break
                if "From:" in line:
                    text_data += (
                        " " + line.lstrip("From: ").split("@")[-1].split(".")[0]
                    )
            data.append(Message(text_data, is_spam))
    return (data,)


@app.cell
def _(data: list[Message]):
    len(data)
    return


@app.cell
def _(data: list[Message]):
    random.seed(0)
    train_messages, test_messages = split_data(data, 0.75)
    spam_model = NaiveBayesClassifier(min_token_count=2)
    spam_model.train(train_messages)
    return spam_model, test_messages


@app.cell
def _(spam_model, test_messages):
    predictions = [
        (message, spam_model.predict(message.text)) for message in test_messages
    ]
    confusion_matrix = Counter(
        (message.is_spam, spam_prob > 0.5) for message, spam_prob in predictions
    )
    for val, count in confusion_matrix.items():
        print(f"{val} -> {count}")
    return (confusion_matrix,)


@app.cell
def _(confusion_matrix):
    precision = confusion_matrix[(True, True)] / (
        confusion_matrix[(True, True)] + confusion_matrix[(False, True)]
    )
    recall = confusion_matrix[(True, True)] / (
        confusion_matrix[(True, True)] + confusion_matrix[(True, False)]
    )
    precision * 100, recall * 100
    return


@app.function
def p_spam_given_token(token: str, model: NaiveBayesClassifier) -> float:
    prob_if_spam, prob_if_ham = model._probabilities(token)
    return prob_if_spam / (prob_if_spam + prob_if_ham)


@app.cell
def _(spam_model):
    words = sorted(
        spam_model.tokens, key=lambda t: p_spam_given_token(t, spam_model)
    )
    print("spammiest_words", words[-10:])
    print("hammiest_words", words[:10])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
