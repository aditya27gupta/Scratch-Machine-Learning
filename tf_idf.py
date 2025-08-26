"""Class to implement TF-IDF (Term Frequency-Inverse Document Frequency) algorithm."""

import math
import re
from pathlib import Path


class TfIdf:
    def __init__(self, documents: list[str], stopwords: str = "none") -> None:
        self.stop_words = self.load_stop_words(stopwords)
        self.documents = self.clean_documents(documents)

    def load_stop_words(self, stopwords: str) -> list[str]:
        stop_words = []
        if stopwords == "english":
            file_path = "docs/stop_words.txt"
            path = Path(file_path)
            if path.is_file():
                with path.open() as file:
                    data = file.read()
                    stop_words = [word.strip() for word in data.split()]
        return stop_words

    def clean_documents(self, documents: list[str]) -> list[str]:
        cleaned_documents = []
        for document in documents:
            doc = document.lower()
            for word in self.stop_words:
                pattern = r"\b" + re.escape(word) + r"\b"
                doc = re.sub(pattern, "", doc, flags=re.IGNORECASE)
            doc = re.sub(r"[^a-zA-Z0-9 ]+", "", doc.lower())
            doc = re.sub(r"\s+", " ", doc).strip()
            cleaned_documents.append(doc)
        return cleaned_documents

    def calculate_tfidf(self) -> dict[str, list[float]]:
        """Calculate the TF-IDF score for each word in the documents."""
        tfidf_scores = {}
        total_documents = len(self.documents)
        all_words = set()

        for document in self.documents:
            words = document.split()
            all_words.update(words)

        for word in all_words:
            if word in self.stop_words:
                continue
            tfidf_scores[word] = []
            doc_freq = sum(1 for document in self.documents if word in document)
            idf = math.log10(total_documents / (doc_freq + 1))
            for document in self.documents:
                tf = document.split().count(word) / len(document.split())
                tfidf_scores[word].append(tf * idf)

        return tfidf_scores


def main() -> None:
    doc_path = Path("docs/sample.txt")
    with doc_path.open() as file:
        documents = [doc.strip() for doc in file.readlines() if doc.strip()]
    tfidf = TfIdf(documents=documents, stopwords="english")
    tfidf_scores = tfidf.calculate_tfidf()
    print(tfidf_scores)


if __name__ == "__main__":
    main()
