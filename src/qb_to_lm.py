from typing import List
import json
import click
import spacy
from tqdm import tqdm


def write_fold(nlp, pages: List[str], texts: List[str], out_path: str):
    with open(out_path, "w") as f:
        tokenized_texts = nlp.pipe(texts, n_process=-1, batch_size=512)
        for title, doc in zip(tqdm(pages), tokenized_texts):
            f.write(f'= {title.replace("_", " ")} =\n\n')
            for sent in doc.sents:
                f.write(" ".join(t.text for t in sent))
                f.write("\n")


@click.command()
@click.argument("mapped_path")
@click.argument("train_path")
@click.argument("dev_path")
def main(mapped_path: str, train_path: str, dev_path: str):
    with open(mapped_path) as f:
        all_questions = json.load(f)["questions"]

    train_questions = []
    train_texts = []
    train_pages = []
    dev_questions = []
    dev_texts = []
    dev_pages = []
    for q in all_questions:
        if q["page"] is not None:
            if q["fold"] == "guesstrain":
                train_questions.append(q)
                train_texts.append(q["text"])
                train_pages.append(q["page"])
            elif q["fold"] == "guessdev":
                dev_questions.append(q)
                dev_texts.append(q["text"])
                dev_pages.append(q["page"])
    nlp = spacy.load("en_core_web_sm")
    write_fold(nlp, train_pages, train_texts, train_path)
    write_fold(nlp, dev_pages, dev_texts, dev_path)


if __name__ == "__main__":
    main()
