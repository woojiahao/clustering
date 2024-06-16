import logging
import pandas as pd

from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from typing import List

logging.basicConfig(level=logging.WARN)


@dataclass
class Entry:
    key: str
    categories: List[str]
    description: str


data = [
    Entry(
        key="a",
        categories=["animal", "insect"],
        description="This is both an animal and an insect",
    ),
    Entry(
        key="b",
        categories=["plant", "insect"],
        description="This is both a plant and an insect",
    ),
    Entry(key="c", categories=["insect"], description="This is only an insect"),
    Entry(key="d", categories=["insect"], description="This is only an insect"),
    Entry(key="e", categories=["plant"], description="This is only a plant"),
    Entry(key="f", categories=["animal"], description="This is only an animal"),
]

model = SentenceTransformer("all-MiniLM-L6-v2")

grouping = {}
for entry in data:
    for category in entry.categories:
        if category not in grouping:
            grouping[category] = []
        grouping[category].append(entry)

for category, entries in grouping.items():
    if len(entries) < 3:
        continue

    sentences = [e.description for e in entries]
    embeddings = model.encode(sentences)

    Z = linkage(embeddings, method="weighted", metric="cosine")
    clusters = fcluster(Z, 2, criterion="maxclust")
    key_with_cluster = {}
    for e, c in zip(entries, clusters):
        if c not in key_with_cluster:
            key_with_cluster[c] = set()
        key_with_cluster[c].add(e.key)

    print(key_with_cluster)

    target = Entry(
        key="z", description="This is one of the other clusters", categories=["insect"]
    )

    target_embedding = model.encode(target.description)
