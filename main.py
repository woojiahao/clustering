import logging
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import cosine
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
    clusters = fcluster(Z, 3, criterion="maxclust")
    key_with_cluster = {}
    for i, (e, c) in enumerate(zip(entries, clusters)):
        if c not in key_with_cluster:
            key_with_cluster[c] = []
        key_with_cluster[c].append((e.key, embeddings[i]))

    centroids = {
        cluster: np.mean(np.array([v[1] for v in vectors]), axis=0)
        for cluster, vectors in key_with_cluster.items()
    }

    # target = Entry(
    #     key="z", description="This is one of the other clusters", categories=["insect"]
    # )
    target = Entry(
        key="z",
        description="This is only an insect, but maybe an animal",
        categories=["insect"],
    )

    target_embedding = model.encode(target.description)

    closest_cluster = None
    min_distance = 10**9
    for cluster, centroid in centroids.items():
        distance = cosine(target_embedding, centroid)
        if distance < min_distance:
            min_distance = distance
            closest_cluster = cluster

    print(closest_cluster)
    print(min_distance)

    print([v[0] for v in key_with_cluster[closest_cluster]])
