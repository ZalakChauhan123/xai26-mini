#!/usr/bin/env python
# coding: utf-8
#
# 1__analysis.py
# -----------------------------------------------------------------------------
# Exploratory analysis of the AIFB RDF knowledge graph.
# Converted from notebooks/1__analysis.ipynb.
#
# Produces (into outputs/plots/Analysis/):
#   - Top 20 - Predicate Distribution.png
#   - Top 20 - Node Types Distribution.png
#   - Research Group Label Distribution.png
# -----------------------------------------------------------------------------

from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless-safe: render straight to PNG files

import rdflib as rdf
from rdflib import Graph
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths (resolved relative to the repo root, independent of the CWD)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "dataset" / "aifbfixed_complete.n3"
ANALYSIS_DIR = ROOT / "outputs" / "plots" / "Analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Load Graph
# ---------------------------------------------------------------------------
g = Graph()
g.parse(str(DATASET_PATH))

print("Graph loaded successfully!")
print("Number of triples:", len(g))

subjects = set()
objects = set()

for s, p, o in g:
    subjects.add(s)
    objects.add(o)

all_nodes = subjects.union(objects)

print("Unique subjects:", len(subjects))
print("Unique objects:", len(objects))
print("Total unique nodes:", len(all_nodes))

# Get all unique predicates (properties)
predicates = set()
for s, p, o in g:
    predicates.add(str(p))

# Count of Total Unique Predicates
print(f"\nTotal unique predicates: {len(predicates)}")

# Show first 20 Properties
print("\nAll properties in dataset:")
for pred in sorted(predicates)[:20]:  # Show first 20
    print(f"  - {pred.split('#')[-1]}")  # Show readable name

# Count how many times each property appears
predicates_count = Counter()
for s, p, o in g:
    predicates_count[str(p).split('#')[-1]] += 1

print("Top 15 most common properties:")
for pred, count in predicates_count.most_common(15):
    print(f"  {pred}: {count}")

# What are the main entities?
rdf_type = rdf.term.URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")

entity_types = Counter()
for s, p, o in g.triples((None, rdf_type, None)):
    entity_types[str(o).split('#')[-1]] += 1

print("Entity types in dataset:")
for etype, count in entity_types.most_common(20):
    print(f"  {etype}: {count}")

# Show first 10 triples
print("Sample triples from dataset:")
for i, (s, p, o) in enumerate(g):
    if i >= 10:
        break
    print(f"{str(s)[-30:]} → {str(p).split('#')[-1]} → {str(o)[-40:]}")

degree_counter = Counter()

for s, p, o in g:
    degree_counter[s] += 1
    degree_counter[o] += 1

print("\nTop connected nodes:\n")

for node, degree in degree_counter.most_common(10):
    print(degree, node)

for s, p, o in g:
    if "publication" in str(p).lower():
        print(s, p, o)
        break

# Build a DataFrame view of the triples
triples = []

for s, p, o in g:
    triples.append([str(s), str(p), str(o)])

df = pd.DataFrame(triples, columns=["subject", "predicate", "object"])

print(df.head())


# ---------------------------------------------------------------------------
# PLOT 1 : Predicate Distribution (Top 20)
# ---------------------------------------------------------------------------
top_preds = predicates_count.most_common(10)

names = [x[0].split("/")[-1] for x in top_preds]
counts = [x[1] for x in top_preds]

plt.figure(figsize=(10, 5))
plt.bar(names, counts)
plt.xticks(rotation=80)
plt.title("Predicates Distribution (Top 20)")
plt.xlabel("Predicate")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(ANALYSIS_DIR / "Top 20 - Predicate Distribution.png", bbox_inches="tight")
plt.close()


# ---------------------------------------------------------------------------
# PLOT 2 : Node Type Distribution (Top 20)
# ---------------------------------------------------------------------------
type_counter = Counter()

for s, p, o in g:
    # rdf:type relation
    if p == rdf.RDF.type:
        type_counter[str(o)] += 1

# Count Nodes
print("\nNode Type Distribution:\n")

for node_type, count in type_counter.most_common():
    print(count, node_type)

# Prepare data for Ploting
labels = []
counts = []

for node_type, count in type_counter.most_common(20):
    # Short readable label
    short_name = node_type.split("/")[-1]
    short_name = short_name.split("#")[-1]

    labels.append(short_name)
    counts.append(count)

# Bar Graph
plt.figure(figsize=(12, 6))
plt.bar(labels, counts)
plt.title("Node Type Distribution (Top 20)")
plt.xlabel("Node Types")
plt.ylabel("Frequency")
plt.xticks(rotation=80)
plt.tight_layout()
plt.savefig(ANALYSIS_DIR / "Top 20 - Node Types Distribution.png", bbox_inches="tight")
plt.close()


# ---------------------------------------------------------------------------
# PLOT 3 : Research Group Label Distribution
# ---------------------------------------------------------------------------
# Find Correct Predicate Label
predicate_counter = Counter()

for s, p, o in g:
    predicate_counter[str(p)] += 1

for pred, count in predicate_counter.most_common(30):
    print(count, pred)

TARGET_PREDICATE = "http://swrc.ontoware.org/ontology#affiliation"
swrc_name = rdf.term.URIRef("http://swrc.ontoware.org/ontology#name")

# Count Labels
label_counter = Counter()
research_groups = {}

for s, p, o in g:
    if str(p) == TARGET_PREDICATE:
        label_counter[str(o)] += 1
        research_groups[str(o)] = None

print("\nLabel Distribution:\n")

for label, count in label_counter.items():
    print(count, label)

# Get research group names from swrc:name property
for group_uri in research_groups.keys():
    group_ref = rdf.term.URIRef(group_uri)

    # Get the name from swrc:name
    names = list(g.objects(group_ref, swrc_name))
    if names:
        research_groups[group_uri] = str(names[0])
    else:
        # Fallback to URI fragment if no name found
        short_name = group_uri.split("/")[-1]
        short_name = short_name.split("#")[-1]
        research_groups[group_uri] = short_name

# Prepare Data for Visualization
labels = []
counts = []

for group_uri, count in sorted(label_counter.items(), key=lambda x: x[1], reverse=True):
    labels.append(research_groups[group_uri])
    counts.append(count)

# Plot Bar Graph
plt.figure(figsize=(12, 6))
plt.bar(labels, counts)
plt.title("Research Group Label Distribution")
plt.xlabel("Research Groups")
plt.ylabel("Number of Members")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(ANALYSIS_DIR / "Research Group Label Distribution.png", bbox_inches="tight")
plt.close()

print("\n✅ Analysis plots saved to:", ANALYSIS_DIR)
