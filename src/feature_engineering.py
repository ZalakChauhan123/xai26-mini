# IMPORTS
import rdflib as rdf
from rdflib import Graph
from rdflib import Literal
import torch
from collections import Counter


# LOAD GRAPH DATASET
g = Graph()
g.parse("dataset/aifbfixed_complete.n3")

print("Graph loaded successfully!")



# ---------------------------------
# NODE
# ---------------------------------


# Node - Create Node IDs
all_nodes = set()

for s,p,o in g:

    # subject is a node
    all_nodes.add(s)
    # Only add Object if not literal
    if not isinstance(o, Literal):
        all_nodes.add(o)

print("Total unique nodes", len(all_nodes))

# Create Node-to-ID Mapping
node_to_id = {}

for idx, node in enumerate(all_nodes):
    node_to_id[node] = idx

# Create Reverse Mapping
id_to_node = {}

for node, idx in node_to_id.items():
    id_to_node[idx] = node


# Test the Mapping
# print("\nSample Node IDs:\n")
for i, (node, idx) in enumerate(node_to_id.items()):
    # print(idx, node)

    if i >= 3:
        break



# ----------------
# EDGES
# ----------------

# Edge - Create Predicate IDs
predicate_to_id = {}
relation_id = 0

for s, p, o in g:

    # Skip Literals
    if isinstance(o, Literal):
        continue

    if p not in predicate_to_id:
        predicate_to_id[p] = relation_id
        relation_id += 1

print("Total Relation Types :- ", len(predicate_to_id))

# Reverse Mapping
id_to_relation = {idx: rel for rel, idx in predicate_to_id.items()}




# Edge - RDF Triple into Edge IDs & Edge Types

# RDF triple into Edge Ids
REMOVE_RELATIONS = [
    "http://swrc.ontoware.org/ontology#affiliation",
    "http://swrc.ontoware.org/ontology#employs"
]

edges = []
edge_types = []

for s, p, o in g:

    # Skip Literals
    if isinstance(o, Literal):
        continue

    # Remove Leakage Relations
    if(str(p) in REMOVE_RELATIONS):
        continue

    source = node_to_id[s]
    target = node_to_id[o]

    relation_id = predicate_to_id[p]

    edges.append((source, target))
    edge_types.append(relation_id)

# print(edges[0:10])
# print(edge_types)


# Seperate Edges 
edge_ids = []
for s, p, o in g:
    if isinstance(o, Literal):
        continue
    source = node_to_id[s]
    target = node_to_id[o]
    edge_id = len(edge_ids)
    edge_ids.append((edge_id, source, target, p))

# print(edge_ids[0:10])



# Edge - Create Edge_index & Edge Type Tensors
# Edge Index
edge_index = torch.tensor(
    edges,
    dtype = torch.long
).t().contiguous()

# print("Edge Index Shape : ", edge_index.shape)
# print("edge_index : ",edge_index[ :, :10 ])


# Convert Edge Type into Tensors
edge_type = torch.tensor(
    edge_types,
    dtype = torch.long
)

# print("Edge Type Shape : ", edge_type.shape)
# print("edge_type : ", edge_type)




# --------------------
# LABELS
# --------------------

LABEL_PREDICATE = "http://swrc.ontoware.org/ontology#affiliation"

# Collect unique Labels
unique_labels = set()

for s, p, o in g:
    if str(p) == LABEL_PREDICATE:
        unique_labels.add(o)

# print("Classes :- ", unique_labels)
# print("Total Classes :- ", len(unique_labels))



# Create label IDs
label_to_id = {

    label : idx
    for idx, label in enumerate(unique_labels)
}
# print("Label IDs :- ",label_to_id)



# Label - Create Node Labels
labels = {}

for s, p, o in g:

    if str(p) == LABEL_PREDICATE:

        node_id = node_to_id[s]
        class_id = label_to_id[o]
        labels[node_id] = class_id

# print(node_id)
# print(class_id)
# print(labels)

# Label - Convert Labels to Tensors
num_nodes = len(node_to_id)

y = torch.full(
    (num_nodes,),
    -1,
    dtype = torch.long
)

for node_id, class_id in labels.items():
    y[node_id] = class_id

# print(y.shape)
# print(y[:20])
print(torch.unique(y))