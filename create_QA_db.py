import json
import torch
import sys

from function import LM
## vector db
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

## load dataset
with open('data/faq.jsonl', 'r') as f:
    data = [json.loads(i.rstrip('\n')) for i in f.readlines()]

questions = [i['prompt'] for i in data]
answers = [i['completion'] for i in data]

max_len = max([len(i) for i in questions])
max_len = 100

print("Load pre-trained BERT model")
llm_emb = LM()

print("Get question embeddings")
q_embs = llm_emb.get_embs(questions)

## connect to Milvus Database
COLLECTION_NAME = "TEST_LLM_DB"

print("Connect to db")
connections.connect("default", host="localhost", prot="19530")
if utility.has_collection(COLLECTION_NAME):
    utility.drop_collection(COLLECTION_NAME)

## create collection, id, question, q_embs, answer
fields = [
    FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name='questions', dtype=DataType.VARCHAR, max_length=5000),
    FieldSchema(name='answers', dtype=DataType.VARCHAR, max_length=5000),
    FieldSchema(name='q_embs', dtype=DataType.FLOAT_VECTOR, dim=768),
]

print("Create collections")
schema = CollectionSchema(fields=fields)
collection = Collection(name=COLLECTION_NAME, schema=schema)

## Insert data
print("Create insertable data")
insert_list = [
    questions,
    [x[:995]+'...' if len(x) > 999 else x for x in answers],
    q_embs
]

print("Insert data")
collection.insert(insert_list)
collection.flush() ##

# Create an IVF_FLAT index for collection.
print('Create index')
index_params = {
    'metric_type':'L2',
    'index_type':"IVF_FLAT",
    'params':{"nlist":1536}
}

collection.create_index(field_name="q_embs", index_params=index_params)
collection.load()

## perform similarity search
query_strings = ["我想要線上開戶該怎麼做?"]
query_embs = llm_emb.get_embs(query_strings)

print('Search')
search_res = collection.search(query_embs, anns_field="q_embs",
                                param={}, output_fields=['questions', 'answers'], limit=3)

print("query question:", query_strings[0])
for hits in search_res:
    for hit in hits:
        print('id:', hit.id)
        print('distance:', hit.distance)
        print('question:', hit.entity.get('questions'))
        print('answer:', hit.entity.get('answers'))

