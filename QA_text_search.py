from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection
)

from function import LM

COLLECTION_NAME = "TEST_LLM_DB"

query = input("問題?")

llm_emb = LM()
query_embs = llm_emb.get_embs([query])

connections.connect("default", host="localhost", prot="19530")

if not utility.has_collection(COLLECTION_NAME):
    print(f"no {COLLECTION_NAME} collection. EXIT")
    sys.exit(1)

coll = Collection(name=COLLECTION_NAME)
coll.load()
utility.load_state(COLLECTION_NAME)


print('Start to search...')
print('=================================')

search_res = coll.search(query_embs, anns_field="q_embs",
                               param={}, output_fields=['questions', 'answers'], limit=3)

print("Query:", query)

for hits in search_res:
    for hit in hits:
        print('=================================')
        print('id:', hit.id)
        print('distance:', hit.distance)
        print('question:', hit.entity.get('questions'))
        print('answer:', hit.entity.get('answers'))
