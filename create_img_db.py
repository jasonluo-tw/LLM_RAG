import sys

from torchvision.io import read_image
from function import ImageExtractor

import glob

## vector db
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

dirs = glob.glob('data/weather_dataset/*')

all_img_files = []
for dd in dirs:
    files = glob.glob(f'{dd}/*.jpg')[:20]
    all_img_files.extend(files)

imgs = [read_image(i) for i in all_img_files]

kind = 'efficientnet_b7'  # or 'resnet50'
img_extractor = ImageExtractor(kind=kind)
img_embs = img_extractor.get_emb(imgs)

emb_dim = img_embs.shape[1]

## connect to Milvus Database
# TEST_IMG_DB_EFF_B7
COLLECTION_NAME = f"TEST_IMG_DB-{kind}"

print("Connect to db")
connections.connect("default", host="localhost", prot="19530")
if utility.has_collection(COLLECTION_NAME):
    utility.drop_collection(COLLECTION_NAME)

fields = [
    FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name='img_path', dtype=DataType.VARCHAR, max_length=500),
    FieldSchema(name='img_embs', dtype=DataType.FLOAT_VECTOR, dim=emb_dim),
]

print("Create collections")
schema = CollectionSchema(fields=fields, description='reverse image search')
collection = Collection(name=COLLECTION_NAME, schema=schema)

print('Create Index')
index_params = {
    'metric_type': 'L2',
    'index_type': 'IVF_FLAT',
    'params': {"nlist": emb_dim}
}
collection.create_index(field_name='img_embs', index_params=index_params)

print('Insert Data')
insert_list = [
    all_img_files,
    img_embs
]

collection.insert(insert_list)
collection.flush()
