import sys

from torchvision.io import read_image
from function import ImageExtractor

import glob

import matplotlib.pyplot as plt
import matplotlib.image as matplot_img

## vector db
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)


img_path = 'data/weather_dataset/snow/1440.jpg'
imgs = [read_image(img_path)]

print(imgs[0].shape)

kind = 'efficientnet_b7'
img_extractor = ImageExtractor(kind=kind)
query_img_emb = img_extractor.get_emb(imgs)

## vector db
COLLECTION_NAME = "TEST_IMG_DB_EFF_B7"

connections.connect("default", host="localhost", port="19530")

if not utility.has_collection(COLLECTION_NAME):
    print(f"no {COLLECTION_NAME} collection. EXIT")
    sys.exit(1)

coll = Collection(name=COLLECTION_NAME)
coll.load()
utility.load_state(COLLECTION_NAME)

print('Start to search...')
print('=================================')

search_res = coll.search(query_img_emb, anns_field="img_embs",
                               param={}, output_fields=['img_path'], limit=3)

print("Query Img Path:", img_path)

plt.figure(figsize=(15, 12))
plt.subplot(2, 3, 2)
plt.imshow(matplot_img.imread(img_path))
plt.title(f'Query Image:\n{img_path}', fontsize=8)
plt.axis('off')

for hits in search_res:
    for index, hit in enumerate(hits):
        img_path = hit.entity.get('img_path')
        id_ = hit.id
        distance = hit.distance

        plt.subplot(2, 3, index+4)
        plt.imshow(matplot_img.imread(img_path))
        plt.title(f'distance:{distance:.2f}\nimg_path:{img_path}', loc='left', fontsize=8)
        plt.axis('off')
        print('=================================')
        print('id:', id_)
        print('distance:', distance)
        print('img_path:', img_path)

#plt.show()
plt.savefig('eff_b7_test.png', bbox_inches='tight')
