## Simple demo of the RAG(Retrieval Augmented Generation)

1. Use [Bert-based Model](https://huggingface.co/ckiplab/bert-base-chinese) developed by ckiplab to extract text embeddings.
2. Use ResNet50 or EfficientNet_B7 to get the image embeddings.
3. Create vector database through Milvus to store embeddings including texts and images embeddings.
4. Use similarity search to find the similar images or QA texts.

### How to run it
- At first, download docker compose yaml file 
    ```bash
    wget https://github.com/milvus-io/milvus/releases/download/v2.3.2/milvus-standalone-docker-compose.yml -O docker-compose.yml
    ```
- To run Milvus, execute the command below
    ```bash
    sudo docker-compose up -d
    ```
- Check if all the containers are activated. There will be three docker containers running(milvus-etcd, milvus-minio, milvus-standalone)
    ```bash
    sudo docker compose ps
    ```
- Download the demo data found on the Internet(include texts & images)
    ```bash
    mkdir data
    cd data
    ## Get text data
    wget https://github.com/A-baoYang/finetune-with-openai/raw/main/example_data/faq.jsonl
    
    ## Get image data (manual download from the web-site or use the kaggle API)
    kaggle datasets download -d jehanbhathena/weather-dataset
    unzip archive.zip
    ```
- Run create_img_db.py to create the collection for images
    ```bash 
    python create_img_db.py
    ```
- Run create_QA_db.py to create the collection for texts
    ```bash
    python create_QA_db.py
    ```
- Run img_search.py or QA_text_search.py to do the similarity search


Ref.
- [Milvus](https://milvus.io/docs/overview.md)
- [faq jsonl data](https://github.com/A-baoYang/finetune-with-openai)
- [Bert-based model from ckiplab](https://huggingface.co/ckiplab/bert-base-chinese)
- [weather image recognition](https://www.kaggle.com/datasets/jehanbhathena/weather-dataset/)
- [Towhee - reverse image search](https://github.com/towhee-io/examples/blob/main/image/reverse_image_search/1_build_image_search_engine.ipynb?fbclid=IwAR3lM0DBT8fD0IPWj3Gpz1TP1L9YyYPLkZ0GR0qJT5aniWHh10AHPv5Iwz0)

