# LLMKnowledgebase


## Run Qdrant

https://qdrant.tech/documentation/quick_start/

```shell
docker pull qdrant/qdrant


```

```shell
docker run -p 6333:6333 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant

```