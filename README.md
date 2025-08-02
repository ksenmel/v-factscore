# v-factscore
Evaluates LLM outputs by extracting atomic facts, identifying key entities, and calculating the percentage of facts supported by the provided database.



Compared to the original [FActScore](https://github.com/shmsw25/FActScore), this enhanced pipeline quantifies LLM truthfulness by:
1. Extracting atomic facts from generations;
2. Retrieving supporting entities (NER);
3. Verifying facts with a provided knowledge source;

This version also provides significant improvements:
1. Boosted performance through asynchronous API queries; 
2. Boosted accuracy via Named Entity Recognition (NER) integration; 
3. More reliable document retrieval using a sharded FAISS vector index that matches titles by semantic similarity rather than character-level comparison; 
4. Automatic topic extraction;

## Prerequisites
1. **Knowledge source**.
A reference database in the specified format Ensure the table has two columns: title, text.
You can use pre-built .db Wikipedia 2023/04/01 dump, download it directly from [here](https://drive.google.com/file/d/1mekls6OGOKLmt7gYtHs0WGf5oTamTNat/view?usp=sharing).
2. **Embeddings**. Vector representations of knowledge source titles (article titles). Pre-computed embeddings from the Wikipedia 2023/04/01 dump, generated using the `sentence-transformers/all-mpnet-base-v2` model, are available [here](https://drive.google.com/file/d/15NioK8CzUYMeFpe9kynSxl5UT7OPcZvh/view?usp=sharing).
3. **Trained FAISS Index**. A trained **FAISS IVF Index** using the embeddings above. This must be trained on the same embeddings to ensure compatibility and optimal retrieval performance. If the trained index is too large (>5GB), it may not fit in RAM. See [factscore/create_index.py](https://github.com/ksenmel/v-factscore/blob/main/factscore/create_index.py) about handling this
4. **API Configuration**. As this implementation uses model APIs, you must set base URLs and API keys in their corresponding environment variables before execution.
```bash
export EMBEDDINGS_API_KEY="key-for-embeddings"
export COMPLETIONS_API_KEY="key-for-completions"

export EMBEDDINGS_BASE_URL="https://embeddings-api.url"
export COMPLETIONS_BASE_URL="https://completions-api.url"
```

## Run
Make a new Python 3.11+ environment `conda`

1. Install the requirements
```shell
cd v-factscore
pip install -r requirements.txt
```

2. Initialize the factscore instance
```python
from factscore.factscorer import FactScorer

fs = FactScorer()
```
3. Use the knowledge source database:
```python
fs.register_knowledge_source(faiss_index="path/to/index",
                             data_db="path/to/database", 
                             table_name="tablename")
```
4. Score generations
```python
res = fs.get_score(generations=[generation1, generation2], k=1)
```

See see [demo.ipynb](https://github.com/ksenmel/v-factscore/blob/main/demo.ipynb) for more details.

## License
This project is licensed under the MIT License — see the [LICENSE](https://github.com/ksenmel/v-factscore/blob/main/LICENSE) file for details.