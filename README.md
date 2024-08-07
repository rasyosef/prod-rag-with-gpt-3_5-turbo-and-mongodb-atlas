# Retrieval Augmented Generation with GPT 3.5 Turbo, MongoDB Atlas Vector Search, and LlamaIndex: Question Answering demo

### This demo uses the GPT 3.5 Turbo LLM and MongoDB Atlas Vector Search for fast and performant Retrieval Augmented Generation (RAG).

The context is the new Oppenheimer movie's entire wikipedia page. The movie came out very recently in July, 2023, so the GPT 3.5 turbo model is not aware of it.

Retrieval Augmented Generation (RAG) enables us to retrieve just the few small chunks of the document that are relevant to the our query and inject it into our prompt. The model is then able to answer questions by incorporating knowledge from the newly provided document. RAG can be used with thousands of documents, but this demo is limited to just one txt file.

## RAG Components
- **LLM** : `GPT 3.5 Turbo`
- **Text Embedding Model** : `OpenAI Embeddings (text-embedding-3-small)`
- **Vector Database** :  `MongoDB Atlas Vector Search`
- **Framework** : `LlamaIndex`

## Demo
The demo (built with `gradio`) has been depolyed to the following HuggingFace space.

https://huggingface.co/spaces/rasyosef/RAG-with-GPT-3.5-MongoDBAtlas-Llamaindex

## Code
- [Building Production grade RAG with GPT 3_5-turbo and MongoDB Atlas.ipynb](https://github.com/rasyosef/prod-rag-with-gpt-3_5-turbo-and-mongodb-atlas/blob/main/Building%20Production%20grade%20RAG%20with%20GPT%203_5-turbo%20and%20MongoDB%20Atlas.ipynb)
    - Chunking the Document into short passages (Nodes)
    - Creating and Embedding for each Node (Chunk) using OpenAI's `text-embedding-3-small` model
    - Pushing the embedding vectors to `MongoDB Atlas` along with matadata
    - Creating a Vector Search Index
- [app.py](https://github.com/rasyosef/prod-rag-with-gpt-3_5-turbo-and-mongodb-atlas/blob/main/app.py)
    - A friendly web interface created with `gradio`
    - Connects with the MongoDB Atlas Vector Store
    - Creates a query engine using the Vector Store and OpenAI's `gpt-3.5-turbo-0125` LLM 