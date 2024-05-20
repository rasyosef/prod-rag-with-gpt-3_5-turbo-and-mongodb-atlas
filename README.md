# Retrieval Augmented Generation with GPT 3.5 Turbo and MongoDB Atlas Vector Search: Question Answering demo

### This demo uses the GPT 3.5 Turbo LLM and MongoDB Atlas Vector Search for fast and performant Retrieval Augmented Generation (RAG).

The context is the new Oppenheimer movie's entire wikipedia page. The movie came out very recently in July, 2023, so the GPT 3.5 turbo model is not aware of it.

Retrieval Augmented Generation (RAG) enables us to retrieve just the few small chunks of the document that are relevant to the our query and inject it into our prompt. The model is then able to answer questions by incorporating knowledge from the newly provided document. RAG can be used with thousands of documents, but this demo is limited to just one txt file.