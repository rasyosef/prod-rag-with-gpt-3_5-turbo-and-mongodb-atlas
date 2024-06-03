import os
import gradio as gr
import openai
import pymongo
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.core import VectorStoreIndex
from llama_index.llms.openai import OpenAI

MONGO_URI = os.environ["MONGO_URI"]

# MongoDB Atlas Vector Store
mongodb_client = pymongo.MongoClient(MONGO_URI)
store = MongoDBAtlasVectorSearch(
    mongodb_client=mongodb_client,
    db_name="oppenheimer",
    collection_name="oppenheimer_wiki_chunks",
    index_name="vector_index",
    embedding_key="embedding",
)


# Check if the OPENAI API KEY is valid
def is_valid_openai_api_key(api_key):
    client = openai.OpenAI(api_key=api_key)
    try:
        client.models.list()
    except openai.AuthenticationError:
        return False
    else:
        return True


# Create a query engine using the provided api key
def prepare_query_engine(api_key):
    # OpenAI Embeddings
    embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        embed_batch_size=16,
        api_key=api_key,
        max_retries=2,
    )

    # Loading Index
    index_loaded = VectorStoreIndex.from_vector_store(
        vector_store=store, embed_model=embed_model
    )

    # GPT 3.5 Turbo
    llm = OpenAI(
        model="gpt-3.5-turbo-0125", temperature=0, max_tokens=512, api_key=api_key
    )

    # Query Engine
    query_engine = index_loaded.as_query_engine(
        llm=llm, streaming=True, similarity_top_k=3
    )

    return query_engine


# Generates response using the query engine
def generate(query, api_key):
    if api_key.strip() == "" or not is_valid_openai_api_key(api_key):
        yield "Please enter a valid openai api key"
    else:
        query_engine = prepare_query_engine(api_key)
        response = ""
        try:
            streaming_response = query_engine.query(query)
            for token in streaming_response.response_gen:
                response += token
                yield response
        except openai.RateLimitError as rl:
            yield "RateLimitError - " + str(rl)
        except Exception as e:
            yield str(e)


with gr.Blocks() as demo:
    gr.Markdown(
        """
  # Retrieval Augmented Generation with GPT 3.5 Turbo, MongoDB Atlas Vector Search, and LlamaIndex: Question Answering demo
  ### This demo uses the GPT 3.5 Turbo LLM and MongoDB Atlas Vector Search for fast and performant Retrieval Augmented Generation (RAG).
  ### The context is the new Oppenheimer movie's entire wikipedia page. The movie came out very recently in July, 2023, so the GPT 3.5 turbo model is not aware of it.
  Retrieval Augmented Generation (RAG) enables us to retrieve just the few small chunks of the document that are relevant to the our query and inject it into our prompt.
  The model is then able to answer questions by incorporating knowledge from the newly provided document. RAG can be used with thousands of documents, but this demo is limited to just one txt file.
  """
    )
    OPENAI_API_KEY = gr.Textbox(
        label="OPENAI_API_KEY",
        placeholder="Enter your OPENAI_API_KEY",
        lines=1,
        type="password",
    )
    gr.Markdown("## Enter your question")
    with gr.Row():
        with gr.Column():
            ques = gr.Textbox(label="Question", placeholder="Enter text here", lines=2)
        with gr.Column():
            ans = gr.Textbox(label="Answer", lines=4, interactive=False)
    with gr.Row():
        with gr.Column():
            btn = gr.Button("Submit")
        with gr.Column():
            clear = gr.ClearButton([ques, ans])

    btn.click(fn=generate, inputs=[ques, OPENAI_API_KEY], outputs=[ans])
    examples = gr.Examples(
        examples=[
            "Who portrayed J. Robert Oppenheimer in the new Oppenheimer movie?",
            "In the plot of the movie, why did Lewis Strauss resent Robert Oppenheimer?",
            "What happened while Oppenheimer was a student at the University of Cambridge?",
            "How much money did the Oppenheimer movie make at the US and global box office?",
            "What score did the Oppenheimer movie get on Rotten Tomatoes and Metacritic?",
        ],
        inputs=[ques],
    )

demo.queue().launch(debug=True)
