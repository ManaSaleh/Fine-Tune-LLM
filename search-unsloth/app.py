import gradio as gr
from unsloth import FastLanguageModel
import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from datasets import load_from_disk
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware

# Initialize FastAPI app
app = FastAPI()

@app.get("/")
def greet_json():
    return {"Hello": "World!"}

# Define the Search class for the RAG functionality
class Search:
    def __init__(self):
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(model_name="ManaSaleh/unsloth_llama_3.1_lora_model")
        self.model = FastLanguageModel.for_inference(self.model)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.testdata = load_from_disk('test_with_embeddings')
        self.client = QdrantClient(path="qdrant")
        self.embedding = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

    def search(self, query):
        with torch.no_grad():
            query_embedding = self.embedding.encode([query], convert_to_tensor=True)

        results = self.client.search(
            collection_name="my_collection",
            query_vector=query_embedding[0].tolist(),
            limit=1
        )

        if not results:
            return None

        relevant_contexts = [(result.payload["context"], result.payload["question"]) for result in results]

        return relevant_contexts

    def rag(self, query):
        rag_dataset_prompt = "Context: {0}\n\nQuestion: {1}\n\nAnswer:"
        top_results = self.search(query)

        if not top_results:
            return None

        context, question = top_results[0]

        inputs = self.tokenizer(
            [rag_dataset_prompt.format(context, query)],
            return_tensors="pt"
        ).to(self.device)

        generated_answer = self.model.generate(**inputs, max_new_tokens=4096)
        answer = self.tokenizer.decode(generated_answer[0], skip_special_tokens=True).strip()

        return answer

# Instantiate the Search class
searcher = Search()

# Define the Gradio function
def generate_answer(query):
    return searcher.rag(query)

# Create the Gradio interface
iface = gr.Interface(fn=generate_answer, inputs="text", outputs="text", title="Search and RAG")

# Mount the Gradio interface to the FastAPI app
app.mount("/gradio", WSGIMiddleware(iface))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
