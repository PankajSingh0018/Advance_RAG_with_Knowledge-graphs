import os
import spacy
import PyPDF2
import numpy as np
import networkx as nx
import faiss
from sentence_transformers import SentenceTransformer, util
import subprocess

# Ensure the spaCy model is installed
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Load the pre-trained model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = "".join(e for e in text if e.isalnum() or e.isspace())
    return text


# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text


# Function to chunk text
def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = [
        " ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)
    ]
    return chunks


# Function to embed text chunks into a vector database
def embed_chunks(chunks, metadata):
    if not chunks:
        print("Warning: No text chunks to embed.")
        return {
            "documents": [],
            "embeddings": np.array([]),
            "metadata": [],
        }

    embeddings = model.encode(chunks)
    return {
        "documents": chunks,
        "embeddings": np.array(embeddings),
        "metadata": metadata,
    }


# Function to index embeddings using FAISS
def index_embeddings(embeddings):
    if embeddings.size == 0:
        print("Warning: No embeddings to index.")
        return None

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


# Function to build a knowledge graph from the indexed embeddings
def build_knowledge_graph(documents, embeddings, metadata, index, threshold=0.7):
    G = nx.Graph()

    # Add nodes
    for i, doc in enumerate(documents):
        G.add_node(i, text=doc, embedding=embeddings[i], metadata=metadata[i])

    # Add edges based on cosine similarity
    for i in range(len(documents)):
        distances, indices = index.search(embeddings[i].reshape(1, -1), len(documents))
        for j in range(1, len(documents)):
            if (
                distances[0][j] < threshold
            ):  # FAISS returns L2 distance; adjust threshold as needed
                G.add_edge(i, indices[0][j], weight=distances[0][j])

    return G


# Function to search in the knowledge graph
def search_knowledge_graph(G, query_embedding, index, top_k=5):
    distances, indices = index.search(query_embedding.reshape(1, -1), top_k)
    top_k_documents = [
        (G.nodes[idx]["text"], 1 - dist, G.nodes[idx]["metadata"])
        for idx, dist in zip(indices[0], distances[0])
        if idx != -1
    ]
    return top_k_documents


# Function to create vector databases for each PDF
def create_vector_database(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        print(f"Warning: No text extracted from {pdf_path}")
        return None, None

    chunks = chunk_text(text)
    metadata = [{"source": pdf_path, "chunk_id": i} for i in range(len(chunks))]
    chunked_embeddings = embed_chunks(chunks, metadata)

    if not chunked_embeddings["documents"]:
        print(f"Warning: No documents to index for {pdf_path}")
        return None, None

    documents = chunked_embeddings["documents"]
    embeddings = chunked_embeddings["embeddings"]
    metadata = chunked_embeddings["metadata"]
    index = index_embeddings(embeddings)

    if index is None:
        print(f"Warning: No index created for {pdf_path}")
        return None, None

    knowledge_graph = build_knowledge_graph(documents, embeddings, metadata, index)
    return knowledge_graph, index


# Paths to individual PDF files
pdf_paths = [
    "D:\\Personal\\My Work and Learnings\\Manish_demo\\Nutrition_RAG\\Food Nutrition Health and fitness.pdf",
    "D:\\Personal\\My Work and Learnings\\Manish_demo\\Nutrition_RAG\\Role-of-Nutrition-and-Wellness-in-Healthy-Schools.pdf",
    "D:\\Personal\\My Work and Learnings\\Manish_demo\\Nutrition_RAG\\The Role of Nutrition in Health and Wellness.pdf",
    "D:\\Personal\\My Work and Learnings\\Manish_demo\\Nutrition_RAG\\The Pathway to Health.pdf",
    "D:\\Personal\\My Work and Learnings\\Manish_demo\\Nutrition_RAG\\The Pathway to Health.pdf",
]

# Create vector databases for each PDF
knowledge_graphs = []
indices = []

for pdf_path in pdf_paths:
    kg, index = create_vector_database(pdf_path)
    if kg is not None and index is not None:
        knowledge_graphs.append(kg)
        indices.append(index)


# Unified search function
def unified_search(query, top_k=3):
    query_embedding = model.encode(preprocess_text(query))
    results = []
    for kg, index in zip(knowledge_graphs, indices):
        results.extend(search_knowledge_graph(kg, query_embedding, index, top_k))

    # Sort results by score
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


# Example query
query = str(input("Enter your query : "))
final_results = unified_search(query)

# Save results to a text file
with open("results.txt", "w", encoding="utf-8") as f:
    for result in final_results:
        f.write(f"Document: {result[0]}\nScore: {result[1]}\nMetadata: {result[2]}\n\n")

# Display results
for result in final_results:
    print(f"Document: {result[0]}, Score: {result[1]}, Metadata: {result[2]}")
