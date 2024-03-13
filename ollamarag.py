from langchain_community.document_loaders import DirectoryLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
import os
import shutil

DATAPATH = "data"
CHROMAPATH = "chroma"
PROMPT_TEMPLATE = """
Answer the question based on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# subject to change depending on size of doc
size_of_chunk = 1000
size_of_overlap = 500

model = Ollama(model="mistral")

def load_documents():
    loader = DirectoryLoader(DATAPATH, glob="*.txt")
    documents = loader.load()
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = size_of_chunk,
        chunk_overlap = size_of_overlap,
        length_function = len,
        add_start_index = True
    )
    chunks = text_splitter.split_documents(documents)

    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    doc = chunks[10]
    print(doc.metadata)
    print(doc.page_content)

    return chunks

def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMAPATH):
        shutil.rmtree(CHROMAPATH)
    
    db = Chroma.from_documents(
        chunks, OllamaEmbeddings(model = "mistral"), persist_directory=CHROMAPATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMAPATH}.")

def generate_database():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def main():
    generate_database()
    embedding_function = OllamaEmbeddings(model = "mistral")
    vectordb = Chroma(persist_directory=CHROMAPATH, embedding_function=embedding_function)

    query_text = input("(\'quit\' to exit)Query Text: ")
    while True:
        if(query_text=="quit"):
            break
        else:
            query_embeddings = embedding_function.embed_query(query_text)
            results = vectordb.similarity_search_by_vector_with_relevance_scores(query_embeddings, k = 5)
            if(len(results)==0 or results[0][1]<0.7):
                print("No results found.")
            context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            prompt = prompt_template.format(context = context_text, question = query_text)
            response = model.invoke(prompt)
            response2 = model.invoke(query_text)
        
        print("Without context: " + response2)
        print("With context: " + response)
        query_text = input("(\'quit\' to exit)Query Text: ")


if __name__ == "__main__":
    main()

