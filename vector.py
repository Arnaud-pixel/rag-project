from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import pandas as pd

#definition of the path with the documents that need to be used by the RAG
dir= "docs/"

#loading the PDF document from the directory
print("Loading documents:")
loader=DirectoryLoader(dir,loader_cls=PyPDFLoader,use_multithreading=True,max_concurrency=128,show_progress=True,silent_errors=True)
documents=loader.load()

#defining how the document text should be split
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n","\n",".",","]
)

#splitting the document into chunks and storing it in docs
docs = text_splitter.split_documents(documents)

#defining the location of the db to store the document in vectorized format
db_location = "./chroma_langchain_db"

#definig the embedding model to be used to process the document vectorization
embeddings = OllamaEmbeddings(model="nomic-embed-text")

#checking if the databased was already created (False if it exist, Tru if it doesn't)
add_documents = not os.path.exists(db_location)

#initialization of the database access
vector_store = Chroma (
    persist_directory=db_location,
    embedding_function=embeddings
)

#If the database doesn't exist, add the document in the database
if add_documents:
    vector_store.add_documents(documents=docs)

#Define the access to the vectorized database and the search parameters
#More details on the parameters here: https://api.python.langchain.com/en/latest/vectorstores/langchain_core.vectorstores.VectorStore.html
retriever = vector_store.as_retriever(
    search_kwargs={"k":3}
)