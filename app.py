import os
import getpass
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.runnables import chain
from typing import List
from langchain.globals import set_debug, get_debug
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
set_debug(True)  # Enables debugging

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: OPENAI_API_KEY")
  
# load_dotenv function is used to call the content in env which are secret keys
load_dotenv()

#indicate the file path
file_path = "./Dresume.pdf"
# file_path ="../example_data/nke-10k-2023.pdf"

#load the file to the pdf loader
pdf_loader = PyPDFLoader(file_path)

#load the pdf loader to load
loader = pdf_loader.load()
# print(loader)
#splitting the text into chunks
#customised the type of text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,chunk_overlap = 0,add_start_index=True
)

#now split the text with the splitter
# text = text_splitter.split_text(loader)
text = text_splitter.split_documents(loader) 
# print(text)

#embedding
#embedding has different model,
#specify the embedding model to use
embedding = OpenAIEmbeddings(model="text-embedding-3-large")

#after choosing the model of embeddings now set the embedding query
vector1 = embedding.embed_query(text[0].page_content)
vector2 = embedding.embed_query(text[1].page_content)
len(vector1) == (vector2)

#choose and import the vector store
vector_store = Chroma(embedding_function=embedding)


#now pass the embedded text to vector store
saved_vector = vector_store.add_documents(documents=text)
# print(saved_vector)

#set retrivers to retrive query # Define retriever with the @chain decorator
@chain
def retriever(query: str) -> list[Document]:
    result = vector_store.similarity_search(query, k=1)
    return result

#an input to ask a query
query = input("Ask me a question based on the PDF (or type 'exit' to quit): \n USER: ")
  
def chat():
     while True:
        # User input for the question
        query = input()
       
        # If the user types 'exit', break the loop
        if query == 'exit':
            print("Exiting the program.")
            break


#use chat model
model = init_chat_model("gpt-4o-mini", model_provider="openai")
messages =[
    SystemMessage("you are answering the question based on the content provided"),
    HumanMessage(content=query)
]

result = model.invoke(messages)
chat()
print(result)
