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
set_debug(False)  # Enables debugging
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableLambda


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

#specify the type of splitter to split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,chunk_overlap =0,add_start_index=True
)

#now split the text with the splitter
text = text_splitter.split_documents(loader) 
# print(text)


#embedding has different model,#specify the embedding model to use
embedding = OpenAIEmbeddings(model="text-embedding-3-large")

#choose a vector store and put the embedding
vectorstore = Chroma.from_documents(text, embedding) # print(vectorstore)

#retrieve the embedding from vector store
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})

# Define retriever function using RunnableLambda
retriever_chain = RunnableLambda(lambda query: retriever.invoke(query))


#use chat model
model = init_chat_model("gpt-4o-mini", model_provider="openai")

rag_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="you are an assistant to provide response"),
        MessagesPlaceholder(variable_name="retrieved_context"),
        HumanMessage(content="{user_query}")
]
    
)
while True:
    query = input("Ask me a question based on the PDF (or type 'exit' to quit): \n USER: ")
    
    
    if query == 'quit':
       print("goodbye!")
       break
     

     # Retrieve relevant document
    retrieved_docs = retriever_chain.invoke(query)
    retrieved_text = "\n".join([doc.page_content for doc in retrieved_docs])

    #format the model to get response
    formatted= rag_prompt.format(retrieved_context=[retrieved_text], user_query=query)
    result = model.invoke(formatted)

    print(result.content)

