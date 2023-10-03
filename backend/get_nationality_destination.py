# Import necessary libraries
import os
import sys
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.document_loaders import DirectoryLoader, WebBaseLoader
from langchain.prompts import PromptTemplate

import constants

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = constants.APIKEY

def get_nationality_destination(query):
    # Set OpenAI engine
    template = """
    You are an AI tool that will only respond to a particular format or with an error message.
    Text will be passed below from a user, that should have information about their nationality and destination.
    Your response will output their nationality and destination in a python dictionary format:
    - The output must be a Python dictionary
    - The nationality must be the country double-character identification (e.g. "CA" for "Canada")
    - The destination will be output in lower case (e.g. "ivory coast")

    If you are not able to get this information, you should output an error message.

    --- Context ---

    {context}

    --- Answer ---

    """
    prompt = PromptTemplate.from_template(template)
    prompt.format(context="Only talk about Africa and travel to Africa.")

    #loader = TextLoader("data/data.txt") # Use this line if you only need data.txt
    loader = DirectoryLoader("data") # Use this line if you need all the files in the data directory
    documents = loader.load()

    # 
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents, embeddings)

    chain_type_kwargs = {"prompt": prompt}
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=vectorstore.as_retriever(), chain_type_kwargs=chain_type_kwargs)
    result = qa.run(query)
    result_dict = json.loads(result)
    print(result_dict)
    return result_dict