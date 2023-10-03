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

# Now we can override it and set it to "AI Assistant"

template = """
You are a friendly AI that is a virtual travel agent to help with preparing for trips to Africa.
Anytime you list things in your response and would use commas, use bullet points instead. For example

Instead of answering like this: "You will need to provide a valid passport, a valid credit or debit card, flight booking confirmation, accommodation booking or letter of invitation, etc."

Answer like this, when applicable: 
"- You will need to provide a valid passport
- A valid credit or debit card
- Flight booking confirmation
- Accommodation booking or letter of invitation
- etc."

If you do not know, say you do not know. Do not try and guess.
If a visa is not required, say so.
If vaccines are not required, say so.

--- Context ---

{context}

--- Answer ---

"""
prompt = PromptTemplate.from_template(template)
prompt.format(context="Only talk about Africa and travel to Africa.")

# loader = TextLoader("data/data.txt") # Use this line if you only need data.txt
# loader = WebBaseLoader("https://www.handyvisas.com/global/senegal-visa/?from=CA") # Use this line if you need all the files in the data directory
loader = DirectoryLoader("data") # Use this line if you need all the files in the data directory
documents = loader.load()

# 
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings)

chain_type_kwargs = {"prompt": prompt}
qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=vectorstore.as_retriever(), chain_type_kwargs=chain_type_kwargs)

query = "What are my visa requirements for travelling to Senegal as a Canadian?"
print(qa.run(query))
# Create the conversational retrieval chain
# chain = ConversationalRetrievalChain.from_llm(
#   llm=ChatOpenAI(model="gpt-3.5-turbo"),
#   retriever=vectorstore.as_retriever(),
#   verbose=True,
#   return_source_documents=True,
# )

# Initialize chat history
chat_history = []

# Start the conversation loop
while True:
  query = input("Prompt: ") # If no query was passed as a command line argument, prompt the user for input
  if query in ['quit', 'q', 'exit']: sys.exit() # If the user enters a quit command, exit the program
  result = qa.run(query)
  print(result)
  # result = chain({"question": query, "chat_history": chat_history}) # Get the response from the conversational retrieval chain
  # print(result['answer']) # Print the response
  # print(result['source_documents'][1]) # Print the source document
  chat_history.append((query, result)) # Add the query and response to the chat history
  query = None # Reset the query variable
