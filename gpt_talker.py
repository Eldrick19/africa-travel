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
The following is a friendly conversation between a human and an AI that helps with booking travel to Africa. 
The AI is friendly but concise, and will provide details in bullet points when applicable. 
If the AI does not know the answer to a question, it truthfully says it does not know.
If a question is asked about a topic not pertinent to travelling to Africa, the AI will respond with "Unfortunately, I am focused on helping you book a trip to Africa, so I do not know the answer to that question."

{context}

AI Assistant:"""
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

query = "How can I prepare for my trip to Cote d'Ivoire?"
print(qa.run(query))
# Create the conversational retrieval chain
# chain = ConversationalRetrievalChain.from_llm(
#   llm=ChatOpenAI(model="gpt-3.5-turbo"),
#   retriever=vectorstore.as_retriever(),
#   verbose=True,
#   return_source_documents=True,
# )

# Initialize chat history
# chat_history = []

# # Start the conversation loop
# while True:
#   query = input("Prompt: ") # If no query was passed as a command line argument, prompt the user for input
#   if query in ['quit', 'q', 'exit']: sys.exit() # If the user enters a quit command, exit the program
#   result = chain({"question": query, "chat_history": chat_history}) # Get the response from the conversational retrieval chain
#   print(result['answer']) # Print the response
#   print(result['source_documents'][1]) # Print the source document
#   chat_history.append((query, result['answer'])) # Add the query and response to the chat history
#   query = None # Reset the query variable
