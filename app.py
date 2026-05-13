import json
import os
import sys
import boto3

## use Titan Embedding Model to generate Embedding
from langchain_community.embeddings import BedrockEmbeddings
from langchain_aws import Bedrock

## Data Ingetion
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

## Vector Embedding and Vector store
from langchain_community.vectorstores import FAISS


## LLm Models
from langchain_core.prompts import PromptTemplate
from langchain_community.chains import RetrievalQA

## Bedrock Clients
bedrock = boto3.client(service_name="bedrock_runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

## Data Ingetion
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    
    # - in our testing Character split works better with this PDF data set
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=10000)
    docs = text_splitter.split_documents(documents)
    return docs

## Vector Embedding and vector store
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    vectorstore_faiss.save_local("faiss_index")
    
def get_claude_llm():
    
    ##create the Anthropic Model
    llm = Bedrock(model_id="anthropic.claude-opus-4-5-20251101-v1:0", 
                  client=bedrock, model_kwargs={"maxTokens":32000}
                  )
    return llm

     

