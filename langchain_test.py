import openai
import os
import requests
import json

from langchain.document_loaders import UnstructuredFileLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import MarkdownTextSplitter

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")

from langchain.llms import OpenAI



llm = OpenAI(temperature=0.9)

loader = UnstructuredFileLoader("inter_grids_full.md")
docs = loader.load()
# print(docs[0].page_content[:400])

splitter = MarkdownTextSplitter()
docs = splitter.split_documents(docs)

chain = load_summarize_chain(llm, chain_type="map_reduce")
chain.run(docs)



# text = "What would be a good company name for a company that makes colorful socks?"
# print(llm(text))