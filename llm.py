#!/usr/bin/env python

import env
from rich import console 
from langchain_openai.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.prompts import PromptTemplate
from rich.markdown import Markdown
from langchain_community.document_loaders import PyPDFLoader


con = console.Console()

def markdown_response(text): 
    md = Markdown(text) 
    return md 

# Read from File
def read_from_file(filename):
    with open(filename, 'r') as md_file:
        txt = md_file.read() 
        return txt


# Parse PDF 
load_pdf = PyPDFLoader("pdfs/ciw444.pdf")



# Create ChatBot
llm = AzureChatOpenAI(
            model="gpt-4o",
            api_version="2024-02-15-preview",
            temperature=0
        )


# Create PromptTemplate 
context = read_from_file("pdfs/ciw444.md")

question = """
Get the title and Year of publication mentioned in the context
"""
prompt_text = """
Based on the context below, answer the question 

{context}

{question}
"""
create_table_template = PromptTemplate.from_template(prompt_text)



# CREATE CHAINING 
chain = create_table_template | llm 
llm_response = chain.invoke(
        {
            "question": question,
            "context": context
        }
    ) 

# readable 
con.print(markdown_response(llm_response.content))


if __name__ == "__main__":
    print("Langchain Tutorial")
