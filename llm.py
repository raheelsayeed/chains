#!/usr/bin/env python

import env
from rich import console 
from langchain_openai.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.prompts import PromptTemplate
from rich.markdown import Markdown

con = console.Console()

def markdown_response(text): 
    md = Markdown(text) 
    return md 

# Create ChatBot
llm = AzureChatOpenAI(
            model="gpt-4o",
            api_version="2024-02-15-preview",
            temperature=0
        )


# Create PromptTemplate 
context = """
LDL: 145 mg/dL 
HDL: 55 mg/dL 
"""
question = """
Make a table with data in the above context
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
