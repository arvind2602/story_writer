import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain,SequentialChain 

os.environ['OPENAI_API_KEY'] = apikey

# App Framework
st.title(" Story GPT")
prompt=st.text_input("Enter your prompt here")

# Template
title_template = PromptTemplate(
    input_variables=["topic"],
    template='Give me a good story title on {topic}.',
)

script_template = PromptTemplate(
    input_variables=["title"],
    template='Give me a real story on Title: {title}.',
)
# Llms
llm=OpenAI(temperature=0.9)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True ,output_key="title")
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key="script")
sequential_chain=SequentialChain(chains=[title_chain,script_chain],input_variables=['topic'],output_variables=['title','script'],verbose=True)

#for Showing the response   
if prompt:
    response=sequential_chain.run({"topic": prompt})
    st.write(response['title'])
    st.write(response['script'])