## Design and Implementation of LangChain Expression Language (LCEL) Expressions
#### Name: Malar Mariam S
#### Register Number: 212223230118

### AIM:
To design and implement a LangChain Expression Language (LCEL) expression that utilizes at least two prompt parameters and three key components (prompt, model, and output parser), and to evaluate its functionality by analyzing relevant examples of its application in real-world scenarios.

### PROBLEM STATEMENT:
Design an LCEL pipeline using LangChain with at least two dynamic prompt parameters. Integrate prompt, model, and output parser components to form a complete expression. Evaluate its functionality through real-world query-response scenarios.


### DESIGN STEPS:


#### STEP 1:
Setup API and Environment: Load environment variables using dotenv and set openai.api_key from the local environment.

#### STEP 2:
 Create Prompt and Model: Use LangChain to define a ChatPromptTemplate and initialize ChatOpenAI for text generation.

#### STEP 3:
Build a Retrieval System: Store predefined texts in DocArrayInMemorySearch with OpenAIEmbeddings and create a retriever.

#### STEP 4:
 Define Question-Answering Chain: Use RunnableMap to fetch relevant documents and pass them to a chat model for responses.

#### STEP 5:
Invoke the Chain: Run chain.invoke() with a question to retrieve context-based answers using the LangChain pipeline.

### PROGRAM:t

SIMPLE CHAIN
```py
import os
import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser

model = ChatOpenAI()
output_parser = StrOutputParser()

simple_prompt = ChatPromptTemplate.from_template(
    "Tell me a short joke about {topic}"
)

simple_chain = simple_prompt | model | output_parser

print("------ Simple Chain ------")
response = simple_chain.invoke({"topic": "Cinderella"})
print(response)
```

COMPLEX CHAIN
```py
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
vectorstore = DocArrayInMemorySearch.from_texts(
    [
        "Cinderella lived with her cruel stepmother and two stepsisters.",
        "She was forced to do all the household chores.",
        "A fairy godmother helped Cinderella attend the royal ball.",
        "She wore a beautiful gown and glass slippers.",
        "At midnight the magic disappeared.",
        "The prince searched the kingdom using the glass slipper.",
        "The slipper fit Cinderella, and they were married."
    ],
    embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()
retriever.get_relevant_documents("Who helped Cinderella?")
retriever.get_relevant_documents("Why did Cinderella leave the ball?")
retriever.get_relevant_documents("How did the prince find Cinderella?")
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
from langchain.schema.runnable import RunnableMap
chain = RunnableMap({
    "context": lambda x: retriever.get_relevant_documents(x["question"]),
    "question": lambda x: x["question"]
}) | prompt | model | output_parser
chain.invoke({"question": "Who helped Cinderella attend the ball?"})

chain.invoke({"question": "What happened at midnight?"})

chain.invoke({"question": "How did the prince identify Cinderella?"})
inputs = RunnableMap({
    "context": lambda x: retriever.get_relevant_documents(x["question"]),
    "question": lambda x: x["question"]
})
inputs.invoke({"question": "Who helped Cinderella attend the ball?"})
```

### OUTPUT:
<img width="1061" height="142" alt="image" src="https://github.com/user-attachments/assets/026367cc-8d35-4829-8dfa-a189a5358d6e" />


### RESULT:
The implemented LCEL expression takes at least two prompt parameters, processes them using a model, and formats the output with a parser, demonstrating its effectiveness through real-world examples.
