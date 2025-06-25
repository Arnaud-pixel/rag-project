

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from vector import retriever

#Define the LLM to handle the query
llm = OllamaLLM(model="llama3.2")

# Define the prompt template to be passed to the LLM model with some context including the question and the information from the vectorized db.
template = """
You're an expert in answering questions.
Maintain a professional tone and ensure your responses are accurate and helpful.
Strictly adhere to the user's question and provide relevant information.
If you do not know the answer, then respond "I don't know".
Do not refer to your knowledge base.


Here are some relevant information: {info}

Here is the question to answer: {question}

"""

#Create an object prompt based on the template
prompt = ChatPromptTemplate.from_template (template)

#Create an object chain the feed the llm with the prompt
chain = prompt | llm

#Create a loop for the user to input a question
while True:
    print ("\n\n--------------------------")
    question = input("Ask your question (q to quit): ")
    print ("\n\n")
    if question == "q":
        break
    #Get the information from the vector db based on the question
    vector_output=retriever.invoke(question)
    #Get the response from the LLM using the customised prompt with the information from the vector db and the question
    result = chain.invoke({"info":vector_output,"question":question})
    #Display the response
    print (result)

