from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from VectorStore import vectorstore

llm = ChatGroq(
    api_key="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",
    model_name="mixtral-8x7b-32768"
)

def rag(question: str, num_docs: int = 1):
    # 1. Retrieve relevant documents
    docs = vectorstore.similarity_search(question, k=num_docs)

    # 2. Format context from documents
    context = "\n\n".join([doc.page_content for doc in docs])

    # 3. Create RAG prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant.
        Answer the question based on the provided context.
        If you cannot answer from the context, say "I cannot answer this from the provided information."

        Context:
        {context}"""),
        ("human", "{question}")
    ])

    # 4. Get formatted prompt
    formatted_prompt = prompt.format(
        context=context,
        question=question
    )

    # 5. Get response (ensure `llm` is defined and an instance of a language model)
    response = llm.invoke(formatted_prompt)

    # Debugging prints
    print("\n=== RAG Process Details ===")
    print("\nQuery:", question)
    print("\nRetrieved Context:", context)
    print("\nFormatted Prompt:", formatted_prompt)
    print("\n=========================== RAG Final Response ===========================")
    print("\nResponse:", response.content)

    return response.content

question = "Project made by Yash Jain"
answer = rag(question)