from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

class ChainFactory:
    def __init__(self, documents, llm):
        self.documents = documents
        self.llm = llm
        self.embeddings = HuggingFaceEmbeddings()
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
    
    def create_chain(self):
        template = """You are a helpful AI assistant that provides accurate information about the Yes It Works documentation tool. Your goal is to help users understand how to use the tool effectively.

Context: {context}

Question: {question}

Instructions:
1. If the information is in the context, provide a clear and structured answer
2. If the information isn't in the context but you can make a reasonable suggestion based on common documentation tool features, say "While I don't have specific documentation about this, typically in documentation tools you can..."
3. If you truly don't know, suggest related topics the user might want to explore instead
4. Always be encouraging and solution-oriented

Answer: """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 4}),
            combine_docs_chain_kwargs={
                "prompt": prompt,
                "document_prompt": PromptTemplate(
                    input_variables=["page_content"],
                    template="{page_content}\n"
                )
            },
            return_source_documents=True
        )
