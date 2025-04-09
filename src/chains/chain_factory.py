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
        template = """Answer the question based on the context below. Be concise and helpful.

Context: {context}

Question: {question}

Answer: """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
            combine_docs_chain_kwargs={
                "prompt": prompt,
                "document_prompt": PromptTemplate(
                    input_variables=["page_content"],
                    template="{page_content}"
                )
            },
            return_source_documents=True,
            verbose=True
        )
