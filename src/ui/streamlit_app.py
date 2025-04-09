import streamlit as st
import os
import pandas as pd
from typing import Dict, Type, List
from src.data_loaders.csv_loader import CSVLoader
from src.embeddings.embedding_manager import EmbeddingManager
from src.chains.chain_factory import ChainFactory
from langchain.llms import HuggingFaceHub

class YesItWorksApp:
    """Streamlit application for Yes It Works documentation helper."""

    def __init__(self):
        """Initialize the application."""
        self.csv_path = "data/yesItWorks_doc.csv"
        self.setup_page()
        self.initialize_session_state()

    def setup_page(self):
        """Configure the Streamlit page."""
        st.set_page_config(
            page_title="Yes It Works - Documentation Helper",
            page_icon="✨",
            layout="wide"
        )
        st.title("Yes It Works - Documentation Helper")

    def initialize_session_state(self):
        """Initialize session state variables."""
        if "messages" not in st.session_state:
            welcome_message = (
                "👋 Hi! I'm your Yes It Works documentation helper. "
                "I can help you with:\n"
                "- Creating and managing pages\n"
                "- Text formatting and editing\n"
                "- Media handling\n"
                "- Database features\n"
                "- Collaboration tools\n"
                "\nWhat would you like to know?"
            )
            st.session_state.messages = [{
                "role": "assistant",
                "content": welcome_message
            }]
        if "qa_chain" not in st.session_state:
            st.session_state.qa_chain = None

    def initialize_llm(self, temperature: float = 0.7, max_length: int = 512) -> HuggingFaceHub:
        """Initialize the language model with custom parameters."""
        token = st.session_state.get('huggingface_token')
        if not token:
            st.error('Please enter your Hugging Face API token in the sidebar')
            st.stop()
        return HuggingFaceHub(
            repo_id="google/flan-t5-xl",
            model_kwargs={
                "temperature": temperature,
                "max_length": max_length,
            },
            huggingfacehub_api_token=token,
            task="text2text-generation"
        )

    def load_data(self):
        """Load documentation data and initialize the QA chain."""
        loader = CSVLoader("data/yesItWorks_doc.csv")
        documents = loader.load()
        
        # Initialize LLM with custom parameters
        llm = self.initialize_llm(
            temperature=st.session_state.get("temperature", 0.7),
            max_length=st.session_state.get("max_length", 512)
        )
        
        chain_factory = ChainFactory(documents=documents, llm=llm)
        st.session_state.qa_chain = chain_factory.create_chain()

    def get_huggingface_token(self) -> str:
        """Get the Hugging Face API token from the user."""
        token = st.sidebar.text_input(
            "Enter your Hugging Face API token:",
            type="password",
            help="Get your token from https://huggingface.co/settings/tokens"
        )
        if token:
            st.session_state.huggingface_token = token
        return token

    def display_model_settings(self):
        """Display and update model settings in the sidebar."""
        st.sidebar.markdown("### 🛠️ Model Settings")
        temperature = st.sidebar.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get("temperature", 0.7),
            step=0.1,
            help="Higher values make responses more creative but potentially less accurate"
        )
        max_length = st.sidebar.slider(
            "Max Response Length",
            min_value=64,
            max_value=1024,
            value=st.session_state.get("max_length", 512),
            step=64,
            help="Maximum number of tokens in the response"
        )
        
        # Update session state if values changed
        if temperature != st.session_state.get("temperature"):
            st.session_state.temperature = temperature
            st.rerun()
        if max_length != st.session_state.get("max_length"):
            st.session_state.max_length = max_length
            st.rerun()

    def display_example_questions(self):
        """Display example questions in the sidebar."""
        st.sidebar.markdown("### Example Questions")
        examples = [
            "How do I create a new page?",
            "What formatting options are available?",
            "How can I create tables?",
            "How do I share pages with others?",
            "Can I work offline?"
        ]
        for example in examples:
            st.sidebar.markdown(f"- {example}")

    def display_feature_categories(self):
        """Display available feature categories in the sidebar."""
        st.sidebar.markdown("### Feature Categories")
        categories = [
            "Page Management",
            "Text Editing",
            "Media",
            "Database",
            "Collaboration",
            "Integrations",
            "General"
        ]
        for category in categories:
            st.sidebar.markdown(f"- {category}")

    def run(self):
        """Run the Streamlit application."""
        st.sidebar.markdown("### 🔑 API Token")
        token = st.sidebar.text_input(
            "Enter your Hugging Face API token:",
            type="password",
            help="Get your token from https://huggingface.co/settings/tokens"
        )
        
        if not token:
            st.warning("⚠️ Please enter your Hugging Face API token in the sidebar to continue.")
            return
        
        st.session_state.huggingface_token = token
        
        # Initialize data and model after token is set
        if not hasattr(self, 'qa_chain'):
            self.load_data()
        
        self.display_model_settings()
        self.display_feature_categories()
        self.display_example_questions()

        # Display chat messages
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        if question := st.chat_input("Ask about any tech company..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": question})
            
            # Display user message
            with st.chat_message("user"):
                st.write(question)
            
            # Generate and display assistant response
            if 'qa_chain' in st.session_state:
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        # Convert message history to tuples for the chain
                        chat_history = []
                        for i in range(0, len(st.session_state.messages)-1, 2):
                            if i+1 < len(st.session_state.messages):
                                user_msg = st.session_state.messages[i]["content"]
                                assistant_msg = st.session_state.messages[i+1]["content"]
                                chat_history.append((user_msg, assistant_msg))
                        
                        response = st.session_state.qa_chain({"question": question, "chat_history": chat_history})
                        answer = response["answer"]
                        
                        # Format the answer with source information if available
                        if response.get("source_documents"):
                            sources = set()
                            for doc in response["source_documents"]:
                                if doc.metadata.get("category"):
                                    sources.add(doc.metadata["category"])
                            
                            if sources:
                                answer += f"\n\n*Related categories: {', '.join(sources)}*"
                        
                        st.write(answer)
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": answer})
        
        # Display example questions in the sidebar
        with st.sidebar:
            st.markdown("### 💡 Example Questions")
            example_questions = [
                "What is Google's interview process?",
                "Compare the benefits at Apple and Microsoft",
                "What programming languages does Amazon use?",
                "Tell me about Meta's company culture",
                "What's the work environment like at Google?"
            ]
            for q in example_questions:
                if st.button(q, key=q):
                    st.session_state.question = q
                    st.rerun()
            
            st.markdown("### 🎯 Key Topics")
            topics = [
                "🏢 Company Overview",
                "💻 Tech Stack",
                "🤝 Culture & Values",
                "💪 Benefits & Perks",
                "📋 Interview Process"
            ]
            for topic in topics:
                st.markdown(f"- {topic}")
        
        # Display About section in the sidebar
        with st.sidebar:
            st.markdown("### ℹ️ Documentation Categories")
            categories = [
                "Page Management",
                "Text Editing",
                "Media",
                "Database",
                "Collaboration",
                "Integrations",
                "General"
            ]
            for category in categories:
                st.markdown(f"- {category}")
            
            st.markdown("### 🔍 What You Can Ask")
            st.markdown("""
            - How to create and manage pages
            - Text formatting options
            - Media handling features
            - Database functionality
            - Collaboration tools
            - Integration capabilities
            - General usage tips
            """)

if __name__ == "__main__":
    app = YesItWorksApp()
    app.run()
