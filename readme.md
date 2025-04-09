# Yes It Works Documentation Helper

An interactive chat application that helps users understand and navigate the Yes It Works tool using RAG (Retrieval-Augmented Generation) with Hugging Face models and Streamlit.

## Features

- 💬 Modern chat interface with conversation history
- ✨ Instant answers to your Yes It Works questions
- 🔍 Intelligent documentation search using RAG
- 📖 Comprehensive feature coverage
- 🎯 Topic-based information navigation
- ⚡ Fast responses using FAISS vector storage
- 🎨 Clean, user-friendly Streamlit interface

## Project Structure

```
csv_rag/
├── app.py                    # Main entry point
├── data/
│   └── yesItWorks_doc.csv    # Documentation database
├── src/
│   ├── chains/              # LangChain components
│   ├── data_loaders/        # CSV data loader
│   ├── embeddings/          # Embedding management
│   └── ui/                  # Streamlit UI components
└── .streamlit/              # Streamlit configuration
```

## Example Questions

- "How do I create a new page?"
- "What formatting options are available?"
- "How can I create tables?"
- "How do I share pages with others?"
- "Can I work offline?"

## Feature Categories

- Page Management
- Text Editing
- Media
- Database
- Collaboration
- Integrations
- General

## Setup

1. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Get Hugging Face API Token:
   - Go to [Hugging Face Settings](https://huggingface.co/settings/tokens)
   - Create a new token with "read" access
   - Copy the generated token

## Usage

1. Start the application:
   ```bash
   streamlit run app.py
   ```

2. In the application:
   - Enter your Hugging Face API token in the sidebar
   - Browse feature categories in the sidebar
   - Use the chat interface to ask questions
   - Reference the example questions for guidance

## How It Works

1. **Data Loading**: Documentation is loaded from a curated CSV file
2. **Embeddings**: Text is converted to embeddings using HuggingFace's sentence transformers
3. **Vector Storage**: FAISS is used for efficient similarity search
4. **RAG Pipeline**: 
   - User questions are processed through a retrieval chain
   - Relevant documentation is retrieved
   - A language model generates natural responses
   - Chat history is maintained for context

## Environment Variables

- `HUGGINGFACE_API_TOKEN`: Your Hugging Face API token (set via UI)

## Dependencies

- `streamlit==1.28.2`: Web interface
- `langchain==0.0.335`: RAG pipeline
- `pandas==2.2.3`: Data handling
- `faiss-cpu==1.7.4`: Vector storage
- `sentence-transformers==2.2.2`: Text embeddings
- `huggingface-hub==0.17.3`: Model access
- `transformers==4.35.0`: Text generation

## License

MIT