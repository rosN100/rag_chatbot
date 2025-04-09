import csv
from typing import List
from langchain.docstore.document import Document

class CSVLoader:
    """Loader for CSV files."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
    
    def load(self) -> List[Document]:
        """Load data from CSV file and return a list of Documents."""
        documents = []
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            # Read CSV with DictReader to handle headers
            reader = csv.DictReader(f)
            
            for row in reader:
                # Format the document content
                content = (
                    f"Title: {row['Title']}\n"
                    f"Content: {row['Content']}\n"
                    f"Category: {row['Category']}"
                )
                
                # Add metadata
                metadata = {
                    "source": self.file_path,
                    "category": row['Category'],
                    "keywords": [kw.strip() for kw in row['Keywords'].split(',')] if row.get('Keywords') else []
                }
                
                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)
        
        return documents
    
    def preprocess(self, documents: List[Document]) -> List[Document]:
        # Add any CSV-specific preprocessing here
        return documents
