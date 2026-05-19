import os
import re
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader

def normalize_text(text):
    if not text:
        return ""

    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"\r\n?", "\n", text)

    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]

    def is_header_or_footer(line):
        if not line:
            return True
        if re.fullmatch(r"\d{1,4}", line):
            return True
        if re.fullmatch(r"page\s+\d+(\s+of\s+\d+)?", line, re.IGNORECASE):
            return True
        if len(line) <= 3 and re.search(r"\d", line):
            return True
        return False

    if lines and is_header_or_footer(lines[0]):
        lines = lines[1:]
    if lines and is_header_or_footer(lines[-1]):
        lines = lines[:-1]

    text = " ".join(lines)
    text = re.sub(r"[\t\f\v]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def load_docs():
    pdf_directory = "docs/pdfs"
    text_directory = "docs/texts"

    pdf_loader = DirectoryLoader(pdf_directory, glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader)
    text_loader = DirectoryLoader(
        text_directory,
        glob="**/*.txt",
        show_progress=True,
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8", "autodetect_encoding": True},
    )
    documents = []

    if os.path.exists(pdf_directory):
        pdf_docs = pdf_loader.load()
        #print("Total length of first document content: ", len(pdf_docs[1].page_content) if pdf_docs else "No PDF documents loaded.")
        for doc in pdf_docs:
            doc.page_content = normalize_text(doc.page_content)
        documents.extend(pdf_docs)
    else: 
        print(f"PDF directory '{pdf_directory}' does not exist.")
    
    if os.path.exists(text_directory):
        text_docs = text_loader.load()
        documents.extend(text_docs)
    else:
        print(f"Text directory '{text_directory}' does not exist.")
    
    print(f"Loaded {len(documents)} documents.")
    #print(f"First document length after normalization: {len(documents[1].page_content) if documents else 'No documents loaded.'}")
    #print(f"Sample document metadata: {documents[1].metadata if documents else 'No documents loaded.'}")
    #print(f"Sample document content: {documents[1].page_content[:2000]}...") if documents else print("No documents loaded.")
    return documents

if __name__ == "__main__":
    load_docs()