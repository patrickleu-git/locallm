# trying different pdf loaders and their quality / flexibility / aptness for the PAV documents

from langchain_community.document_loaders.blob_loaders import FileSystemBlobLoader # load multiples
from langchain_community.document_loaders.generic import GenericLoader # generic loader
from langchain_pymupdf4llm import PyMuPDF4LLMParser
from langchain_community.document_loaders.parsers import LLMImageBlobParser
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_chroma import Chroma

DATA_PATH = "data/pdfs/pav"
CHROMA_PATH = "chroma_langchain_db/pav"

EMBEDDING = "snowflake-arctic-embed2"

loader = GenericLoader(
    blob_loader=FileSystemBlobLoader(
        path=DATA_PATH, 
        glob="*.pdf"
    ), 
    blob_parser=PyMuPDF4LLMParser(
        mode="single",
        extract_images=True,
        images_parser=LLMImageBlobParser(
            model = ChatOllama(
                model="gemma3:4b",
                temperature=1, 
                num_predict=1024
            )
        ),
    )
)


docs = loader.load()
print(docs[0].metadata)
print(len(docs))
