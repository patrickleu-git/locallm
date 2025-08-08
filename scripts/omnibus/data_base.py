from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_chroma import Chroma
from get_embedding import get_embedding

DATA_PATH = "data/pdfs/eu-omnibus"
CHROMA_PATH = "chroma_langchain_db/eu-omnibus"

EMBEDDING = "snowflake-arctic-embed2"

# populate data base
def main():
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)


# function to load pdf documents from DATA_PATH
def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()



# function to split the pdfs into smaller chunks
def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 80,
        length_function = len,
        is_separator_regex = False
    )
    return text_splitter.split_documents(documents)



# function to populate the data base
def add_to_chroma(chunks: list[Document]):
    
    # load an existing data base
    db = Chroma(
        collection_name = "omnibus",
        embedding_function = get_embedding(embedding = EMBEDDING),
        persist_directory = CHROMA_PATH,
    )

    # get chunk ids
    chunks_ids = get_chunk_ids(chunks)

    # add or update documents
    existing_itmes = db.get(include = [])
    existing_ids = set(existing_itmes["ids"])
    print(f"The number of existing documents in DB: {len(existing_ids)}")

    # only add documents 
    new_chunks = []
    
    # check whether a chunk's id is already in the data base, if not append to list
    for chunk in chunks_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if new_chunks:
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    
    else:
        print("No new documents to add")



# function to get unique chunk ids s.t. data base can be updated
def get_chunk_ids(chunks):
    
    last_id = None
    chunk_index = 0

    for chunk in chunks:
        # get data source and page
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_id = f"{source}:{page}"
        
        # check if chunk is on same page as before, if yes add 1, else reset chunk index
        if current_id == last_id:
            chunk_index += 1
        else:
            chunk_index = 0

        # create the chunk id and append it to meta data
        chunk_id = f"{current_id}:{chunk_index}"
        chunk.metadata["id"] = chunk_id

        # update last id
        last_id = current_id

    return chunks



# execute script
if __name__ == "__main__":
    main()
