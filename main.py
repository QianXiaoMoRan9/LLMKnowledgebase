from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.vectorstores import Qdrant
from langchain import OpenAI, VectorDBQA

import pypdf
import tiktoken
enc = tiktoken.get_encoding("gpt2")
import json
from langchain.text_splitter import TokenTextSplitter
import time

model_name = "sentence-transformers/all-mpnet-base-v2"
hf_embedding = HuggingFaceEmbeddings(model_name=model_name)
EMBEDDING = hf_embedding

CHUNK_SIZE = 800

text_splitter = TokenTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=50)

START_END_TEXT_LENGTH = 10

VIDEO_CHUNK_MAX_TOKEN_SIZE = 50
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    filePath: str
    fileName: str
    page: int
    startPosition: str
    endPosition: str
    text: str


@dataclass
class VideoObj:
    video_id: str
    thumbnail: str
    title: str
    description: str
    length_sec: int
    transcript_text: str
    transcript_obj: list
    overall_summary: str
    section_summary: list


@dataclass(frozen=True)
class VideoChunk:
    video_id: str
    text: str
    start: str
    duration: float


def file_storage_to_chunks(file_storage, file_name):
    num_pages = 1
    chunks = []
    print(f"Start processing file {file_name}, type of file_storage {type(file_storage)}")
    if file_name.endswith(".html"):
        # Handle HTML files
        print("Handling HTML file:", file_name)

    elif file_name.endswith(".pdf"):
        # Handle PDF files
        print("Handling PDF file:", file_name)
        pdfReader = pypdf.PdfReader(file_storage)
        num_pages = len(pdfReader.pages)
        for idx, page in enumerate(pdfReader.pages):
            text = page.extract_text()
            splits = text_splitter.split_text(text)
            current_position = 0

            for split in splits:
                if len(split) == 0:
                    continue
                chunkEndPostion = current_position + len(split)
                chunk = Chunk(filePath=str(file_name), fileName=file_name, page=idx, startPosition=current_position,
                              endPosition=chunkEndPostion, text=split)
                chunks.append(chunk)
                current_position = chunkEndPostion

    elif file_name.endswith(".txt"):
        # Handle TXT files
        print("Handling TXT file:", file_name)
    else:
        # Handle other file types
        print("Skipping file:", file_name)
    return chunks, num_pages

with open("1803.03354.pdf", "rb") as f:
    obj = f
    chunks,num_pages= file_storage_to_chunks(obj, "war.pdf")

t0=time.time()
# We have content of this length stored in the database record for easier answering
FILE_PREVIEW_TOKEN_SIZE = 7000
preview_doc_array = []
num_tokens = 0
num_words = 0
num_chars = 0
docs = []
metadatas = []
for chunk in chunks:
    docs.append(chunk.text)
    num_chars = num_chars + len(chunk.text)
    num_words = num_words + len(chunk.text.split())
    num_tokens = num_tokens + len(enc.encode(chunk.text))
    startText = chunk.text[0: min(START_END_TEXT_LENGTH, len(chunk.text) - 1)]
    endText = chunk.text[max(0, len(chunk.text) - START_END_TEXT_LENGTH): max(0, len(chunk.text) - 1)]
    source_location = {
        "File": chunk.fileName,
        "page": chunk.page,
        "startPosition": chunk.startPosition,
        "endPosition": chunk.endPosition,
        "startText": startText,
        "endText": endText
    }
    # Store the object as json string
    metadata = {"source": json.dumps(source_location)}
    metadatas.append(metadata)
    if num_tokens < FILE_PREVIEW_TOKEN_SIZE:
        obj = {
            "text": chunk.text,
            "metadata": metadata
        }
        preview_doc_array.append(obj)
t1=time.time()
print(f"chunking took {t1-t0} seconds")

print(len(docs))
print(docs[0])

# make it into a gpu index
doc_store = Qdrant.from_texts(docs, hf_embedding, url="http://localhost:6333")
llm = OpenAI(openai_api_key="REPLACE_WITH_API_KEY")
qa = VectorDBQA.from_chain_type(llm=llm, chain_type="map_reduce", vectorstore=doc_store)

result = qa.run("What is Visual saliency patterns?")
print(result)