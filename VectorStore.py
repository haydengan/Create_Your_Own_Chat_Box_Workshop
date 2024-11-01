import langchain_community
from langchain_community.document_loaders import WebBaseLoader, YoutubeLoader
from langchain_community.document_loaders.merge import MergedDataLoader

# Load documents from a website
loader1 = WebBaseLoader("https://yashjain14.github.io/")
docs1 = loader1.load()

# Load documents from a YouTube video
loader2 = YoutubeLoader.from_youtube_url(
    "https://www.youtube.com/watch?v=UF8uR6Z6KLc", add_video_info=False
)
docs2 = loader2.load()

# Merge both loaders into a single loader
loader_all = MergedDataLoader(loaders=[loader1, loader2])
docs = loader_all.load()

# Split text into smaller chunks for vector storage
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200
)
splits = text_splitter.split_documents(docs1)

# Create embeddings for the documents using HuggingFace embeddings
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Store embeddings in Chroma Vector Store (without persistence)
from langchain_chroma import Chroma
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./chroma_db")

# Run a similarity search query
print(vectorstore.similarity_search("Steve Jobs got fired", k=1))
print(vectorstore.similarity_search("Yash Jain projects", k=1))

# Store embeddings in Chroma Vector Store with persistence
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Reload from the persistent directory
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)