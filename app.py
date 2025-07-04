import os
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Step 1: Load API key
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Step 2: Load your text file
loader = TextLoader("data.txt")
documents = loader.load()

# Step 3: Split into chunks
splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
docs = splitter.split_documents(documents)

# Step 4: Create embeddings using Gemini
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma.from_documents(docs, embedding)

# Step 5: Set up Gemini Pro LLM
llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.2)



# Step 6: Create RetrievalQA chain (RAG)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# Step 7: Ask a question
query = "Who is khasif?"
response = qa_chain.invoke(query)

# Step 8: Display result
print("Question:", query)
print("Answer:", response['result'])
print("successfully executed........ well done")
