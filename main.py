import os
from dotenv import load_dotenv
from datasets import load_dataset
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from pinecone import Pinecone, ServerlessSpec  # Correct Pinecone imports

# Load environment variables from .env file
load_dotenv()

# Initialize the Pinecone client
pc = Pinecone(
    api_key=os.getenv('PINECONE_API_KEY')
)

# Define index name
index_name = "medical-chatbot-index"

# Initialize HuggingFace Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Load the dataset from Hugging Face
dataset = load_dataset("mohammad2928git/complete_medical_symptom_dataset")

# Preprocess the dataset
docs = []
for item in dataset['train']:
    document = Document(
        page_content=f"Text: {item['text']}, Symptoms: {item['symptoms']}",
        metadata={"source": "medical_dataset"}
    )
    docs.append(document)

# Split documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
split_docs = text_splitter.split_documents(docs)

# Check if the index exists; create it if not
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-west-2'
        )
    )

# Connect to the existing index
index = pc.Index(index_name)

# Use LangChain's Pinecone wrapper to create a vector store
docsearch = LangchainPinecone.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Define the LLM
repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
llm = HuggingFaceHub(
    repo_id=repo_id,
    model_kwargs={
        "temperature": 0.7,
        "max_length": 512,
        "top_k": 50,
        "top_p": 0.9
    },
    huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
)

# Define the prompt template
template = """
You are an empathetic medical assistant specializing in therapeutic education for patients.
Context: {context}
Question: {question}
Answer:
"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# Define the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(),
    chain_type_kwargs={"prompt": prompt}
)

class ChatBot:
    def __init__(self):
        self.qa_chain = qa_chain

    def ask_question(self, question):
        return self.qa_chain.run(question)

    def process_uploaded_file(self, filepath):
        import mimetypes
        mime_type, _ = mimetypes.guess_type(filepath)

        if mime_type == 'application/pdf':
            text = self._extract_text_from_pdf(filepath)
        else:
            text = self._extract_text_from_txt(filepath)

        document = Document(page_content=text, metadata={"source": "user_uploaded"})
        split_docs = text_splitter.split_documents([document])
        docsearch.add_documents(split_docs)

    def _extract_text_from_pdf(self, filepath):
        import PyPDF2
        text = ""
        with open(filepath, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text()
        return text

    def _extract_text_from_txt(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()

if __name__ == "__main__":
    bot = ChatBot()
    query = input("Ask me anything: ")
    result = bot.ask_question(query)
    print(result)
