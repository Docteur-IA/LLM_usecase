from langchain.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import AutoTokenizer, pipeline
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA


dataset_name = "databricks/databricks-dolly-15k"
page_content_column = "context"  

# Create a loader instance
loader = HuggingFaceDatasetLoader(dataset_name, page_content_column)

# Load the data
data = loader.load()

# Display the first 15 entries
print(data[:2])


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

# 'data' holds the text you want to split, split the text into documents using the text splitter.
docs = text_splitter.split_documents(data)


# Define the path to the pre-trained model you want to use
modelPath = "sentence-transformers/all-MiniLM-l6-v2"

model_kwargs = {'device':'cpu'}

encode_kwargs = {'normalize_embeddings': False}

# Initialize an instance of HuggingFaceEmbeddings with the specified parameters
embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,     # Provide the pre-trained model's path
    model_kwargs=model_kwargs, # Pass the model configuration options
    encode_kwargs=encode_kwargs # Pass the encoding options
)


text = "This is a test document."
query_result = embeddings.embed_query(text)
query_result[:3]

########################
# Prepare the database #
########################

db = FAISS.from_documents(docs, embeddings)

# search
question = "How to make a risotto?"
searchDocs = db.similarity_search(question)
print(searchDocs[0].page_content)

# use a tiny model
model_name = "Intel/dynamic_tinybert"

# Load the tokenizer associated with the specified model
tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, truncation=True, max_length=512)

# Define a question-answering pipeline using the model and tokenizer
question_answerer = pipeline(
    "question-answering",
    model=model_name,
    tokenizer=tokenizer,
    return_tensors='pt'
)

# Create an instance of the HuggingFacePipeline, which wraps the question-answering pipeline
llm = HuggingFacePipeline(
    pipeline=question_answerer,
    model_kwargs={"temperature": 0.7, "max_length": 512},
)

# retriever
retriever = db.as_retriever()

# Search
docs = retriever.get_relevant_documents("What is risotto?")
print(docs[0].page_content)
# top k4
retriever = db.as_retriever(search_kwargs={"k": 4})


# Q&A
#qa = RetrievalQA.from_chain_type(llm=llm, chain_type="refine", retriever=retriever, return_source_documents=False)


qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="refine",
    return_source_documents=False
)

question = "What are the approaches to Task Decomposition?"
result = qa_chain({"query": question})
print(result["result"])