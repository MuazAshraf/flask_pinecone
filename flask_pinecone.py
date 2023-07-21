from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import JSONLoader
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
import os
from flask import *

from flask_cors import CORS
import openai
import pinecone


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config['SECRET_KEY'] = 'your-secret-key'
# Load environment variables from .env file
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv('.env')
openai_api_key  = os.environ.get('API_KEY')
hugging_face_api = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
pinecone_api_key = os.environ.get('PINECONE_API_KEY')
pinecone_env = os.environ.get('PINECONE_API_ENV')

#Template
template = """
Based on the provided context, answer every question under each heading:

Context: {context}
Question: Describe the answer to every question under each of the following headings:
- Introduction
- Key Points in bullets
- Conclusion
{query}

Answer:
"""
# Prompt Template
script_template = PromptTemplate(
    input_variables=['query','context'],
    template=template
)

# Llms
model_name="gpt-3.5-turbo-16k"
llm = ChatOpenAI(temperature=0.8,model_name=model_name, max_tokens=5000)
script_chain = LLMChain(llm=llm, prompt=script_template,output_key='Answer',verbose=True)

@app.route('/')
def main():
    return render_template("upload.html")

#question answering 
chain = load_qa_chain(llm, chain_type="stuff")
ALLOWED_EXTENSIONS = {'txt', 'pdf'}
def allowed_file(file):
    return '.' in file and \
           file.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
@app.route('/upload', methods=['POST','GET'])
def upload():
    file_extension = None
    if request.method == 'POST':
        # Get the list of files from webpage
        files = request.files.getlist("file")
  
        # Iterate for each file in the files List, and Save them
        for file in files:
            file.save(file.filename)
        _, file_extension = os.path.splitext(file.filename)
        documents = None
    if allowed_file in files:
        if file_extension.lower() == '.pdf':
            # Load the PDF file with PyPDFLoader
            loader = PyPDFLoader(file.filename)
            documents = loader.load()
        elif file_extension.lower() == '.txt':
            # Load the text file with TextLoader
            loader = TextLoader(file.filename,encoding='utf8')
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    else:
    # Handle the case when file_extension is None
        raise ValueError("File extension could not be determined.")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    
    #embedings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key,show_progress_bar=True)
    
    #for existing
    pinecone.init(
        api_key=pinecone_api_key,  # find at app.pinecone.io
        environment=pinecone_env  # next to api key in console
    )
    index_name = "testing"
    namespaces = 'docs'
    meta = [{'text': str(doc.page_content)} for doc in docs]
    print("Meta",meta)
    # Upsert data into the selected namespace
    # Create an index
    index = pinecone.Index(index_name)
    # Store the embeddings in the Pinecone index with metadata
    to_upsert = [(f"doc-{i}", embeddings.embed_documents([doc.page_content]), meta[i]) for i, doc in enumerate(docs)]
    print("Upsert",to_upsert)
    index.upsert(vectors=to_upsert, namespace=namespaces)

    index = Pinecone.from_documents(docs, embeddings, index_name=index_name,namespace=namespaces)
    
    def get_similiar_docs(query ,top_k=5, score=True):
        # Convert the query to a vector
        query_vector = embeddings.embed_documents([query])
        if score:
            similar_docs = index.similarity_search_with_score(query_vector)
        else:
            similar_docs = index.similarity_search(query_vector, top_k=top_k)
        return similar_docs
    def get_answer(query):
        # Get similar documents
        similar_docs = get_similiar_docs(query)
        inputs = {'query': query, 'context': similar_docs}
        answer = script_chain.run(inputs)
        return answer
    if request.form.get('go'):
       query = request.form.get('question')
       return get_answer(query)
    return render_template('question.html')
   

         
if __name__ == '__main__':
    app.run(debug=True)

