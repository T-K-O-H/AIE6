import gradio as gr
import os
from aimakerspace.text_utils import PDFLoader, CharacterTextSplitter
from aimakerspace.vectordatabase import VectorDatabase
from aimakerspace.openai_utils.prompts import SystemRolePrompt, UserRolePrompt
from aimakerspace.openai_utils.chatmodel import ChatOpenAI
from aimakerspace.openai_utils.embedding import EmbeddingModel
import asyncio

def load_notebook():
    notebook_path = "Pythonic_RAG_Assignment.ipynb"
    if os.path.exists(notebook_path):
        with open(notebook_path, "r", encoding="utf-8") as f:
            return f.read()
    return "Notebook not found"

with gr.Blocks() as demo:
    gr.Markdown("# RAG Implementation Notebook")
    gr.Markdown("This space contains a Jupyter notebook demonstrating a Retrieval Augmented Generation (RAG) implementation.")
    
    with gr.Tabs():
        with gr.TabItem("Notebook Preview"):
            notebook_content = gr.Markdown(load_notebook())
        
        with gr.TabItem("About"):
            gr.Markdown("""
            ## About This Space
            
            This space contains a Jupyter notebook that demonstrates:
            - PDF document processing
            - Text chunking and embedding
            - Vector database implementation
            - RAG pipeline with context-aware responses
            
            To run the notebook locally:
            1. Clone this repository
            2. Install requirements: `pip install -r requirements.txt`
            3. Run: `jupyter notebook Pythonic_RAG_Assignment.ipynb`
            """)

# Initialize the RAG pipeline
def initialize_rag():
    # Load the PDF
    pdf_loader = PDFLoader("data/How-to-Build-a-Career-in-AI.pdf")
    documents = pdf_loader.load_documents()
    
    # Split the documents
    text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    split_documents = text_splitter.split_texts(documents)
    
    # Create vector database
    embedding_model = EmbeddingModel()
    vector_db = VectorDatabase(embedding_model=embedding_model)
    vector_db = asyncio.run(vector_db.abuild_from_list(split_documents))
    
    # Set up prompts
    RAG_PROMPT_TEMPLATE = """ \
    Use the provided context to answer the user's query.

    You may not answer the user's query unless there is specific context in the following text.

    If you do not know the answer, or cannot answer, please respond with "I don't know".
    """
    rag_prompt = SystemRolePrompt(RAG_PROMPT_TEMPLATE)
    
    USER_PROMPT_TEMPLATE = """ \
    Context:
    {context}

    User Query:
    {user_query}
    """
    user_prompt = UserRolePrompt(USER_PROMPT_TEMPLATE)
    
    # Create ChatOpenAI instance
    chat_openai = ChatOpenAI()
    
    # Create and return pipeline
    return RetrievalAugmentedQAPipeline(vector_db_retriever=vector_db, llm=chat_openai)

class RetrievalAugmentedQAPipeline:
    def __init__(self, llm: ChatOpenAI, vector_db_retriever: VectorDatabase) -> None:
        self.llm = llm
        self.vector_db_retriever = vector_db_retriever

    def run_pipeline(self, user_query: str) -> str:
        context_list = self.vector_db_retriever.search_by_text(user_query, k=4)
        context_prompt = ""
        for context in context_list:
            context_prompt += context[0] + "\n"
        
        formatted_system_prompt = SystemRolePrompt(""" \
        Use the provided context to answer the user's query.
        You may not answer the user's query unless there is specific context in the following text.
        If you do not know the answer, or cannot answer, please respond with "I don't know".
        """).create_message()
        
        formatted_user_prompt = UserRolePrompt(""" \
        Context:
        {context}

        User Query:
        {user_query}
        """).create_message(user_query=user_query, context=context_prompt)
        
        response = self.llm.run([formatted_system_prompt, formatted_user_prompt])
        return response

# Create Gradio interface
def create_interface():
    # Initialize RAG pipeline
    rag_pipeline = initialize_rag()
    
    def query_rag(question):
        return rag_pipeline.run_pipeline(question)
    
    with gr.Blocks(title="RAG Implementation") as demo:
        gr.Markdown("# RAG Implementation Demo")
        gr.Markdown("Ask questions about the 'How to Build a Career in AI' document")
        
        with gr.Row():
            with gr.Column():
                question = gr.Textbox(label="Your Question", placeholder="Type your question here...")
                submit_btn = gr.Button("Submit")
            
            with gr.Column():
                answer = gr.Textbox(label="Answer", lines=5)
        
        submit_btn.click(
            fn=query_rag,
            inputs=question,
            outputs=answer
        )
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch() 