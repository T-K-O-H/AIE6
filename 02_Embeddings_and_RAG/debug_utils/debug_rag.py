from aimakerspace.text_utils import PDFLoader, CharacterTextSplitter
from aimakerspace.vectordatabase import VectorDatabase
from aimakerspace.openai_utils.prompts import SystemRolePrompt, UserRolePrompt
from aimakerspace.openai_utils.chatmodel import ChatOpenAI
from aimakerspace.openai_utils.embedding import EmbeddingModel
import asyncio
import os
from getpass import getpass

def analyze_chunks(chunks):
    print("\nChunk Analysis:")
    print(f"Total number of chunks: {len(chunks)}")
    
    # Calculate average chunk size
    total_size = sum(len(chunk) for chunk in chunks)
    avg_size = total_size / len(chunks)
    print(f"Average chunk size: {avg_size:.2f} characters")
    
    # Show size distribution
    print("\nChunk size distribution:")
    size_ranges = {
        "0-500": 0,
        "501-1000": 0,
        "1001-1500": 0,
        "1501+": 0
    }
    
    for chunk in chunks:
        size = len(chunk)
        if size <= 500:
            size_ranges["0-500"] += 1
        elif size <= 1000:
            size_ranges["501-1000"] += 1
        elif size <= 1500:
            size_ranges["1001-1500"] += 1
        else:
            size_ranges["1501+"] += 1
    
    for range_name, count in size_ranges.items():
        percentage = (count / len(chunks)) * 100
        print(f"{range_name} chars: {count} chunks ({percentage:.1f}%)")
    
    # Show sample chunks
    print("\nSample chunks from different parts of the document:")
    print("\nFirst chunk:")
    print(chunks[0][:200] + "...")
    print(f"Length: {len(chunks[0])} characters")
    
    print("\nMiddle chunk:")
    middle_idx = len(chunks) // 2
    print(chunks[middle_idx][:200] + "...")
    print(f"Length: {len(chunks[middle_idx])} characters")
    
    print("\nLast chunk:")
    print(chunks[-1][:200] + "...")
    print(f"Length: {len(chunks[-1])} characters")

def debug_rag(question: str):
    # Get OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = getpass("Enter your OpenAI API key: ")
        os.environ["OPENAI_API_KEY"] = api_key
    
    # Load the PDF
    print("Loading PDF...")
    pdf_loader = PDFLoader("data/How-to-Build-a-Career-in-AI.pdf")
    documents = pdf_loader.load_documents()
    print(f"Loaded {len(documents)} documents")
    
    # Split into chunks
    print("\nSplitting into chunks...")
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_texts(documents)
    
    # Analyze chunks
    analyze_chunks(chunks)
    
    # Create vector database
    print("\nCreating vector database...")
    embedding_model = EmbeddingModel()
    vector_db = VectorDatabase(embedding_model=embedding_model)
    vector_db = asyncio.run(vector_db.abuild_from_list(chunks))
    
    # Search for relevant chunks
    print(f"\nSearching for relevant chunks for question: '{question}'")
    context_list = vector_db.search_by_text(question, k=4)
    
    print("\nTop matching chunks:")
    for i, (chunk, score) in enumerate(context_list, 1):
        print(f"\nChunk {i} (similarity score: {score:.4f}):")
        print(chunk[:200] + "...")
        print(f"Chunk length: {len(chunk)} characters")
    
    # Create prompts
    RAG_PROMPT_TEMPLATE = """ \
    Use the provided context to answer the user's query.

    You may not answer the user's query unless there is specific context in the following text.

    If you do not know the answer, or cannot answer, please respond with "I don't know".
    """
    
    USER_PROMPT_TEMPLATE = """ \
    Context:
    {context}

    User Query:
    {user_query}
    """
    
    # Format prompts
    context_prompt = ""
    for context, _ in context_list:
        context_prompt += context + "\n"
    
    system_prompt = SystemRolePrompt(RAG_PROMPT_TEMPLATE).create_message()
    user_prompt = UserRolePrompt(USER_PROMPT_TEMPLATE).create_message(
        user_query=question,
        context=context_prompt
    )
    
    # Get response
    print("\nGetting response from LLM...")
    chat_openai = ChatOpenAI()
    response = chat_openai.run([system_prompt, user_prompt])
    
    print("\nFinal Response:")
    print(response)

if __name__ == "__main__":
    question = input("Enter your question: ")
    debug_rag(question) 