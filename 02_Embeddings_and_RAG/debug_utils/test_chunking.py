from aimakerspace.text_utils import CharacterTextSplitter, PDFLoader

def test_chunking():
    # Load the PDF file
    pdf_loader = PDFLoader("data/How-to-Build-a-Career-in-AI.pdf")
    pdf_documents = pdf_loader.load_documents()
    
    # Initialize the splitter with appropriate chunk size for PDF content
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    # Split the text
    chunks = splitter.split_texts(pdf_documents)
    
    # Print results
    print(f"Total chunks: {len(chunks)}")
    print("\nFirst few chunks:")
    for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
        print(f"\nChunk {i+1} (length: {len(chunk)}):")
        print(chunk[:200] + "...")  # Show first 200 characters of each chunk
        print("-" * 50)
    
    # Verify overlap
    print("\nVerifying overlap between chunks:")
    for i in range(min(2, len(chunks)-1)):  # Check overlap for first 2 pairs
        current_chunk = chunks[i]
        next_chunk = chunks[i+1]
        overlap = len(set(current_chunk[-200:]) & set(next_chunk[:200]))
        print(f"Overlap between chunk {i+1} and {i+2}: {overlap} characters")

if __name__ == "__main__":
    test_chunking() 