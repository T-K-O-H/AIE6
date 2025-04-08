from aimakerspace.text_utils import PDFLoader

def test_pdf_loading():
    # Initialize the PDF loader
    pdf_loader = PDFLoader("data/How-to-Build-a-Career-in-AI.pdf")
    
    # Load the documents
    documents = pdf_loader.load_documents()
    
    # Print results
    print(f"Number of documents loaded: {len(documents)}")
    print("\nFirst document preview:")
    print(documents[0][:500] + "...")  # Show first 500 characters
    
    # Print document lengths
    print("\nDocument lengths:")
    for i, doc in enumerate(documents):
        print(f"Document {i+1}: {len(doc)} characters")

if __name__ == "__main__":
    test_pdf_loading() 