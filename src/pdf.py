import requests
from indexify import IndexifyClient, ExtractionGraph

# Define the extraction graph specification
extraction_graph_spec = """
name: 'pdfknowledgebase2'
extraction_policies:
   - extractor: 'tensorlake/pdfextractor'
     name: 'pdf_to_text'
"""

# Initialize the Indexify client
client = IndexifyClient()

# # Create the extraction graph from the specification
# extraction_graph = ExtractionGraph.from_yaml(extraction_graph_spec)
# client.create_extraction_graph(extraction_graph)

def process_pdf(path, kd_base_path):
    # Upload the PDF file to the specified knowledge base path
    content_id = client.upload_file(kd_base_path, path)
    
    # Wait for the extraction to complete
    client.wait_for_extraction(content_id)
    
    # Retrieve and print extracted content
    extracted_content = client.get_extracted_content(
        ingested_content_id=content_id,
        graph_name=kd_base_path,
        policy_name="pdf_to_text"
    )
    
    # Decode and return the extracted text content
    text = extracted_content[0]['content'].decode('utf-8')
    return text

if __name__ == "__main__":
    # Specify the path to the PDF file and the knowledge base path
    pdf_path = "../book/Daniel Kahneman-Thinking, Fast and Slow  .pdf"
    kd_base_path = "pdfknowledgebase2"
    
    # Process the PDF file and print the extracted text
    extracted_text = process_pdf(pdf_path, kd_base_path)
    #print(extracted_text)

    # Save the extracted text to a file
    output_file_path = "../text/extracted_text.txt"
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write(extracted_text)
    
    print(f"Extracted text saved to {output_file_path}")
