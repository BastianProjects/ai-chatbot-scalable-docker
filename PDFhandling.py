import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.utils.constants import PartitionStrategy
from langchain_ollama.llms import OllamaLLM

os.environ["PATH"] += os.pathsep + r"D:\Release-24.08.0-0\poppler-24.08.0\Library\bin"+os.pathsep+r"C:\Program Files\Tesseract-OCR\tesseract.exe"+os.pathsep+r"D:\ffmpeg\ffmpeg\bin"

pdfs_directory = 'pdfs/'
figures_directory = 'figures/'
model = OllamaLLM(model="gemma3")
def upload_pdf(file):
    with open(pdfs_directory + file.name, "wb") as f:
        f.write(file.getbuffer())

def load_pdf(file_path):
    elements = partition_pdf(
        file_path,
        strategy=PartitionStrategy.HI_RES,
        extract_image_block_types=["Image", "Table"],
        extract_image_block_output_dir=figures_directory
    )

    text_elements = [element.text for element in elements if element.category not in ["Image", "Table"]]

    for file in os.listdir(figures_directory):
        extracted_text = extract_text(figures_directory + file)
        text_elements.append(extracted_text)
        file_path = os.path.join(figures_directory, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

    return "\n\n".join(text_elements)

def extract_text(file_path):
    model_with_image_context = model.bind(images=[file_path])
    return model_with_image_context.invoke("Tell me what do you see in this picture.")