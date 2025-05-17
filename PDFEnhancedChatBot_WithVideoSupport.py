import streamlit as st
import os
import shutil
import gc
import cv2
import whisper
from PIL import Image
import pytesseract
from PDFhandling import load_pdf, upload_pdf, pdfs_directory, figures_directory
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document


os.environ["OLLAMA_HOST"] = "http://ollama:11434"
# -- Directories --
CHROMA_DB_DIR = "chroma_DB"
VIDEO_UPLOAD_DIR = "video_uploads"
SLIDE_DIR = "slide_frames"
os.makedirs(VIDEO_UPLOAD_DIR, exist_ok=True)
os.makedirs(SLIDE_DIR, exist_ok=True)
os.makedirs(pdfs_directory, exist_ok=True)
os.makedirs(figures_directory, exist_ok=True)
# -- Models and Config --
EMBED_MODEL = "llama3.2"
LLM_MODEL = "gemma3"
embeddings = OllamaEmbeddings(model=EMBED_MODEL)
llm = OllamaLLM(model=LLM_MODEL)

template = """You are an assistant for question-answering tasks. Use chat history and content to answer.
Chat History:
{history}

Content:
{context}

Question: {question}
Answer:"""

@st.cache_resource
def get_vector_store():
    return Chroma(collection_name="everything", embedding_function=embeddings, persist_directory=CHROMA_DB_DIR)

# -- Utilities --
def load_page(url):
    loader = SeleniumURLLoader(urls=[url])
    documents = loader.load()
    return documents

def split_text(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

def index_docs(docs, label):
    store = get_vector_store()
    wrapped = [Document(page_content=doc.page_content, metadata={"source_url": label}) for doc in docs]
    store.add_documents(wrapped)
    store.persist()

def retrieve_docs(query, filter_source=None):
    store = get_vector_store()
    if filter_source and filter_source != "All":
        return store.similarity_search(query, k=4, filter={"source_url": filter_source})
    all_metadatas = store.get()["metadatas"]
    sources = set(meta["source_url"] for meta in all_metadatas if meta and "source_url" in meta)

    results = []
    for src in sources:
        results += store.similarity_search(query, k=2, filter={"source_url": src})
    return results

def answer_question(question, context):
    history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.chat_history[-6:]])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm
    return chain.invoke({"question": question, "context": context, "history": history})

def summarize_pdf_text(text: str):
    prompt = ChatPromptTemplate.from_template("Summarize the following content:\n{context}")
    chain = prompt | llm
    return chain.invoke({"context": text[:3000]})

def transcribe_audio(video_path):
    model = whisper.load_model("base", device="cuda")
    return model.transcribe(video_path)["text"]

def extract_frames(video_path, every_n_sec=5):
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    interval = int(fps * every_n_sec)
    success, image = vidcap.read()
    count, frame_idx = 0, 0
    paths = []
    while success:
        if count % interval == 0:
            path = os.path.join(SLIDE_DIR, f"frame_{frame_idx}.jpg")
            cv2.imwrite(path, image)
            paths.append(path)
            frame_idx += 1
        success, image = vidcap.read()
        count += 1
    return paths

def ocr_image(path):
    return pytesseract.image_to_string(Image.open(path))

# -- Streamlit App --
st.title("🤖 AI Chatbot for PDFs, URLs & Presentations")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.sidebar.button("🧹 Clear Vector DB"):
    get_vector_store.clear()
    if os.path.exists(CHROMA_DB_DIR):
        shutil.rmtree(CHROMA_DB_DIR)
    st.sidebar.success("Vector DB cleared.")

if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []

store = get_vector_store()
try:
    metadatas = store.get()["metadatas"]
    all_sources = sorted(set(meta.get("source_url") for meta in metadatas if meta and meta.get("source_url")))
    st.sidebar.markdown("### Sources Indexed:")
    for src in all_sources:
        label = f"📄 {src}" if src.endswith(".pdf") else f"🌐 {src}" if "http" in src else f"🎥 {src}"
        st.sidebar.write(f"- {label}")
except:
    all_sources = []

# -- URL Input --
url = st.text_input("Enter URL:")
if url:
    docs = load_page(url)
    summary = summarize_pdf_text(docs)
    st.expander("📄 URL Summary").write(summary)
    chunks = split_text(docs)
    index_docs(chunks, url)
    st.success("URL indexed.")

# -- PDF Upload --
uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
if uploaded_files:
    for pdf in uploaded_files:
        upload_pdf(pdf)
        text = load_pdf(os.path.join(pdfs_directory, pdf.name))
        summary = summarize_pdf_text(text)
        st.expander("📄 PDF Summary").write(summary)
        chunks = split_text([Document(page_content=text)])
        index_docs(chunks, pdf.name)

# -- Video Upload --
video_file = st.file_uploader("Upload Presentation (.mp4)", type=["mp4", "mov"])
if video_file:
    path = os.path.join(VIDEO_UPLOAD_DIR, video_file.name)
    with open(path, "wb") as f:
        f.write(video_file.read())
    st.video(path)

    with st.spinner("Transcribing audio..."):
        transcript = transcribe_audio(path)
        st.expander("🗣 Transcript").write(transcript)
        chunks = split_text([Document(page_content=transcript)])
        index_docs(chunks, video_file.name + "_audio")

    with st.spinner("Extracting slides and running OCR..."):
        frames = extract_frames(path)
        ocr_texts = [ocr_image(p) for p in frames if ocr_image(p).strip()]
        if ocr_texts:
            chunks = split_text([Document(page_content="\n".join(ocr_texts))])
            index_docs(chunks, video_file.name + "_slides")
        st.success(f"OCR extracted from {len(ocr_texts)} slides.")
        for file in os.listdir(SLIDE_DIR):
            file_path = os.path.join(SLIDE_DIR, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

# -- Chat --
selected_source = st.sidebar.selectbox("Filter by Source (or All):", ["All"] + all_sources)
question = st.chat_input("Ask a question...")

if question:
    st.chat_message("user").write(question)
    docs = retrieve_docs(question, selected_source)
    context = "\n\n".join([d.page_content for d in docs])
    answer = answer_question(question, context)
    st.chat_message("assistant").write(answer)
    st.session_state.chat_history.append({"role": "user", "content": question})
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    with st.expander("📑 Chunks Retrieved"):
        for i, doc in enumerate(docs):
            st.markdown(f"**Chunk {i+1} from {doc.metadata.get('source_url')}:**\n{doc.page_content[:500]}...")
