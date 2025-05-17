 docker build -t ai-chatbot .
 docker run -p 8501:8501 ai-chatbot
or
 docker-compose up --build

Project Description
This project is an AI-powered chatbot platform for processing and interacting with PDF documents (and optionally, video content). It uses advanced language models and document processing libraries to extract, analyze, and discuss both text and images from PDFs.
Key Components
•	PDFhandling.py
Handles PDF uploads and processing. It uses the unstructured library to partition PDFs into text and image/table elements. Text is extracted directly, while images and tables are processed using an AI model (OllamaLLM) to generate descriptive text, enabling the chatbot to discuss visual content.
•	AI Model Integration
Utilizes the OllamaLLM language model (e.g., "gemma3") for both text and image understanding, providing context-aware responses.
•	Dockerized Deployment
The project is containerized for easy setup and reproducibility.
•	Dockerfile: Defines the environment for the chatbot service, including dependencies for PDF and image processing (such as Poppler, Tesseract, and FFmpeg).
•	docker-compose.yml: Orchestrates multiple services:
•	ollama: Runs the language model backend.
•	ai-chatbot: Runs the chatbot and PDF processing logic.
•	Volumes are used for persistent storage of PDFs, figures, and model data.
•	Environment variables and port mappings are set for inter-service communication.
•	Extensible for Video
The presence of PDFEnhancedChatBot_WithVideoSupport.py suggests support for video content analysis.
Typical Workflow
1.	User uploads a PDF.
2.	The system extracts text, images, and tables.
3.	Images/tables are analyzed by the AI model for descriptive text.
4.	The chatbot can answer questions or discuss the content of the PDF, including visual elements.
Docker Usage
•	Build and Run:
Use the following commands to build and start the services:
  docker-compose build
  docker-compose up

•	Persistent Data:
Docker volumes ensure that uploaded PDFs, extracted figures, and model data persist across container restarts.
•	Environment Setup:
The Docker environment includes all necessary tools (Poppler, Tesseract, FFmpeg) for PDF and image processing, as configured in the Dockerfile.
