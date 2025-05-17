
FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    tesseract-ocr \
    chromium-driver \
    chromium \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1
ENV PATH="/usr/bin/chromium:/usr/bin/chromium-browser:$PATH"

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "PDFEnhancedChatBot_WithVideoSupport.py", "--server.port=8501", "--server.address=0.0.0.0"]
