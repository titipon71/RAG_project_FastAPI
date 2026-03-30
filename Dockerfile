FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    curl \
    libmagic1 \
    libmagic-dev \
    libglib2.0-0 \
    libgl1 \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libjpeg62-turbo-dev \
    libopenjp2-7 \
    libtiff6 \
    tesseract-ocr \
    ghostscript \
    qpdf \
    poppler-utils \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

# Normalize requirements encoding to UTF-8 for reliable pip installs.
RUN python -c "from pathlib import Path; p=Path('requirements.txt'); b=p.read_bytes(); t=(b.decode('utf-8', 'ignore') if b'\x00' not in b else b.decode('utf-16', 'ignore')); p.write_text(t.replace('\ufeff','').replace('\ufffe',''), encoding='utf-8', newline='\n')"

RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.txt

COPY . .

RUN mkdir -p file_storage/uploads file_storage/trash uploads trash

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]