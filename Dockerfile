FROM python:3.11-slim

# Install system dependencies
# - python3-tk is required for Tkinter (guiDemo.py)
# - libgomp1 is required for OpenMP (used by scipy/numpy)
# - build-essential is useful if any pip dependencies need compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency definition
COPY requirements.txt ./

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source code (filtered by .dockerignore)
COPY . .

# Configure Python path so the package directory is discoverable
ENV PYTHONPATH=/app

# Default command: run the test suite
CMD ["pytest"]