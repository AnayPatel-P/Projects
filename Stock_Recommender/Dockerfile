# 1. Base image: lightweight Python runtime
FROM python:3.10-slim

# 2. Set /app as the working directory
WORKDIR /app

# 3. Copy dependency list and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of your project files
COPY . .

# 5. Expose the port Streamlit uses
EXPOSE 8501

# 6. Launch the Streamlit app by default
ENTRYPOINT ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
