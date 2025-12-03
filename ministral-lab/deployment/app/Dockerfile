FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN pip install streamlit openai pypdf python-docx

# Copy code
COPY . .

# Run app
CMD ["streamlit", "run", "main.py", "--server.address=0.0.0.0"]
