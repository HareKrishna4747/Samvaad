# Use official Python image
FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy all project files into the container
COPY . .

# Install system dependencies
RUN apt-get update && apt-get install -y poppler-utils

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Streamlit will run on
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
