# Use an official Python runtime as a parent image
FROM python:3.10-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the models directory
COPY models/ /models/

# Copy the application script
COPY process_pdfs_1b.py .

# Command to run the application (example, adjust as needed)
# ENTRYPOINT ["python", "process_pdfs_1b.py"]
