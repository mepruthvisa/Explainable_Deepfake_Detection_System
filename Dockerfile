FROM python:3.10-slim

# Install system dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

# Set environment variables for Python
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Copy the current directory contents into the container at $HOME/app setting the owner to the user
COPY --chown=user . $HOME/app

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Create the uploads directory explicitly just in case
RUN mkdir -p $HOME/app/static/uploads

# Expose the standard Hugging Face Spaces port
EXPOSE 7860

# Run the app using Gunicorn with preload to save RAM and timeout for heavy models
CMD ["gunicorn", "-b", "0.0.0.0:7860", "-w", "2", "--timeout", "120", "--preload", "app:app"]
