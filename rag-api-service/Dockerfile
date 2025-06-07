# File name: Dockerfile

FROM python:3.10.15

# Install system dependencies
RUN apt-get update && apt-get install -y \
    iputils-ping \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python path to include the root directory for imports
ENV PYTHONPATH="${PYTHONPATH}:/opt"

# Install kubectl for Kubernetes operations
RUN curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" \
    && install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Set working directory
WORKDIR /opt

# Copy requirements and install dependencies
COPY requirements.txt /opt/
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ /opt/src/

# Create logs directory
RUN mkdir -p /opt/logs

# Expose the default port
EXPOSE 8000

# Command to run the service with config file
ENTRYPOINT ["python", "-m", "src.main"]