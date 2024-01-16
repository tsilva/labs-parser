FROM node:16.19.0-bullseye-slim

# Set the working directory to /workspace
WORKDIR /workspace

# Install system packages
RUN apt-get update && \
    apt-get install -y \
    npm \
    gettext \
    iputils-ping \
    wget \
    curl \
    python3 \
    python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory to /workspaces/labs-parser
WORKDIR /workspaces/labs-parser

# Copy the entire project directory
COPY . .

# Install python packages
RUN pip3 install --no-cache-dir -r ./backend/requirements.txt

# Set the default command to run when the container starts
CMD ["/bin/bash"]
