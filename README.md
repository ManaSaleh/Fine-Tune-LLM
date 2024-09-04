# Search Unsloth Deployment

This project deploys a comprehensive search engine application using embeddings, Qdrant for vector search, and Retrieval-Augmented Generation (RAG) for text generation. The entire pipeline is managed through a single class-based script, `app.py`, which automates data processing, tuning, embedding generation, and interaction with Qdrant.

## Features

- **Data Tuning**: Fine-tunes the model on training data for improved performance.
- **Embeddings Generation**: Uses state-of-the-art models to generate embeddings for text data.
- **Qdrant Integration**: Stores and retrieves embeddings using Qdrant for fast and efficient searches.
- **RAG**: Utilizes Retrieval-Augmented Generation to generate contextually relevant responses.

## Prerequisites

- Docker
- Docker Compose
- Python 3.8+ with necessary dependencies (listed in `requirements.txt`)

## Project Structure

- **`app.py`**: Main script containing the class that handles all operations.
- **`Dockerfile`**: Defines the environment setup for the application.
- **`requirements.txt`**: Lists the Python dependencies required for the project.

## Setup and Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/search-unsloth.git
   cd search-unsloth
   ```

2. **Install Python Dependencies**:
   Install the necessary Python packages using:
   ```bash
   cd Fine-Tune-LLM/search-unsloth
   pip install -r requirements.txt
   ```

3. **Build and Run with Docker Compose**:
   Use Docker Compose to set up the environment and run the application.
   ```bash
   docker compose up --build
   ```

4. **Access the Application**:
   The application will be available at `http://localhost:8000`.

## Usage

- **Modify `app.py`**: Customize the class methods if you need specific tuning parameters or model adjustments.
- **Run the Script**:
  ```bash
  python app.py
  ```

## Class Workflow Overview

1. **Tuning on Train Data**: Fine-tunes the model based on provided training data to enhance performance.
2. **Import Tuning**: Applies necessary tuning configurations.
3. **Test Data Processing**: Prepares test data for evaluation.
4. **Embeddings**: Generates embeddings using the specified model.
5. **Qdrant Interaction**: Manages storage and retrieval of embeddings in Qdrant.
6. **RAG Generation**: Generates responses using Retrieval-Augmented Generation.

## Troubleshooting

- Check logs if the application does not start correctly:
  ```bash
  docker compose logs
  ```

## Contributing

Feel free to fork the repository and submit pull requests for enhancements or bug fixes.
