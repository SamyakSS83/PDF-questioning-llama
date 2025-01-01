# PDF-based RAG using LLaMA

This project uses a PDF-based knowledge base and the LLaMA model for creating a retrieval-augmented chatbot. The agent responds to yes/no questions based on the content of the book in the PDF, providing hints about the genre or author, and if the guess is close to the book's title, the agent declares a win.

## Dependencies

### 1. **Install Ollama**

#### For Windows:
Download and install Ollama from the following link:  
[Ollama Download for Windows](https://ollama.com/download)

#### For Linux:
Run the following command to install Ollama:
```bash
curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION=0.3.6 sh


