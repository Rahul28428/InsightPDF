# InsightPDF ✨

InsightPDF is an interactive Streamlit application that allows users to chat with their PDF documents using Google Generative AI. This project enhances document interaction and information retrieval by providing detailed answers to user queries based on the content of the uploaded PDFs.
<br>
<br>
Demo : <a href="https://www.loom.com/share/3ea3c24f0cfb477d90a1f52650e4048e?sid=c413fdae-8975-4222-9b75-18b778ac2a2a"> Link </a>
<br>
## Features


- **PDF Upload**: Upload multiple PDF files to analyze.
- **Text Extraction**: Extract text from uploaded PDF files.
- **Text Chunking**: Split extracted text into manageable chunks.
- **Text Embeddings**: Generate embeddings for text chunks using Google Generative AI.
- **Conversation History**: Maintain a chat history of user interactions.
- **Question Answering**: Answer user questions u

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/InsightPDF.git
    cd InsightPDF
    ```

2. Set up a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

4. Configure the Google API key:
    - Create a `.env` file in the project root directory.
    - Add your Google API key to the `.env` file:
        ```sh
        GOOGLE_API_KEY=your_google_api_key
        ```

## Usage

1. Run the Streamlit application:
    ```sh
    streamlit run app.py
    ```

2. Open your web browser and go to `http://localhost:8501`.

3. Upload your PDF files using the sidebar menu and click on "Submit & Process".

4. Ask questions about the content of the uploaded PDFs in the main interface.

## Project Structure

- `app.py`: Main application script for running the Streamlit interface.
- `requirements.txt`: List of required Python packages.
- `.env`: Environment file to store the Google API key.

## Technologies Used
- Streamlit: Frontend framework for interactive web applications.
- PyPDF2: PDF processing library for Python.
- LangChain: Python library for natural language processing tasks.
- Google Generative AI: AI models for text embeddings and conversational AI.

## Contributing

Contributions are welcome! Please fork the repository and submit pull requests.

## License

This project is licensed under the MIT License.
