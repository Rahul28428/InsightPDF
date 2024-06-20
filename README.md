# InsightPDF ✨

InsightPDF is an interactive Streamlit application that allows users to chat with their PDF documents using Google Generative AI. This project enhances document interaction and information retrieval by providing detailed answers to user queries based on the content of the uploaded PDFs.

## Features

- **Interactive PDF Chat**: Users can ask questions about the content of their PDF files and receive detailed, context-aware responses.
- **Advanced AI Integration**: Utilizes Google Generative AI embeddings and FAISS for accurate and efficient information retrieval.
- **User-Friendly Interface**: Easy-to-use interface built with Streamlit, allowing seamless PDF upload and query processing.

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

## Contributing

Contributions are welcome! Please fork the repository and submit pull requests.

## License

This project is licensed under the MIT License.
