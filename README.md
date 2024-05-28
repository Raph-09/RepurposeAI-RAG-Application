# Repurpose AI Documentation

## Overview
Repurpose AI is an application designed to convert blog content into posts for various social media platforms using AI. It utilizes Streamlit for the web interface, Google Generative AI for embeddings and text generation, and FAISS for managing the vector database.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/repurpose-ai.git
   cd repurpose-ai
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   Create a `.env` file in the root directory and add the necessary environment variables, especially for Google API Key.

## Configuration

### Environment Variables
Ensure you have the following environment variables set in your `.env` file:

```
GOOGLE_API_KEY=<your_google_api_key>
```

### Style
The application uses a custom CSS file for styling. Ensure `style.css` is in the root directory or update the path in the code accordingly.

## Code Explanation

### Imports

```python
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from bs4 import BeautifulSoup
import requests
from dotenv import load_dotenv
```

### Environment Setup

```python
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
```

### Session State Initialization

```python
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
```

### Read Blog Content

```python
def read_blog(URL):
    r = requests.get(URL)
    soup = BeautifulSoup(r.text, 'html.parser')
    results = soup.find_all(['h1', 'p'])
    text = [result.text for result in results]
    blog = ' '.join(text)
    return blog
```

### Chunk Text

```python
def chunk_text(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=11000, chunk_overlap=1100)
    chunks = text_splitter.split_text(text)
    return chunks
```

### Embedding Vector

```python
def embedding_vector(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_db = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_db.save_local("faiss_index")
```

### Chain Prompt Question

```python
def chain_prompt_question():
    prompt_template = """
    You are a repurpose AI system, your job is to convert the text or blog into a post in the provided social media platform\n\n
    Context:\n {context}?\n
    social media platform: \n{platform}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "platform"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain
```

### Get Context and Question

```python
def get_context_question(input):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    context = new_db.similarity_search(input)
    chain = chain_prompt_question()
    response = chain({"input_documents": context, "platform": input}, return_only_outputs=True)
    return response["output_text"]
```

### Streamlit UI

```python
def main():
    st.set_page_config(page_title="Repurpose AI")
    st.title("Repurpose AI")
    st.subheader("Convert Blogs Into Social Media Posts")
    with open("style.css") as source_des:
        st.markdown(f"<style>{source_des.read()}</style>", unsafe_allow_html=True)
    
    URL = st.text_input("Paste Blog Url and press Enter: ")
    st.write("")
    st.write("")
    if URL:
        with st.spinner("Processing..."):
            text = read_blog(URL)
            text_chunks = chunk_text(text)
            embedding_vector(text_chunks)
   
    submit1 = st.button("Generate LinkedIn Post")
    input = "Linkedin"
    if submit1:
        if URL:
            res = get_context_question(input)
            st.write("Answer: ", res)
            st.session_state['chat_history'].append(("AI", res))
            st.session_state['chat_history'].append(("Article", URL))
            st.write("Read More:", URL)
        else:
            st.write("")
    
    submit2 = st.button("Generate Twitter Post")
    input = "Twitter: must not be more than 280 characters"
    if submit2:
        if URL:
            res = get_context_question(input)
            st.write("Answer: ", res)
            st.session_state['chat_history'].append(("AI", res))
            st.session_state['chat_history'].append(("Article", URL))
            st.write("Read More:", URL)
        else:
            st.write("")
    
    submit3 = st.button("Generate WhatsApp Status Post")
    input = "WhatsApp Status: must not be more than 150 characters"
    if submit3:
        if URL:
            res = get_context_question(input)
            st.write("Answer: ", res)
            st.session_state['chat_history'].append(("AI", res))
            st.session_state['chat_history'].append(("Article", URL))
            st.write("Read More:", URL)
        else:
            st.write("")
    
    submit4 = st.button("Generate Twitter Threads Post")
    input = "Twitter Threads: Each thread must be 150 words"
    if submit4:
        if URL:
            res = get_context_question(input)
            st.write("Answer: ", res)
            st.session_state['chat_history'].append(("AI", res))
            st.session_state['chat_history'].append(("Article", URL))
            st.write("Read More:", URL)
        else:
            st.write("")
    
    # Sidebar for chat history
    sidebar = st.sidebar
    sidebar.title("History")
    if 'chat_history' in st.session_state:
        for role, text in st.session_state['chat_history']:
            sidebar.write(f"{role}: {text}")

if __name__ == "__main__":
    main()
```

## Running the Application

1. **Start the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

2. **Access the app:**
   Open your web browser and go to `http://localhost:8501`.

## Usage

1. Paste the URL of a blog in the input field and press Enter.
2. Choose the type of social media post you want to generate (LinkedIn, Twitter, WhatsApp Status, Twitter Threads).
3. View the generated post content and the original blog link.

## Dependencies

- Python 3.7 or later
- Streamlit
- LangChain
- FAISS
- Google Generative AI
- BeautifulSoup
- Requests
- dotenv

Ensure you have all dependencies installed as specified in the `requirements.txt` file.

## Contributing

Feel free to submit issues or pull requests if you find any bugs or have suggestions for improvements. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Contact

For any inquiries or support, please contact [yourname@domain.com](mailto:yourname@domain.com).
