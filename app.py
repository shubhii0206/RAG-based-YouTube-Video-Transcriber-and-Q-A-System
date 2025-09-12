import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# Get the OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Check if the API key is set
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")
#I have used FAISS because this project only required fast, accurate retrieval of relevant transcript chunks, without needing metadata filtering or cloud persistence.
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from pyngrok import ngrok
import subprocess
import time


#document ingestion: load the data from youtube transcript into my memory
def get_transcript_text(video_id: str) -> str:
    """
    Fetch YouTube transcript for a video in English, with fallback for auto-generated captions.
    Works for both dict and object return types, and shows debug info.
    """
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.list(video_id)

        # Try manual transcript first
        transcript = None
        try:
            transcript = transcript_list.find_transcript(['en', 'en-US', 'en-GB'])
        except Exception:
            # Fallback: auto-generated
            transcript = transcript_list.find_generated_transcript(['en', 'en-US', 'en-GB'])

        transcript_data = transcript.fetch() #download the transcript into our memory
        st.info(f"Fetched {len(transcript_data)} transcript snippets.")

        # Debug: show first few
        for i, snippet in enumerate(transcript_data[:5]):
            text = getattr(snippet, 'text', None) or snippet.get('text', '')
            start = getattr(snippet, 'start', None) or snippet.get('start', '?')
            st.write(f"Snippet {i+1}: '{text}' (start={start})")

        # Join all snippet texts
        if transcript_data and hasattr(transcript_data[0], 'text'):
            transcript_text = " ".join(chunk.text for chunk in transcript_data)
        else:
            transcript_text = " ".join(chunk['text'] for chunk in transcript_data)

        return transcript_text

    except TranscriptsDisabled:
        st.error("❌ Captions are disabled for this video.")
    except NoTranscriptFound:
        st.error("❌ No transcript available in English.")
    except Exception as e:
        st.error(f"❌ Error fetching transcript: {e}")
    return ""

# ========== Build Retriever ==========
def build_retriever(video_id):
    transcript = get_transcript_text(video_id)
    if not transcript.strip():
        return None, None
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)#heirachial splliting, papr->sentence->word->character
    chunks = splitter.create_documents([transcript])#each chunk is a document object with page content
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")#1536 dimensional embeddings
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})#retrieval based on cosine similarity between query embedding and stored chunk embeddings and return top 4 chunks
    return retriever, transcript

# ========== Prompt ==========
prompt = PromptTemplate(
    template="""
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, say you don't know.

{context}
Question: {question}
""",
    input_variables=['context', 'question']
)

# Format docs for context
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ========== Streamlit App ==========
st.title("YouTube Video Transcriber and Q&A")

# Input for YouTube video ID
video_id = st.text_input("Enter YouTube Video ID:", "")

if video_id:
    with st.spinner("Building retriever..."):
        retriever, transcript = build_retriever(video_id)

    if retriever:
        st.success("Retriever built successfully!")
        st.subheader("Transcript")
        st.text_area("Transcript Text", transcript, height=300)

        # Input for user question
        question = st.text_input("Ask a question about the video:")

        if question:
            with st.spinner("Fetching answer..."):
                docs = retriever.get_relevant_documents(question)
                context = format_docs(docs)

                # Use the prompt to generate an answer
                llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
                answer = llm.invoke(prompt.format(context=context, question=question)).content

            st.subheader("Answer")
            st.write(answer)
    else:
        st.error("Failed to build retriever. Please check the video ID or try again.")
