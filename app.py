import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Initialize ChromaDB
client = chromadb.Client()
collection_name = "email_responses"
collection = client.create_collection(name=collection_name)

# Initialize the Sentence Transformer model for embeddings
embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

# Initialize sentiment analyzer
nltk.download('vader_lexicon', quiet=True)
sentiment_analyzer = SentimentIntensityAnalyzer()

# Load a language model using HuggingFace pipeline
model_name = "gpt2"
text_generation_pipeline = pipeline("text-generation", model=model_name)
llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

def analyze_sentiment(email_content):
    """Analyze the sentiment of the email content."""
    scores = sentiment_analyzer.polarity_scores(email_content)
    return scores

def generate_auto_reply(email_content):
    """Generate an auto-reply based on the email content."""
    sentiment_scores = analyze_sentiment(email_content)

    # Create a context for the response
    if sentiment_scores['compound'] >= 0.05:
        sentiment = "positive"
    elif sentiment_scores['compound'] <= -0.05:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    prompt = f"The sentiment of the email is {sentiment}. Write a suitable auto-reply."
    response = llm.invoke(prompt)
    return response

def store_email_response(email_content, auto_reply):
    """Store the email and its auto-reply in ChromaDB."""
    collection.add(
        documents=[email_content],
        metadatas=[{"auto_reply": auto_reply}],
        ids=[f"email_{collection.count() + 1}"]
    )

# Example usage
if __name__ == "__main__":
    email_content = "I'm really happy with the service I received. Thank you!"
    auto_reply = generate_auto_reply(email_content)
    print(f"Auto-reply: {auto_reply}")

    # Store the email and auto-reply in the database
    store_email_response(email_content, auto_reply)
