import os
import numpy as np
import pandas as pd
import streamlit as st
import snowflake.connector
from google import genai
from dotenv import load_dotenv
import time

load_dotenv()

EMBEDDING_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.5-flash-lite"
NEW_REVIEWS = 500
TOK_K = 5
CACHE_FILE = "review_embeddings.parquet"

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
def read_reviews_from_snowflake():
    conn = snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
    )

    query = f"""
        SELECT REVIEW_ID, CITY, RATING, COMMENT
        FROM ZOMATO.STAGING.STG_REVIEWS
        SAMPLE ({NEW_REVIEWS} ROWS)
    """
    df = conn.cursor().execute(query).fetch_pandas_all()
    conn.close()

    df.columns = [col.lower() for col in df.columns]
    return df




def embed(texts):
    all_embeddings = []
    batch_size = 10

    for i in range(0, len(texts), batch_size):

        batch = [
            str(t) if pd.notna(t) else ""
            for t in texts[i:i + batch_size]
        ]

        while True:
            try:
                response = client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=batch
                )

                all_embeddings.extend(
                    [item.values for item in response.embeddings]
                )

                break

            except Exception as e:
                if "429" in str(e):
                    print("Gemini quota reached. Waiting 65 seconds...")
                    time.sleep(65)
                else:
                    raise

        print(
            f"Embedded {len(all_embeddings)}/{len(texts)} reviews"
        )

        # Wait after every 100 reviews
        if len(all_embeddings) % 100 == 0 and len(all_embeddings) < len(texts):
            print("Waiting 65 seconds for quota reset...")
            time.sleep(65)

    return all_embeddings

@st.cache_data()
def load_reviews():
    if os.path.exists(CACHE_FILE):
        return pd.read_parquet(CACHE_FILE)

    df = read_reviews_from_snowflake()
    df['embedding'] = embed(df['comment'].tolist())
    df.to_parquet(CACHE_FILE)
    return df

st.title("Chat with your Zomato Reviews")
st.caption(f"Searching {NEW_REVIEWS} review, answering with {CHAT_MODEL} model")

def consine_simiarity(vec_a, vec_b):
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

def find_similar_reviews(question, df):
    question_vector = embed([question])[0]

    scores = []
    for review_vector in df['embedding']:
        scores.append(consine_simiarity(question_vector, review_vector))

    df = df.copy()
    df['score'] = scores
    return df.nlargest(TOK_K, 'score')

def ask_llm(question, top_reviews):
    context = ""

    for _, row in top_reviews.iterrows():
        context += (
            f"({row['city']}, {row['rating']} stars) "
            f"{row['comment']}\n"
        )

    system_prompt = (
        "Answer ONLY using the customer reviews provided. "
        "Be concise. If the reviews don't cover it, say so."
    )

    user_prompt = f"""
    {system_prompt}

    Question: {question}

    Reviews:
    {context}
    """

    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=user_prompt,
        config={"temperature": 0.2}
    )

    return response.text
    
review_df = load_reviews()

question = st.text_input("Ask a question about your reviews:",
                         placeholder="e.g. What are the most common complaints about delivery?")

if question:
    top_reviews = find_similar_reviews(question, review_df)
    answer = ask_llm(question, top_reviews)

    st.markdown(f"**Answer:**")
    st.write(answer)

    with st.expander("Reviews used to build this answer"):
        st.dataframe(top_reviews[['city', 'rating', 'comment']], hide_index=True)