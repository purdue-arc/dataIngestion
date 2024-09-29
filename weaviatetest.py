import weaviate
import weaviate.classes as wvc
import os
import requests
import json

# Best practice: store your credentials in environment variables
wcd_url = "https://sxi7vehirjur8cxqlehmog.c0.us-east1.gcp.weaviate.cloud"
wcd_api_key = "scmKsTnfTazVo4eCEmYi7RmrdfigCtyJdYDg"
openai_api_key = "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"

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=wcd_url,                   # Replace with your Weaviate Cloud URL
    auth_credentials=wvc.init.Auth.api_key(wcd_api_key),    # Replace with your Weaviate Cloud key
    headers={"X-OpenAI-Api-Key": openai_api_key}            # Replace with appropriate header key/value pair for the required API
)

try:
    questions = client.collections.create(
        name="Question",
        vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_openai(),  # If set to "none" you must always provide vectors yourself. Could be any other "text2vec-*" also.
        generative_config=wvc.config.Configure.Generative.openai()  # Ensure the `generative-openai` module is used for generative queries
    )

    resp = requests.get('https://raw.githubusercontent.com/weaviate-tutorials/quickstart/main/data/jeopardy_tiny.json')
    data = json.loads(resp.text)  # Load data

    question_objs = list()
    for i, d in enumerate(data):
        question_objs.append({
            "answer": d["Answer"],
            "question": d["Question"],
            "category": d["Category"],
        })

    questions = client.collections.get("Question")
    questions.data.insert_many(question_objs)


finally:
    client.close()  # Close client gracefully