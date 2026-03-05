import os
import pandas as pd
import numpy as np
import time
from dotenv import load_dotenv
from litellm import completion
from sentence_transformers import SentenceTransformer
import faiss

# Load environment variables
load_dotenv()

# 1. Load Data
DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "structured/daily_sales.csv")
UNSTRUCTURED_DIR = os.path.join(DATA_DIR, "unstructured")

df_sales = pd.read_csv(CSV_PATH)

product_pages = []
for filename in os.listdir(UNSTRUCTURED_DIR):
    if filename.endswith(".txt"):
        with open(os.path.join(UNSTRUCTURED_DIR, filename), "r") as f:
            product_pages.append({
                "product_id": filename.split("_")[0],
                "filename": filename,
                "content": f.read()
            })

# 2. Setup Vector Search
model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [p["content"] for p in product_pages]
embeddings = model.encode(texts)

d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(np.array(embeddings).astype("float32"))

def get_llm_response(prompt, model_name="groq/llama-3.3-70b-versatile", retries=3):
    for i in range(retries):
        try:
            response = completion(
                model=model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            if "rate_limit" in str(e).lower() and i < retries - 1:
                print(f"Rate limit hit. Retrying in 25 seconds...")
                time.sleep(25)
                continue
            return f"LLM Error: {str(e)}"

def retrieve_text(query, k=2):
    query_vector = model.encode([query])
    D, I = index.search(np.array(query_vector).astype("float32"), k)
    results = [product_pages[i] for i in I[0]]
    context = ""
    for doc in results:
        context += f"\n--- Source File: {doc['filename']} ---\n{doc['content']}\n"
    return context

def retrieve_csv(query):
    columns = df_sales.columns.tolist()
    unique_products = df_sales['product_name'].unique().tolist()
    unique_categories = df_sales['category'].unique().tolist()
    sample_data = df_sales.head(3).to_string()
    
    prompt = f"""
    You are a expert data analyst. 
    Dataframe: `df_sales`
    Columns: {columns}
    Valid Categories: {unique_categories}
    Valid Products: {unique_products}
    
    Task: Write a SINGLE Python code snippet to answer the query: "{query}"
    
    Rules:
    1. Return ONLY the code snippet. No markdown backticks.
    2. Store the answer in a variable `result`.
    3. Use ONLY the product names and categories listed above. DO NOT HALLUCINATE brand names like Nike.
    4. For the 'highest units sold', group by 'product_name' and sum 'units_sold'.
    
    Code:
    """
    code = get_llm_response(prompt).strip().strip("`").replace("python", "").strip()
    
    local_vars = {"df_sales": df_sales, "pd": pd}
    try:
        exec(code, {}, local_vars)
        return str(local_vars.get("result", "No result calculated."))
    except Exception as e:
        return f"Error executing data analysis: {str(e)}"

def route_query(query):
    prompt = f"""
    You are a query router for a retail RAG system.
    SOURCES:
    1. `CSV`: Structured sales analytics (revenue, units sold, regions, categories, product names).
    2. `TEXT`: Unstructured product descriptions, features, specifications, and customer reviews.
    
    Rule:
    - If the query requires numerical facts/analytics, use `CSV`.
    - If it requires qualitative features/reviews, use `TEXT`.
    - If it's a join (e.g., comparing sales vs reviews), use `BOTH`.
    
    Query: {query}
    Category (CSV/TEXT/BOTH):
    """
    res = get_llm_response(prompt).strip().upper()
    if "CSV" in res: return "CSV"
    if "TEXT" in res: return "TEXT"
    return "BOTH"

def multi_source_rag(query):
    route = route_query(query)
    print(f"Routing: {route}")
    
    context = f"Query Route: {route}\n"
    if route in ["CSV", "BOTH"]:
        csv_context = retrieve_csv(query)
        context += f"\n[ANALYTICS - Structured Sales Data]:\n{csv_context}\n"
        
    if route in ["TEXT", "BOTH"]:
        text_context = retrieve_text(query)
        context += f"\n[PRODUCT_SPECS - Unstructured Documentation & Reviews]:\n{text_context}\n"
        
    final_prompt = f"""
    You are a professional retail assistant. Answer the user query using ONLY the provided context.
    
    STRICT RULES:
    1. Label your data sources clearly (e.g., 'Based on Sales Analytics...' or 'According to Product Reviews...').
    2. If the query asks for a comparison, explicitly use data from both [ANALYTICS] and [PRODUCT_SPECS].
    3. Cite specific .txt files for qualitative info.
    
    Context:
    {context}
    
    Query: {query}
    
    Answer:
    """
    return get_llm_response(final_prompt)

if __name__ == "__main__":
    questions = [
        "What is the total revenue for the category 'Electronics'?",
        "What are the key features of the 'ChefMaster Induction Cooktop'?",
        "Which product has the highest units sold in the 'South' region?",
        "Based on customer reviews, what are the main pros and cons of the 'Wireless Bluetooth Headphones'?",
        "Compare the total revenue of 'Coffee Maker' with its average customer rating from the reviews.",
        "Which region generated the most revenue from 'Home & Kitchen' products, and what is the top-selling product in that category?"
    ]

    results = []
    for i, q in enumerate(questions):
        print(f"\n[{i+1}/{len(questions)}] Processing: {q}")
        answer = multi_source_rag(q)
        results.append(f"# Question {i+1}: {q}\n\n{answer}\n\n" + "="*80 + "\n\n")
        
        if i < len(questions) - 1:
            print("Waiting 30 seconds to avoid rate limits...")
            time.sleep(30)

    with open("part2_results.txt", "w") as f:
        f.writelines(results)
    print("\nPart 2 results saved to part2_results.txt")
