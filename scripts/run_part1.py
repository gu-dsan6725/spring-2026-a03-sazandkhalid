import os
import subprocess
import time
from dotenv import load_dotenv
from litellm import completion

# Load environment variables from .env file
load_dotenv()

def get_llm_response(prompt, model="groq/llama-3.3-70b-versatile", retries=3):
    for i in range(retries):
        try:
            response = completion(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            if "rate_limit" in str(e).lower() and i < retries - 1:
                print(f"Rate limit hit. Retrying in 20 seconds... (Attempt {i+1}/{retries})")
                time.sleep(20)
                continue
            return f"LLM Error: {str(e)}"

def classify_query(query):
    prompt = f"""
    Classify the following query about a codebase into one of these categories:
    - `Structural`: Exploring project layout, finding file names, or checking directory contents.
    - `Business Logic`: Understanding how specific features or services are implemented.
    - `API/Endpoints`: Finding routes, endpoints, or API definitions.
    - `Security/Auth`: Examining authentication, authorization, or token logic.
    - `Dependencies`: Checking for libraries, external packages, or environment setup.
    
    Query: {query}
    Category:
    """
    return get_llm_response(prompt).strip().strip('`').replace('Category: ', '').strip()

PROJECT_FACTS = """
Codebase Root: `mcp-gateway-registry/`
Stack: Python, FastAPI.
CRITICAL: All commands MUST use the `mcp-gateway-registry/` prefix.
Example: `grep -r "auth" mcp-gateway-registry/auth_server/`
Key Paths:
- `mcp-gateway-registry/auth_server/`
- `mcp-gateway-registry/registry/`
- `mcp-gateway-registry/servers/`
- `mcp-gateway-registry/uv.lock` (for dependencies)
"""

def generate_bash_command(query, context="", step="research", codebase_path="mcp-gateway-registry"):
    if step == "research":
        system_msg = f"Step 1: RESEARCH. {PROJECT_FACTS}\nGenerate a command to find relevant files."
    else:
        system_msg = f"Step 2: INSPECTION. {PROJECT_FACTS}\nBased on previous output, read specific files.\nResearch Output: {context[:1000]}"
        
    prompt = f"""
    You are an expert developer assistant. {system_msg}
    
    Rules:
    1. Use `grep -r` to find patterns, `ls -R` for structure.
    2. In 'Inspection', `cat` ONLY small, specific files.
    3. Return ONLY the bash command.
    
    User Question: {query}
    Bash Command:
    """
    cmd = get_llm_response(prompt).strip().strip('`').replace('bash', '').strip()
    return cmd

def execute_bash_command(command, timeout=45):
    try:
        print(f"Executing: {command}")
        # Sanitize command - avoid destructive or complex pipes that might hang
        if "rm " in command or "mv " in command or ">" in command:
            return "Error: Command contains forbidden operations."
            
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout)
        
        output = result.stdout
        if result.returncode != 0:
            output += f"\nError: {result.stderr}"
        
        if not output.strip():
            return "Command executed but returned no output. This might mean no matches were found."
            
        # Truncate if too large for LLM context (Stay well under TPM limits)
        if len(output) > 8000:
            output = output[:4000] + "\n... [TRUNCATED DUE TO SIZE] ...\n" + output[-4000:]
            
        return output
    except subprocess.TimeoutExpired:
        return "Error: Command timed out. Try a more specific search (e.g. searching for a unique class name or filename)."
    except Exception as e:
        return f"Exception occurred: {str(e)}"

def generate_answer(query, context):
    prompt = f"""
    You are an expert developer assistant. Based on the retrieved context from the `mcp-gateway-registry` directory, answer the user's question.
    
    STRICT RULES:
    1. GROUNDING: Answer ONLY based on the provided context. If a file path isn't in the context, DO NOT INVET ONE. If you are unsure, say "The provided context does not mention the specific path for X".
    2. CITATIONS: Cite real files and directories found in the context (e.g., `mcp-gateway-registry/src/auth.py`). 
    3. MISSING DATA: If the context is empty or says "timeout", explain clearly that no direct information was found and suggest what files a developer should likely look at based on the directory structure.
    4. ACCURACY: Focus on being technically accurate. Don't guess.
    
    Context:
    {context}
    
    Question: {query}
    
    Answer:
    """
    return get_llm_response(prompt)

def code_qa_rag(query):
    # 1. Classify (20 pts)
    category = classify_query(query)
    print(f"Query Category: {category}")
    
    # 2. Iterative Retrieval (40 pts)
    # Step A: Research
    research_cmd = generate_bash_command(query, step="research")
    research_output = execute_bash_command(research_cmd)
    
    # Step B: Inspection
    inspection_cmd = generate_bash_command(query, context=research_output, step="inspection")
    inspection_output = execute_bash_command(inspection_cmd)
    
    # Combine for final answer
    full_context = f"Category: {category}\n\n[RESEARCH OUTPUT]\n{research_output}\n\n[INSPECTION OUTPUT]\n{inspection_output}"
    
    # 3. Grounded Answer (40 pts)
    answer = generate_answer(query, full_context)
    return answer

if __name__ == "__main__":
    questions = [
        "What Python dependencies does this project use?",
        "What is the main entry point file for the registry service?",
        "What programming languages and file types are used in this repository? (e.g., Python, TypeScript, YAML, JSON, Dockerfile, etc.)",
        "How does the authentication flow work, from token validation to user authorization?",
        "What are all the API endpoints available in the registry service and what scopes do they require?",
        "How would you add support for a new OAuth provider (e.g., Okta) to the authentication system? What files would need to be modified and what interfaces must be implemented?"
    ]

    results = []
    for i, q in enumerate(questions):
        print(f"\n[{i+1}/{len(questions)}] Processing: {q}")
        answer = code_qa_rag(q)
        results.append(f"# Question {i+1}: {q}\n\n{answer}\n\n" + "="*80 + "\n\n")
        
        # Avoid rate limits even more strictly
        if i < len(questions) - 1:
            print("Waiting 30 seconds to avoid rate limits...")
            time.sleep(30)

    with open("part1_results.txt", "w") as f:
        f.writelines(results)
    print("\nRefined and grounded results saved to part1_results.txt")
