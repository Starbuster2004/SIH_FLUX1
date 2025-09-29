import streamlit as st
import pandas as pd
import itertools
import requests
import json
import time
import random

# ------------------ UI Configuration ------------------
st.set_page_config(layout="wide")
st.title("🎯 Language-Balanced Dataset Generator")
st.markdown("Generate a specific number of unique comments for each specified language.")

st.sidebar.header("🔑 API Provider Configuration")

with st.sidebar.expander("Provider Configuration (Required)", expanded=True):
    api_key1 = st.text_input("API Key", type="password", help="Your OpenRouter API key.")
    model_names1 = st.text_input(
        "Model Name(s)",
        value="google/gemini-flash-1.5, mistralai/mistral-7b-instruct",
        help="Comma-separated model names. The script will rotate through them."
    )
    base_url1 = st.text_input(
        "API Base URL",
        value="https://openrouter.ai/api/v1/chat/completions",
        help="OpenAI-compatible endpoint."
    )

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Generation Parameters")
rows_per_language = st.sidebar.number_input(
    "Rows to Generate Per Language",
    min_value=100,
    max_value=10000,
    value=2000,
    step=100,
    help="The number of unique comments to generate for each language."
)
batch_size = st.sidebar.slider("Batch Size", 5, 100, 25, help="Number of comments to generate per API call.")

generate_btn = st.button("🚀 Generate Balanced Dataset", use_container_width=True)

# ------------------ Dataset Categories ------------------
languages = ["English", "Hindi", "Marathi", "Tamil", "Telugu", "Bengali", "Gujarati", "Kannada", "Malayalam", "Punjabi"]
sentiments = ["Very Positive", "Positive", "Neutral", "Negative", "Very Negative"]
emotions = ["Trust", "Joy", "Anger", "Fear", "Sadness", "Surprise", "Neutral"]
tones = ["Formal", "Informal", "Sarcastic", "Polite", "Analytical", "Casual"]

total_to_generate = rows_per_language * len(languages)
attribute_combos = list(itertools.product(sentiments, emotions, tones))
combinations_to_generate = []

st.sidebar.info(f"This will generate **{rows_per_language}** rows for each of the **{len(languages)}** languages, for a total of **{total_to_generate}** rows.")

for lang in languages:
    cycled_attributes = itertools.cycle(attribute_combos)
    for i in range(rows_per_language):
        attrs = next(cycled_attributes)
        combinations_to_generate.append((lang, attrs[0], attrs[1], attrs[2]))

random.shuffle(combinations_to_generate)

def create_batches(data, size):
    for i in range(0, len(data), size):
        yield data[i:i + size]

# ------------------ API Call Function ------------------
# MODIFIED: Made JSON parsing more robust
def find_list_in_json(data):
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, list):
                return value
            # Recurse for nested dictionaries
            if isinstance(value, dict):
                result = find_list_in_json(value)
                if result is not None:
                    return result
    return None

def generate_batch_of_comments(batch_details: list, provider: dict) -> list:
    tasks_string = ""
    for i, (lang, sentiment, emotion, tone) in enumerate(batch_details, 1):
        tasks_string += f"{i}. A comment in {lang} with '{sentiment}' sentiment, '{emotion}' emotion, and a '{tone}' tone.\n"

    prompt = (
        f"Generate a batch of {len(batch_details)} unique and varied user comments based on the following requirements. Avoid common or cliché phrases.\n"
        f"{tasks_string}"
        "Your response MUST be a single, valid JSON array where each element is an object with a single key 'comment'.\n"
        "Example: [{\"comment\": \"This is the first comment.\"}, {\"comment\": \"This is the second comment.\"}]"
    )

    headers = {"Authorization": f"Bearer {provider['key']}", "Content-Type": "application/json", "HTTP-Referer": "https://yourapp.com", "X-Title": "Dataset Generator"}
    payload = {
        "model": provider['model'],
        "messages": [
            {"role": "system", "content": "You are a data generation expert. You strictly follow instructions and only output valid JSON with unique, creative content."},
            {"role": "user", "content": prompt}
        ],
        # REMOVED: This parameter can cause issues with some models/gateways
        # "response_format": {"type": "json_object"},
        "max_tokens": 8192,
        "temperature": 0.9
    }

    try:
        response = requests.post(provider['base_url'], headers=headers, json=payload, timeout=180) # Increased timeout
        response.raise_for_status()
        raw_response = response.json()['choices'][0]['message']['content']
        json_data = json.loads(raw_response)
        
        comments_list = find_list_in_json(json_data)
        
        if comments_list is None:
             return [f"ERROR: No list found in JSON response." for _ in batch_details]

        return [item.get('comment', 'ERROR: Missing "comment" key').strip() for item in comments_list]
    except Exception as e:
        return [f"ERROR: {str(e)}" for _ in batch_details]

# ------------------ Main Application Logic ------------------
if generate_btn:
    providers = []
    if api_key1 and model_names1 and base_url1:
        for model in [m.strip() for m in model_names1.split(',') if m.strip()]:
            providers.append({"key": api_key1, "model": model, "base_url": base_url1})

    if not providers:
        st.error("⚠️ Please configure the provider with an API Key, Model Name, and Base URL.")
    else:
        st.info(f"⏳ Generating **{total_to_generate}** unique rows. This may involve extra retries for duplicates...")
        
        data = []
        unique_comments_set = set()
        combinations_to_retry = combinations_to_generate[:]
        
        provider_cycle = itertools.cycle(providers)
        progress_bar = st.progress(0, text="Starting generation...")
        start_time = time.time()
        
        main_run_complete = False
        while len(data) < total_to_generate:
            current_run_combinations = combinations_to_retry[:]
            combinations_to_retry.clear()
            
            if not current_run_combinations:
                st.warning("Could not generate more unique comments. Stopping.")
                break

            batches = list(create_batches(current_run_combinations, batch_size))
            
            status_text = "Phase 1: Main generation" if not main_run_complete else f"Phase 2: Retrying {len(current_run_combinations)} failed/duplicate items"

            for i, batch in enumerate(batches):
                # ADDED: Update progress text BEFORE the API call
                progress_value = len(data) / total_to_generate
                progress_bar.progress(progress_value, text=f"{status_text} | Requesting batch {i+1}/{len(batches)}...")
                
                provider_to_use = next(provider_cycle)
                generated_comments = generate_batch_of_comments(batch, provider_to_use)
                
                if len(generated_comments) != len(batch):
                    generated_comments.extend(["ERROR: Mismatched count"] * (len(batch) - len(generated_comments)))

                for combination, comment in zip(batch, generated_comments):
                    if "ERROR:" not in comment and comment and comment not in unique_comments_set:
                        unique_comments_set.add(comment)
                        data.append({
                            "id": f"STB-{(len(data) + 1):05d}", "comment": comment, "language": combination[0], 
                            "sentiment": combination[1], "emotion": combination[2], "tone": combination[3], 
                            "model": provider_to_use['model']
                        })
                    else:
                        combinations_to_retry.append(combination)

                # MODIFIED: Progress bar text updated for clarity
                elapsed_time = time.time() - start_time
                rows_per_second = len(data) / elapsed_time if elapsed_time > 0 else 0
                progress_bar.progress(progress_value, text=f"{status_text} | Progress: {len(data)}/{total_to_generate} unique rows | Speed: {rows_per_second:.2f} rows/sec")

            main_run_complete = True

        st.success(f"✅ Generation complete! Generated **{len(data)}** unique rows in {time.time() - start_time:.2f} seconds.")
        df = pd.DataFrame(data)
        st.dataframe(df)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Balanced Dataset as CSV", csv, "language_balanced_dataset.csv", "text/csv", use_container_width=True)