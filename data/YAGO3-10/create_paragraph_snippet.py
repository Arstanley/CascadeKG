import openai
import pandas as pd
import json
import time
from tqdm import tqdm

# Load JSONL data into a Pandas DataFrame
file_name = "./data/YAGO3-10/event_triggered_kg_updates.jsonl"
jsonObj = pd.read_json(path_or_buf=file_name, lines=True)

# OpenAI API setup
api_key = "sk-423dc1c07533488b8fee62319b6a083e"
client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

# Function to query OpenAI API
def get_direct_results(trigger_event, added_edges, removed_edges, event_type):
    """Sends a query to the OpenAI API to find direct results of an event."""
    event_str = f'"({trigger_event[0], trigger_event[1], trigger_event[2]})"'

    prompt = f"""Write a short paragraph to describe the following event: 
    {{
        "trigger_event": {"Adding" if event_type == "add" else "Deleting"} {event_str},
        "added_edges": {added_edges},
        "removed_edges": {removed_edges}
    }}"""  
    print(prompt)

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are an expert in analyzing events."},
                {"role": "user", "content": prompt},
            ],
            stream=False,
        )

        print(response)

        # Extract response text and parse the events
        output = response.choices[0].message.content.strip()
        return output  # Returning the raw response for now; can be processed further

    except Exception as e:
        print(f"Error: {e}")
        return []

# Output file path
output_file = "./data/YAGO3-10/event_triggered_kg_updates_paragraph.jsonl"

# Open the file in append mode to write results line by line
with open(output_file, "a") as f:
    for _, row in tqdm(jsonObj.iterrows(), total=len(jsonObj)):
        trigger_event = row["event"] # ["s", "r", "t"] 
        added_edges = row["added_edges"] 
        deleted_edges = row["deleted_edges"]
        event_type = row['event_type']

        response = get_direct_results(trigger_event, added_edges, deleted_edges, event_type)

        # Prepare the new JSON object with the result
        new_entry = row.to_dict()
        new_entry["paragraph"] = response

        # Write it to the file line by line
        f.write(json.dumps(new_entry) + "\n")

        time.sleep(1)  # To avoid hitting API rate limits

print(f"Results saved to {output_file}")