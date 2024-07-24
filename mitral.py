import openai
import pandas as pd
import time

# Set your OpenAI API key
openai.api_key = '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'

# Load the dataset
df = pd.read_csv('samples.csv')

# Initialize an empty list to store generated labels
generated_labels = []

# Set rate limit values
max_requests_per_minute = 3
max_tokens_per_minute = 50

# Track usage counters
requests_made = 0
tokens_used = 0
count = 0 
# Iterate through each sentence and generate labels
for sentence in df['sentences']:
    # Check if approaching rate limits
    
    if requests_made >= max_requests_per_minute or tokens_used >= max_tokens_per_minute:
        # Introduce a more conservative sleep time to stay well within rate limits
        sleep_time = 2  # Sleep for 10 seconds
        print(f"Rate limit reached. Sleeping for {sleep_time} seconds...")
        time.sleep(sleep_time)
        requests_made = 0
        tokens_used = 0

    # Make API request using the v1/chat/completions endpoint
    prompt = f"Label each sentence with one of these four labels(Person, Place, Thing, Idea) if it does not fit any of these label as (not applicable): {sentence}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    )

    # Update counters
    requests_made += 1
    tokens_used += response['usage']['total_tokens']

    # Extract label from response
    generated_label = response['choices'][0]['message']['content'].strip()
    generated_labels.append(generated_label)

# Add generated labels to the dataframe
df['generated_labels'] = generated_labels

# Save the labeled data to a new CSV file
df.to_csv('generated_labels_dataset.csv', index=False)
