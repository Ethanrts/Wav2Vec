import openai
import pandas as pd
import time

# Set your OpenAI API key
openai.api_key = ''

# Load the dataset
df = pd.read_csv('samples.csv')

# Initialize an empty list to store labels
labels = []

# Set rate limit values
max_requests_per_minute = 3
max_tokens_per_minute = 50

# Track usage counters
requests_made = 0
tokens_used = 0

# Iterate through each sentence and generate labels
for sentence in df['sentences']:
    # Check if approaching rate limits
    if requests_made >= max_requests_per_minute or tokens_used >= max_tokens_per_minute:
        # Introduce a more conservative sleep time to stay well within rate limits
        sleep_time = 10  # Sleep for 10 seconds
        print(f"Rate limit reached. Sleeping for {sleep_time} seconds...")
        time.sleep(sleep_time)
        requests_made = 0
        tokens_used = 0

    # Make API request
    prompt = f"Label the following sentence: {sentence}"
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
    label = response['choices'][0]['message']['content'].strip()
    labels.append(label)

# Add labels to the dataframe
df['label'] = labels

# Save the labeled data to a new CSV file
df.to_csv('labeled_samples.csv', index=False)
