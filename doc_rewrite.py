import openai
import os
import requests
import json

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")


# Load Markdown file
markdown_file_path = "inter_grids.md"
with open(markdown_file_path, "r") as f:
    markdown_content = f.read()


messages = []
messages.append({"role": "system", "content": "The following document is the first half of a well-written article. However it needs to be re-written for different audience. Reorganise and re-write it for a scientific audience in professional academic English, to a very high standard, under headings of Introduction, Literature Review, Methodology, Findings, Discussion, Conclusion. Very important: Write at least 1,200 words, keeping as much of the original text, including references, as possible."})
messages.append({"role": "user", "content": markdown_content})
response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            max_tokens=2000,
            temperature=0.7
        )

# Parse response
generated_text = ''
if response != 0:
    generated_text = response.choices[0].message['content'].strip()

# Write response to a new text file
output_file_path = "inter_grids_rewrite.md"
with open(output_file_path, "w") as f:
    f.write(generated_text)

print(f"Re-written to {output_file_path}")
