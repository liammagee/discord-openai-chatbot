import openai
import os
import requests
import json

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")


# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    weather_info = {
        "location": location,
        "temperature": "72",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }
    return json.dumps(weather_info)

messages = []
messages.append({"role": "system", "content": "Provide helpful advice about the weather."})
messages=[{"role": "user", "content": "What's the weather like in Boston?"}]

def run_conversation():
    response = openai.ChatCompletion.create(
                # model="gpt-3.5-turbo-0613 ",
                model="gpt-4-0613",
                messages=messages,
                max_tokens=2000,
                temperature=0.7,
                functions=[
                            {
                                "name": "get_current_weather",
                                "description": "Get the current weather in a given location",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "location": {
                                            "type": "string",
                                            "description": "The city and state, e.g. San Francisco, CA",
                                        },
                                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                                    },
                                    "required": ["location"],
                                },
                            }
                        ],
                function_call="auto",            
            )


    message = response["choices"][0]["message"]

    print(message)

    # Step 2, check if the model wants to call a function
    if message.get("function_call"):
        function_name = message["function_call"]["name"]

        # Step 3, call the function
        # Note: the JSON response from the model may not be valid JSON
        function_response = get_current_weather(
            location=message.get("location"),
            unit=message.get("unit"),
        )

        # Step 4, send model the info on the function call and function response
        second_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=[
                {"role": "user", "content": "What is the weather like in boston?"},
                message,
                {
                    "role": "function",
                    "name": function_name,
                    "content": function_response,
                },
            ],
        )
        return second_response

print(run_conversation())


