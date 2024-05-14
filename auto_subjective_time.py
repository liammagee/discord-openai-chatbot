from cmd import PROMPT
import os

import requests
from openai import OpenAI
import anthropic

import asyncio
import discord
from discord.ext import commands
from discord import app_commands
import sys
import json
from urllib.parse import quote
from transformers import AutoTokenizer
from datetime import datetime

BOT_ON = True


intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)
guild_id = os.environ.get("DISCORD_GUILD_ID")
channel_last = None


# Load a pre-trained tokenizer (for example, the GPT-2 tokenizer)
tokenizer = AutoTokenizer.from_pretrained('gpt2')

client_openai = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key = os.environ.get("OPENAI_API_KEY")
)
image_generation = False

all_models = [
    "gpt-4-1106-preview",
    "gpt-4",
    "claude-3",
]


claude_api_key = os.getenv('ANTHROPIC_API_KEY')

client_anthropic = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key=claude_api_key,
)



# Define a dictionary to hold the parameters and their default values
params_gpt = {
    "model_id" : 0,
    "model_random" : True,
    "temperature" : 0.7,
    "top_p" : 1.0,
    "frequency_penalty" : 0.0,
    "presence_penalty" : 0.0,
    "summary_level" : 20,
    "max_size_dialog" : 60,
    "prompt_for_bot" : "",
    "channel_reset" : ""
}
bot_name="?"
stop_sequences = ["\n", "."]
channels = ['gpt-3', 'gpt-3-tests']

max_size_dialog = 10
bot_running_dialog = []
user_refs = []
summaries = {}
command_for_bot="Zhang: "
prompt_suffix = """{0}: {1}
{2}:"""


sleep_counter = 10

async def chat_prompt(author_name, prompt, model='gpt-4', stopSequences=["You:", "Zhang:"]):
    try:
        prompt_for_bot = params_gpt["prompt_for_bot"]
        if type(prompt_for_bot) == list:
            prompt_for_bot = "\n".join(prompt_for_bot).strip()
        prompt_for_bot = eval(f'f"""{prompt_for_bot}"""')

        messages = []
        messages.append({"role": "system", "content": prompt_for_bot})
        history = build_history()
        for item in history:
            messages.append(item)
        messages.append({"role": "user", "content": f"#{author_name}: ${prompt}"})

        if model == "claude-3":
            message = client_anthropic.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=params_gpt["max_tokens"],
                temperature=params_gpt["temperature"],
                system=prompt_for_bot,
                messages=messages
            )
            return message.content
        
        else:
            response = client_openai.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=params_gpt["max_tokens"],
                temperature=params_gpt["temperature"],
            )
            if response != 0:
                return response.choices[0].message
        
        return {"role": "system", "content": "No response"}
    except Exception as e:
        print(e)
        return {"role": "system", "content": "Looks like ChatGPT is down!"}


async def summarise_autobiography(model="gpt-4", num_results=1, stopSequences=["You:", "Zhang:"], topKReturn=2):

    try:
        prompt_for_bot = params_gpt["prompt_for_bot"]
        if type(prompt_for_bot) == list:
            prompt_for_bot = "\n".join(prompt_for_bot).strip()
        prompt_for_bot = eval(f'f"""{prompt_for_bot}"""')

        messages = []
        model = model.strip()
        if model != "claude-3":
            messages.append({"role": "system", "content": prompt_for_bot})
        history = build_history()

        builder = ''
        for message in history:
            builder += message['content'] + "\n"

        prompt = "Convert this into an autobiographical summary: \n\n\"" + builder + "\"\n\n"
        messages.append({"role": "user", "content": prompt})

        if model == "claude-3":
            print("got here")
            message = client_anthropic.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=params_gpt["max_tokens"],
                temperature=params_gpt["temperature"],
                system=prompt_for_bot,
                messages=messages
            )
            print(message.content)
            return message.content
        
        else:
            response = client_openai.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=params_gpt["max_tokens"],
                temperature=params_gpt["temperature"],
            )
            if response != 0:
                return response.choices[0].message
            return {"role": "assistant", "content": "No response"}
    except Exception as e:
        print(e)
        return {"role": "assistant", "content": "Error!"}    
    


async def fetch(url):
    response = requests.get(url)
    return response

async def retrieve(url):
    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(None, fetch, url)
    return response


@client.event
async def on_ready():
    # await tree.sync(guild=discord.Object(guild_id))
    client.loop.create_task(periodic_task())
    print(f'We have logged in as {client.user} to {len(client.guilds)} guilds.')


async def print_history(channel, limit):
    builder = ""
    async for message in channel.history(limit=limit):
        builder += message.author.name + ": " +message['content'] + "\n"
    return builder

def prepare_summary(history):
    return history.replace("\n", " ").replace("\"", "'")



@tree.command(name = "get", description = "Show parameters", guild=discord.Object(guild_id)) #Add the guild ids in which the slash command will appear. If it should be in all, remove the argument, but note that it will take some time (up to an hour) to register the command if it's for all guilds.
async def get_params(interaction: discord.Interaction):
    response = "Current parameters:\n"
    for key, value in params_gpt.items():
        response += f"{key}: {value}\n"
    response += f"bot_running_dialog size: {len(bot_running_dialog)}\n"
    response += f"stop_sequences: {stop_sequences}\n"
    response += f"channels: {channels}\n"
    response += f"summaries: {summaries}\n"
    history = create_prompt("", "")
    # response += f"history: {history}\n"
    print(history)

    await interaction.response.send_message(response)

@tree.command(name = "set", description = "Set parameters", guild=discord.Object(guild_id)) #Add the guild ids in which the slash command will appear. If it should be in all, remove the argument, but note that it will take some time (up to an hour) to register the command if it's for all guilds.
async def set_params(interaction,
                     prompt_for_bot: str = None,
                     model_id: int = None,
                     temperature: float = None,
                     top_p: float = None,
                     frequency_penalty: float = None,
                     presence_penalty: float = None,
                     summary_level: int = None,
                     max_size_dialog: int = None,
                     channel_reset: str = None):
    global bot_running_dialog
    if prompt_for_bot is not None:
        params_gpt["prompt_for_bot"] = prompt_for_bot
    if model_id is not None:
        model = all_models[model_id]
        params_gpt["model_id"] = str(model_id)
        params_gpt["model"] = model
    if temperature is not None:
        params_gpt["temperature"] = str(temperature)
    if top_p is not None:
        params_gpt["top_p"] = top_p
    if frequency_penalty is not None:
        params_gpt["frequency_penalty"] = frequency_penalty
    if presence_penalty is not None:
        params_gpt["presence_penalty"] = presence_penalty
    if summary_level is not None:
        params_gpt["summary_level"] = summary_level
    if max_size_dialog is not None:
        bot_running_dialog = bot_running_dialog[:max_size_dialog]
        params_gpt["max_size_dialog"] = max_size_dialog
    if channel_reset is not None:
        # summaries[channel.id] = ""
        params_gpt["channel_reset"] = channel_reset
    await interaction.response.send_message("Parameters updated.")


@tree.command(name = "reset", description = "Reset memory", guild=discord.Object(guild_id)) #Add the guild ids in which the slash command will appear. If it should be in all, remove the argument, but note that it will take some time (up to an hour) to register the command if it's for all guilds.
async def reset(interaction: discord.Interaction):
    bot_running_dialog.clear()
    summaries = {}
    await interaction.response.send_message("Summary and history have been reset!")


@tree.command(name = "clear_chat", description = "Clear Chat history", guild=discord.Object(guild_id)) #Add the guild ids in which the slash command will appear. If it should be in all, remove the argument, but note that it will take some time (up to an hour) to register the command if it's for all guilds.
async def clear_chat(interaction: discord.Interaction, limit: int = 0):
    channel = client.get_channel(interaction.channel_id)
    await channel.purge(limit=limit)
    await interaction.response.send_message(f"{limit} responses cleared from chat.")


def build_history():
    # GPT limit, minus max response token size
    token_limit = 16097 - params_gpt["max_tokens"]

    # Tokenize the string to get the number of tokens
    tokens_len = 0
    
    # Iterate through the list in reverse order
    bot_running_dialog_limited = []
    for item in reversed(bot_running_dialog):
        tokens_item = tokenizer(str(item) + "\n", return_tensors='pt')
        tokens_item_len = tokens_item.input_ids.size(1)
        if tokens_len + tokens_item_len < token_limit:
            tokens_len = tokens_len + tokens_item_len
            bot_running_dialog_limited.insert(0, item)
        else:
            break
    # Construct the dialog history
    return bot_running_dialog_limited


def create_prompt(author_name, prompt):
    bot_running_dialog_limited = build_history()

    # Prompt suffix formatting
    ps = prompt_suffix.format(author_name, prompt, bot_name).rstrip()

    # Construct the dialog history
    contents = [d['content'] for d in bot_running_dialog_limited]
    history = "\n".join(contents).strip()
    
    # Guesstimate 
    ai_prompt = f"{prompt_for_bot}\n\nCurrent conversation: {summary}\n\n{history}\n{ps}"
    return ai_prompt


@client.event
async def on_message(message):
    global bot_running_dialog
    global prompt_for_bot
    global user_refs
    global summary
    global channel_last

    model = params_gpt["model"]
    max_size_dialog = params_gpt["max_size_dialog"]

    if message.author == client.user:
        return
    
    if message.channel.name not in channels:
        return

    data = message.content
    channel = message.channel
    channel_last = channel
    author = message.author
    mentions = message.mentions
    author_name = author.name
    if author.nick is not None:
        if author.nick not in user_refs:
            user_refs.append(author.nick)
        author_name = author.nick
    elif author.name not in user_refs:
        user_refs.append(author.name)

    prompt_for_bot = params_gpt["prompt_for_bot"]
    prompt_for_bot = eval(f'f"""{prompt_for_bot}"""')

    if data.startswith('CHANNEL_RESET'):
        summaries[channel.id] = ""
        await message.channel.send('Summary reset for channel {0}'.format(channel.id))
    elif BOT_ON:

        # Simulate typing for 3 seconds
        async with channel.typing():
            prompt = data
            threshold = 0


            member_stops = [x + ":" for x in user_refs]

            # For debugging
            result = "[NO-GPT RESPONSE]"
            try:
                if prompt.startswith("SUMMARISE"):
                    reply = await summarise_autobiography("gpt-4")
                else:
                    reply = await chat_prompt(author_name, prompt, stopSequences=stop_sequences + member_stops)

                result = reply.content.strip()
            except Exception as e:
                print(e)
                result = 'So sorry, dear User! ChatGPT is down.'
            
            if result != "":
                dialog_instance = prompt_suffix.format(author_name, prompt, bot_name).rstrip() + ' ' + result
                if len(bot_running_dialog) >= max_size_dialog:
                    bot_running_dialog.pop(0)
                    bot_running_dialog.pop(0)
                bot_running_dialog.append({"role": "user", "content": prompt})
                bot_running_dialog.append({"role": "assistant", "content": result})

                result = '{0}'.format(result)


                max_length = 2000
                chunks = []
                # Split string into words
                words = result.split()

                # Initialize the first chunk
                current_chunk = words[0]

                # Iterate through the rest of the words and add them to the current chunk
                for word in words[1:]:
                    # Check if adding the word to the current chunk will make it too long
                    if len(current_chunk) + len(word) + 1 > max_length:
                        # If the current chunk is too long, add it to the list of chunks and start a new chunk
                        chunks.append(current_chunk)
                        current_chunk = word
                    else:
                        # If the current chunk is not too long, add the word to the current chunk
                        current_chunk += ' ' + word

                # Add the last chunk to the list of chunks
                chunks.append(current_chunk)

                for chunk in chunks:
                    await message.reply(chunk)

            else:
                await message.channel.send("Sorry, couldn't reply")

def str_to_bool(s: str) -> bool:
    if s is None:
        return False
    elif s.lower() == 'true':
        return True
    elif s.lower() == 'false':
        return False
    else:
        raise ValueError(f"Cannot convert '{s}' to a boolean value")




async def periodic_task():
    global bot_running_dialog
    global channel_last
    global sleep_counter
    
    channel_ids = params_gpt['channel_ids']
    channel_last_id = int(channel_ids[0])
    print(channel_last_id)
    channel_last = (client.get_channel(channel_last_id) or await client.fetch_channel(channel_last_id))
    print(channel_last)


    while True:
        # Get the current time
        now = datetime.now()

        # Format the time into a human-readable string
        formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
        formatted_time = f"Time: {formatted_time}"

        print(formatted_time)

        if channel_last:

            await channel_last.send(formatted_time)         

            prompt = formatted_time
            result = ''
            try:
                reply = await chat_prompt("Time", formatted_time, stopSequences=stop_sequences)
                result = reply.content.strip()
            except Exception as e:
                print(e)
                result = 'So sorry, dear User! ChatGPT is down.'
            
            if result != "":
                if len(bot_running_dialog) >= max_size_dialog:
                    bot_running_dialog.pop(0)
                    bot_running_dialog.pop(0)
                bot_running_dialog.append({"role": "user", "content": prompt})
                bot_running_dialog.append({"role": "assistant", "content": result})


            await channel_last.send(result)
        else:
            print("No channel_last_id")
        await asyncio.sleep(sleep_counter)  # sleep for 20 seconds


def main():
    subject = f'settings_{sys.argv[2]}.json'
    with open(subject, "r") as read_file:
        subject_json = json.loads(read_file.read())
    discord_token_env_var = subject_json['discord_token_env_var']
    discord_token = os.environ.get(discord_token_env_var)
    bot_name = subject_json['name']
    guild_id = subject_json['guild_id']
    channels = subject_json['channels']
    channel_ids = subject_json['channel_ids']

    image_generation = str_to_bool(subject_json['image_generation'])

    params_gpt["channel_ids"] = subject_json['channel_ids']
    params_gpt["prompt_for_bot"] = subject_json['prompt_for_bot']
    params_gpt["summarise_level"] = subject_json['summarise_level']
    params_gpt["max_size_dialog"] = subject_json['max_size_dialog']
    gpt_settings = subject_json['gpt_settings']
    stop_sequences = gpt_settings['stop_sequences']
    params_gpt["model"] = gpt_settings['model']
    params_gpt["temperature"] = gpt_settings['temperature']
    params_gpt["max_tokens"] = gpt_settings['max_tokens']
    params_gpt["top_p"] = gpt_settings['top_p']
    params_gpt["frequency_penalty"] = gpt_settings['frequency_penalty']
    params_gpt["presence_penalty"] = gpt_settings['presence_penalty']

    
    # await client.start(discord_token)
    client.run(discord_token)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 auto_subject.py --subject <name-of-subject>")
        sys.exit(1)

    main()
    