from cmd import PROMPT
import os
import random
import requests
from openai import OpenAI

import asyncio
import discord
from discord.ext import commands
from discord.ui import Button, View
from discord import app_commands
import sys
import json
from urllib.parse import quote
from transformers import AutoTokenizer
import aiocron

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
    "gpt-4"
]

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

max_size_dialog = 60
bot_running_dialog = []
user_refs = []
summaries = {}
command_for_bot="Zhang: "
prompt_suffix = """{0}: {1}
{2}:"""


async def ask_prompt(prompt, model="text-davinci-003", num_results=1, stopSequences=["You:", "Zhang:"], topKReturn=1):
    try:
        local_model = model
        if params_gpt["model_random"]:
            local_model = all_models[random.randint(0, 2)]
        messages = []
        messages.append({"role": "system", "content": ""})
        messages.append({"role": "user", "content": prompt})

        response = client_openai.chat.completions.create(
            model=local_model,
            messages=messages,
            max_tokens=params_gpt["max_tokens"],
            temperature=params_gpt["temperature"],
        )            
        if response != 0:
            for choice in response.choices:
                return choice.text
        return "No response"
    except Exception as e:
        print(e)
        return "Looks like GPT is down!"

async def chat_prompt(author_name, prompt, model='gpt-3.5-turbo', stopSequences=["You:", "Zhang:"]):
    try:
        prompt_for_bot = params_gpt["prompt_for_bot"]
        if type(prompt_for_bot) == list:
            prompt_for_bot = "\n".join(prompt_for_bot).strip()
        prompt_for_bot = eval(f'f"""{prompt_for_bot}"""')

        messages = []
        messages.append({"role": "system", "content": prompt_for_bot})
        history = build_history(author_name, prompt)
        for item in history:
            messages.append(item)
        messages.append({"role": "user", "content": prompt})

        response = client_openai.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=params_gpt["max_tokens"],
            temperature=params_gpt["temperature"],
        )
            # top_p=params_gpt["top_p"],
            # frequency_penalty=params_gpt["frequency_penalty"],
            # presence_penalty=params_gpt["presence_penalty"],
            # stop=stopSequences
        if response != 0:
            return response.choices[0].message
        return {"role": "system", "content": "No response"}
    except Exception as e:
        return {"role": "system", "content": "Looks like ChatGPT is down!"}

async def summarise_channel(history, model="gpt-4", num_results=1, stopSequences=["You:", "Zhang:"], topKReturn=2):
    prompt = "Summarise: \n\n\"" + history + "\"\n\n"
    return await ask_prompt(prompt, model, num_results, stopSequences, topKReturn)


async def fetch(url):
    response = requests.get(url)
    return response

async def retrieve(url):
    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(None, fetch, url)
    return response


@client.event
async def on_ready():
    await tree.sync(guild=discord.Object(guild_id))
    client.loop.create_task(periodic_task())
    print(f'We have logged in as {client.user} to {len(client.guilds)} guilds.')


async def print_history(channel, limit):
    builder = ""
    async for message in channel.history(limit=limit):
        builder += message.author.name + ": " +message.content + "\n"
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


def build_history(author_name, prompt):
    # GPT limit, minus max response token size
    token_limit = 4097 - params_gpt["max_tokens"]

    # Prompt suffix formatting
    ps = prompt_suffix.format(author_name, prompt, bot_name).rstrip()

    # Tokenize the string to get the number of tokens
    tmp = f"{prompt_for_bot}\n\nCurrent conversation: {summary}\n\n\n{ps}"
    tokens_running = tokenizer(tmp, return_tensors='pt')
    tokens_len = tokens_running.input_ids.size(1)
    
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
    bot_running_dialog_limited = build_history(author_name, prompt)

    # Prompt suffix formatting
    ps = prompt_suffix.format(author_name, prompt, bot_name).rstrip()

    # Construct the dialog history
    print(bot_running_dialog_limited)
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

    summary = ""
    if params_gpt['summarise_level'] > 0 and channel.id not in summaries:
        channel_history = await print_history(channel, params_gpt['summarise_level'])
        summary = await summarise_channel(channel_history)
        summaries[channel.id] = summary
        await message.channel.send('Summary for channel {0}: {1}'.format(channel.id, summary))
    elif channel.id in summaries:
        summary = summaries[channel.id]

    member_list = ", ".join(user_refs) 
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
            print(model)
            if model.startswith("gpt-3.5") or model.startswith("gpt-4"):
                try:
                    reply = await chat_prompt(author_name, prompt, stopSequences=stop_sequences + member_stops)
                    result = reply.content.strip()
                except Exception as e:
                    print(e)
                    result = 'So sorry, dear User! ChatGPT is down.'
            else:
                ai_prompt = create_prompt(author_name, prompt)
                # 
                # print(ai_prompt)
                result = await ask_prompt(ai_prompt, stopSequences=stop_sequences + member_stops)
                result = result.strip()
            
            if result != "":
                dialog_instance = prompt_suffix.format(author_name, prompt, bot_name).rstrip() + ' ' + result
                if model.startswith("gpt-3.5") or model.startswith("gpt-4"):
                    if len(bot_running_dialog) >= max_size_dialog:
                        bot_running_dialog.pop(0)
                        bot_running_dialog.pop(0)
                    bot_running_dialog.append({"role": "user", "content": prompt})
                    bot_running_dialog.append({"role": "assistant", "content": result})
                else:
                    if len(bot_running_dialog) >= max_size_dialog:
                        bot_running_dialog.pop(0)
                    bot_running_dialog.append(dialog_instance)


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

                if image_generation == True:
                    await message.reply("Image generation", view=view)

                # Specific handling for input text
                if prompt.startswith("Show me a picture") or prompt.startswith("Show me an image") or prompt.startswith("Show me a photo"):
                    # Split the text string into individual paragraphs
                    paragraphs = result.split('\n\n')

                    # Extract the last paragraph, if there is one
                    if len(paragraphs) > 0:
                        last_paragraph = paragraphs[-1]
                    else:
                        last_paragraph = result
                    
                    # OpenAI
                    image_url = await image_openai(last_paragraph)
                    # download file from URL
                    response = requests.get(image_url)
                    # get filename from URL
                    filename = image_url.split('/')[-1] + '.png'
                    # create temporary file to store downloaded file
                    with open(filename, 'wb') as f:
                        f.write(response.content)
                    file = discord.File(filename)
                    await message.channel.send(file=file)
                    # await message.channel.send(image_url)
                    # SD
                    # image_urls, warning_messages = await image_sd(prompt, steps = 30)
                    # if len(image_urls) > 0:
                    #     for image_url in image_urls:
                    #         file = discord.File(image_url)
                    #         await message.channel.send(file=file)
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



# you can also import this from else where. 
class cronjobs():
    def __init__(self) -> None:

        # print Hello world 20secs. 
        @aiocron.crontab("* * * * * */20")
        async def HelloWorld():
            print("Hello World!")
            print(channel_last)
            if channel_last is None:
                print("No channel_last_id")
                return
            await channel_last.send("Hello World!")

    

@aiocron.crontab("* * * * * */20")
async def hello():
    print("Hello World!")
    print(channel_last)
    if client.is_ready() and channel_last is None:
        print("No channel_last_id")
        return
    await channel_last.send("Hello World!")


async def periodic_task():
    while True:
        print("Hello World!")
        if channel_last:
            
            await channel_last.send("Hello World!")
        else:
            print("No channel_last_id")
        await asyncio.sleep(20)  # sleep for 20 seconds



if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 auto_subject.py --subject <name-of-subject>")
        sys.exit(1)
    
    subject = f'settings_{sys.argv[2]}.json'
    with open(subject, "r") as read_file:
        subject_json = json.loads(read_file.read())
    discord_token_env_var = subject_json['discord_token_env_var']
    discord_token = os.environ.get(discord_token_env_var)
    bot_name = subject_json['name']
    guild_id = subject_json['guild_id']
    channels = subject_json['channels']
    image_generation = str_to_bool(subject_json['image_generation'])

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

    client.run(discord_token)
