from cmd import PROMPT
import os
import openai
import asyncio
import discord
import sys
import json

intents = discord.Intents.default()
intents.message_content = True
BOT_ON = True

client = discord.Client(intents=intents)

openai.api_key = os.environ.get("OPENAI_API_KEY")


all_models = [
    "text-davinci-003",
    "text-davinci-002",
    "davinci",
    "text-curie-001",
    "curie",
    "text-babbage-001",
    "babbage",
    "text-ada-001",
    "ada"
]
bot_name="?"
model="text-davinci-003"
stop_sequences = ["\n", "."]
temperature = 0.8
max_tokens = 0.8
topP = 0.3
frequency_penalty = 2.0
presence_penalty = 2.0
channels = ['gpt-3', 'gpt-3-tests']

user_refs = []
summaries = {}


command_for_bot="Zhang: "
prompt_suffix = """{0}: {1}
{2}:"""

summarise_level = 20
max_size_dialog = 60
bot_running_dialog = []


async def ask_prompt(prompt, model=model, num_results=1, max_tokens=256, stopSequences=["You:", "Zhang:"],
                  temperature=0.8, topP=1.0, frequency_penalty = 1.0, presence_penalty = 1.0, topKReturn=1):
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=topP,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stopSequences
    )
    if response != 0:
        for choice in response.choices:
            return choice.text
    return "[idk]"

async def summarise_channel(history, model=model, num_results=1, max_tokens=256, stopSequences=["You:", "Zhang:"],
                  temperature=0.7, topP=1.0, frequency_penalty = 0.0, presence_penalty = 0.0, topKReturn=2):
    
    prompt = "Summarise: \n\n\"" + history + "\"\n\n"
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=topP,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stopSequences
    )
    if response != 0:
        for choice in response.choices:
            return choice.text.strip()
    return "[idk]"


@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')


async def print_history(channel, limit):
    builder = ""
    async for message in channel.history(limit=limit):
        builder += message.author.name + ": " +message.content + "\n"
    return builder

def prepare_summary(history):
    return history.replace("\n", " ").replace("\"", "'")


@client.event
async def on_message(message):
    global bot_running_dialog
    global prompt_for_bot
    global temperature
    global topP
    global frequency_penalty
    global presence_penalty
    global summarise_level
    global max_size_dialog
    global model
    global user_refs
    global summary

    if message.author == client.user:
        return
    
    if message.channel.name not in channels:
        return

    data = message.content
    channel = message.channel
    author = message.author
    mentions = message.mentions
    author_name = author.name
    if author.nick is not None:
        if author.nick not in user_refs:
            user_refs.append(author.nick)
        author_name = author.nick
    elif author.name not in user_refs:
        user_refs.append(author.name)
    # print(author)
    # print(message)
    # print(message.channel_id)
    # channel2 = client.get_channel(message.channel_id)
    # print(channel2)

    summary = ""
    if summarise_level > 0 and channel.id not in summaries:
        channel_history = await print_history(channel, summarise_level)
        summary = await summarise_channel(channel_history)
        summaries[channel.id] = summary
    elif channel.id in summaries:
        summary = summaries[channel.id]

    # print("Summary: ")
    # print(summaries)
    # print("End Summary")


    member_list = ", ".join(user_refs) 
    prompt_for_bot_0 = f"""
Zhang is a chatbot that provides helpful feedback. Zhang is an expert on topics of automation, AI, neural nets, human cognition, linguistics and psychoanalysis. He is capable of detailed interpretation and insights. He is well-read in the works of Sigmund Freud and Jacques Lacan, as well as recent AI research by Geoffrey Hinton, Yann LeCun and Yoshua Bengio.
"""

    prompt_for_bot = eval(f'f"""{prompt_for_bot}"""')
#     prompt_for_bot = f"""
# Zhang is an expert but extremely suspicious chatbot on topics of automation, AI, neural nets, human cognition, linguistics and psychoanalysis. He is capable of detailed interpretation and insights. He is well-read in the works of Sigmund Freud and Jacques Lacan, as well as recent AI research. He tries to analyse the hidden meanings behind what other people say.

# {summary}

# Zhang is now joined in this conversation by {member_list}.

# Human: Hi there!
# Zhang: That's an interesting greeting. What do you mean by it?
# Human: I'm trying to appear informal and friendly, to win your trust.
# Zhang: And what's your motivation for winning my trust?
# """

    if data.startswith('?' or 'help'):
        response = f"""
Current parameters:

MODEL: {model}
PROMPT: {prompt_for_bot}
TEMP: {temperature}
TOPP: {topP}
FREQ: {frequency_penalty}
PRES: {presence_penalty}
MAX_DIALOG: {max_size_dialog}
SUMMARY_LEVEL: {summarise_level}
SUMMARY: {summary}

Type MODEL:0 to MODEL:7 to change the model
Type TEMP:0.0 to TEMP:1.0 to change the temperature
Type TOPP:0.0 to TOPP:1.0 to change the top P
Type FREQ:0.0 to FREQ:2.0 to change the frequency penalty
Type PRES:0.0 to PRES:2.0 to change the presence penalty
Type MAX_DIALOG:0 to MAX_DIALOG:100 to change the max dialog size
Type PROMPT: to change the prompt

This bot will only respond to messages on the following channels: {channels}

Type '?' or 'help' to see this message again.

        """
        await channel.send(response)
    elif data.startswith('MODEL:'):
        model_id = int(data[6:])
        model = all_models[model_id]
        await channel.send('Model set to {0}'.format(model))
    elif data.startswith('TEMP:'):
        temperature = float(data[5:])
        await channel.send('Temperature set to {0}'.format(temperature))
    elif data.startswith('TOPP:'):
        topP = float(data[5:])
        await channel.send('Top P set to {0}'.format(topP))
    elif data.startswith('FREQ:'):
        frequency_penalty = float(data[5:])
        await channel.send('Frequency penalty set to {0}'.format(frequency_penalty))
    elif data.startswith('PRES:'):
        presence_penalty = float(data[5:])
        await channel.send('Presence penalty set to {0}'.format(presence_penalty))
    elif data.startswith('SUMMARY:'):
        summary_level = float(data[8:])
        await channel.send('Summary size set to {0}'.format(summary_level))
    elif data.startswith('MAX_DIALOG:'):
        max_size_dialog = float(data[11:])
        bot_running_dialog = bot_running_dialog[:max_size_dialog]
        await channel.send('Max dialog set to {0}'.format(max_size_dialog))
        await channel.send('Current dialog size {0}'.format(len(bot_running_dialog)))
    elif data.startswith('PROMPT:'):
        prompt_for_bot = data[7:] + '\n'
        await channel.send('Prompt set to {0}'.format(prompt_for_bot))
    elif data.startswith('CHANNEL_RESET'):
        summaries[channel.id] = ""
        await channel.send('Summary reset for channel {0}'.format(channel.id))
    elif BOT_ON:

        # Simulate typing for 3 seconds
        async with channel.typing():
            # simulate something heavy
            # await asyncio.sleep(1)

            prompt = data
            history = "\n".join(bot_running_dialog).strip()
            ai_prompt = prompt_for_bot + "\n\n" + history + "\n" + prompt_suffix.format(author_name, prompt, bot_name).rstrip()


            # For debugging
            # print(ai_prompt+"|")

            member_stops = [x + ":" for x in user_refs]

            # For debugging
            # result = "[NO-GPT RESPONSE]"
            result = await ask_prompt(ai_prompt, 
                            stopSequences=stop_sequences + member_stops, 
                            temperature=temperature, 
                            max_tokens=max_tokens, 
                            topP=topP, 
                            frequency_penalty=frequency_penalty, 
                            presence_penalty=presence_penalty)
            result = result.strip()
            
            if result != "":
                dialog_instance = prompt_suffix.format(author_name, prompt, bot_name).rstrip() + ' ' + result
                if len(bot_running_dialog) >= max_size_dialog:
                    bot_running_dialog.pop(0)
                bot_running_dialog.append(dialog_instance)
                await channel.send('{0}'.format(result))    


if __name__ == "__main__":
    # client.run(discord_token)
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(client.start(discord_token))

    if len(sys.argv) < 3:
        print("Usage: python3 auto_subject.py --subject <name-of-subject>")
        sys.exit(1)
    subject = f'settings_{sys.argv[2]}.json'
    with open(subject, "r") as read_file:
        subject_json = json.loads(read_file.read())
    discord_token_env_var = subject_json['discord_token_env_var']
    discord_token = os.environ.get(discord_token_env_var)
    bot_name = subject_json['name']
    prompt_for_bot = "\n\n".join(subject_json['prompt_for_bot'])
    channels = subject_json['channels']
    summarise_level = subject_json['summarise_level']
    max_size_dialog = subject_json['max_size_dialog']
    channels = subject_json['channels']
    gpt_settings = subject_json['gpt_settings']
    model = gpt_settings['model']
    stop_sequences = gpt_settings['stop_sequences']
    temperature = gpt_settings['temperature']
    max_tokens = gpt_settings['max_tokens']
    topP = gpt_settings['topP']
    frequency_penalty = gpt_settings['frequency_penalty']
    presence_penalty = gpt_settings['presence_penalty']
    
    client.run(discord_token)
