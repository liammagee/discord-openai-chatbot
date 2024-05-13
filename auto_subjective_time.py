from cmd import PROMPT
import os
import io
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
import warnings
from PIL import Image
from stability_sdk import client as client_sd
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
import replicate
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

# Set up our connection to the API.
stability_api = client_sd.StabilityInference(
    key=os.environ['STABILITY_KEY'], # API Key reference.
    verbose=True, # Print debug messages.
    engine="stable-diffusion-xl-1024-v0-9", # Set the engine to use for generation.
    # engine="stable-diffusion-v1-5", # Set the engine to use for generation.
    # Available engines: stable-diffusion-v1 stable-diffusion-v1-5 stable-diffusion-512-v2-0 stable-diffusion-768-v2-0
    # stable-diffusion-512-v2-1 stable-diffusion-768-v2-1 stable-inpainting-v1-0 stable-inpainting-512-v2-0
)

all_models = [
    "gpt-4-1106-preview",
    "gpt-4",
    "gpt-3.5-turbo",
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


class ButtonView(discord.ui.View):
    def __init__(self, data):
        super().__init__()
        self.data = data

    # Put this inside of the View
    @discord.ui.button(label="Generate a Dall-E image", style=discord.ButtonStyle.gray)
    async def button_dalle(self, interaction: discord.Interaction, button: discord.ui.Button):
        data = self.data

        await interaction.response.defer(thinking=True)  # Send a deferred response to the interaction

        try:
            image_url = await image_openai(data)
            loop = asyncio.get_event_loop()
            future1 = loop.run_in_executor(None, requests.get, image_url)
            response = await future1
            filename = image_url.split('/')[-1] + '.png'
            # create temporary file to store downloaded file
            with open(filename, 'wb') as f:
                f.write(response.content)
            file = discord.File(filename)
            await interaction.followup.send("Dall-E:")
            await interaction.followup.send(file=file)
        except Exception as e:
            print(e)

    # # Put this inside of the View
    @discord.ui.button(label="Generate a Stable Diffusion image", style=discord.ButtonStyle.gray)
    async def button_sd(self, interaction: discord.Interaction, button: discord.ui.Button):
        data = self.data

        await interaction.response.defer(thinking=True)  # Send a deferred response to the interaction

        image_urls, warning_messages = await image_sd(data, steps = 50, cgf_scale = 7.0, width = 1024, height = 1024, samples = 4)
        await interaction.followup.send("SDXL 0.9:")
        if len(image_urls) > 0:
            for image_url in image_urls:
                file = discord.File(image_url)
                await interaction.followup.send(file=file)

    # Put this inside of the View
    @discord.ui.button(label="Generate a Replicate (SD) image", style=discord.ButtonStyle.gray)
    async def button_replicate(self, interaction: discord.Interaction, button: discord.ui.Button):
        try:
            data = self.data
            await interaction.response.defer(thinking=True)  # Send a deferred response to the interaction
            loop = asyncio.get_event_loop()

            image_urls = await image_replicate(data, steps = 30)
            await interaction.followup.send("Replicate:")
            for image_url in image_urls:

                try:
                    future1 = loop.run_in_executor(None, requests.get, image_url)
                    response = await future1
                    filename = image_url.split('/')[-1] + '.png'
                    # create temporary file to store downloaded file
                    with open(filename, 'wb') as f:
                        f.write(response.content)
                    file = discord.File(filename)
                    await interaction.followup.send(file=file)
                except requests.exceptions.RequestException as e:
                    print("Error fetching URL:", e)
                    await interaction.followup.send("Request timed out. Maybe try again?")
                except asyncio.exceptions.TimeoutError as e:
                    print("Error fetching URL:", e)
                    await interaction.followup.send("Request timed out. Maybe try again?")
        except Exception as ee:
            print("Error fetching URL:", ee)
            await interaction.followup.send("Request timed out. Maybe try again?")

    # Put this inside of the View
    # @discord.ui.button(label="Generate a Deforum animation", style=discord.ButtonStyle.gray)
    # async def button_deforum(self, interaction: discord.Interaction, button: discord.ui.Button):
    #     data = self.data

    #     await interaction.response.defer(thinking=True)  # Send a deferred response to the interaction

    #     image_urls = await image_deforum(data, steps = 30)
    #     if isinstance(image_urls, str):
    #         image_urls = [image_urls]
    #     for image_url in image_urls:
    #         loop = asyncio.get_event_loop()
    #         try:
    #             future1 = loop.run_in_executor(None, requests.get, image_url)
    #             response = await future1
    #             filename = image_url.split('/')[-1] + '.png'
    #             # create temporary file to store downloaded file
    #             with open(filename, 'wb') as f:
    #                 f.write(response.content)
    #             file = discord.File(filename)
    #             await interaction.followup.send(file=file)
    #         except requests.exceptions.RequestException as e:
    #             print("Error fetching URL:", e)
    #             await interaction.followup.send("Request timed out. Maybe try again?")


@client.event
async def on_ready():
    await tree.sync(guild=discord.Object(guild_id))
    print(f'We have logged in as {client.user} to {len(client.guilds)} guilds.')


async def print_history(channel, limit):
    builder = ""
    async for message in channel.history(limit=limit):
        builder += message.author.name + ": " +message.content + "\n"
    return builder

def prepare_summary(history):
    return history.replace("\n", " ").replace("\"", "'")

async def image_openai(prompt):
    response_openai = openai.Image.create(
        prompt=prompt,
        n=1,
        size="1024x1024"
    )
    image_url = response_openai['data'][0]['url']
    return image_url

@tree.command(name = "dalle", description = "Run Dall-E", guild=discord.Object(guild_id)) #Add the guild ids in which the slash command will appear. If it should be in all, remove the argument, but note that it will take some time (up to an hour) to register the command if it's for all guilds.
async def dalle(interaction: discord.Interaction, prompt: str):
    await interaction.response.defer()
    image_url = await image_openai(prompt)
    await interaction.followup.send(image_url)

async def image_sd(prompt, seed = None, steps = 30, cgf_scale = 0.75, width = 512, height = 512, samples = 1):
    # Set up our initial generation parameters.
    if seed == -1:
        seed = None
    answers = stability_api.generate(
        prompt=[generation.Prompt(text=prompt,parameters=generation.PromptParameters(weight=1))],
        # prompt=prompt,
        seed=seed, # If a seed is provided, the resulting generated image will be deterministic.
                        # What this means is that as long as all generation parameters remain the same, you can always recall the same image simply by generating it again.
                        # Note: This isn't quite the case for Clip Guided generations, which we'll tackle in a future example notebook.
        steps=steps, # Amount of inference steps performed on image generation. Defaults to 30.
        cfg_scale=cgf_scale, # Influences how strongly your generation is guided to match your prompt.
                    # Setting this value higher increases the strength in which it tries to match your prompt.
                    # Defaults to 7.0 if not specified.
        width=width, # Generation width, defaults to 512 if not included.
        height=height, # Generation height, defaults to 512 if not included.
        samples=samples, # Number of images to generate, defaults to 1 if not included.
        # sampler=generation.SAMPLER_K_DPMPP_2M # Choose which sampler we want to denoise our generation with.
        #                                             # Defaults to k_dpmpp_2m if not specified. Clip Guidance only supports ancestral samplers.
        #                                             # (Available Samplers: ddim, plms, k_euler, k_euler_ancestral, k_heun, k_dpm_2, k_dpm_2_ancestral, k_dpmpp_2s_ancestral, k_lms, k_dpmpp_2m)
    )

    img_bytes = None
    image_url = None
    warning_messages = None

    # Set up our warning to print to the console if the adult content classifier is tripped.
    # If adult content classifier is not tripped, save generated images.
    image_urls = []
    warning_messages = []
    try:
        for resp in answers:
            for artifact in resp.artifacts:
                if artifact.finish_reason == generation.FILTER:
                    warning_message = "Your request activated the API's safety filters and could not be processed." + "Please modify the prompt and try again."
                    warnings.warn(warning_message)
                    warning_messages.append(warning_message)
                if artifact.type == generation.ARTIFACT_IMAGE:
                    img_bytes = io.BytesIO(artifact.binary)
                    img = Image.open(img_bytes)
                    image_url = quote(prompt, safe='')[:50] + "_" + str(artifact.seed)+ ".png"
                    img.save(image_url) # Save our generated images with their seed number as the filename.
                    image_urls.append(image_url)
    except Exception as e:
        print(e)
    return image_urls, warning_messages

async def image_replicate(prompt, seed = None, steps = 30, cgf_scale = 0.75, width = 512, height = 512, samples = 1):

    # model = replicate.models.get("cjwbw/stable-diffusion")
    # version = model.versions.get("4049a9d8947a75a0b245e3937f194d27e3277bab4a9c1d6f81922919b65175fd")
    # model = replicate.models.get("daanelson/stable-diffusion-long-prompts")
    # version = model.versions.get("429303627614907bfee34c571c81bf7af9bcc6dec13e3051df2fc09c015e2e2f")
    model = replicate.models.get("stability-ai/stable-diffusion")
    version = model.versions.get("db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf")

    # https://replicate.com/cjwbw/stable-diffusion/versions/4049a9d8947a75a0b245e3937f194d27e3277bab4a9c1d6f81922919b65175fd#input
    inputs = {
        # Input prompt
        'prompt': prompt,

        # The prompt NOT to guide the image generation. Ignored when not using
        # guidance
        'negative_prompt': "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face",

        # pixel dimensions of output image
        'image_dimensions': "768x768",

        # Width of output image. Maximum size is 1024x768 or 768x1024 because
        # of memory limits
        # 'width': 512,

        # Height of output image. Maximum size is 1024x768 or 768x1024 because
        # of memory limits
        # 'height': 512,

        # Inital image to generate variations of. Will be resized to the
        # specified width and height
        # 'init_image': open("path/to/file", "rb"),

        # Black and white image to use as mask for inpainting over init_image.
        # Black pixels are inpainted and white pixels are preserved. Tends to
        # work better with prompt strength of 0.5-0.7. Consider using
        # https://replicate.com/andreasjansson/stable-diffusion-inpainting
        # instead.
        # 'mask': open("path/to/file", "rb"),

        # Prompt strength when using init image. 1.0 corresponds to full
        # destruction of information in init image
        'prompt_strength': 0.8,

        # Number of images to output. If the NSFW filter is triggered, you may
        # get fewer outputs than this.
        # Range: 1 to 10
        'num_outputs': 1,

        # Number of denoising steps
        # Range: 1 to 500
        'num_inference_steps': 30,

        # Scale for classifier-free guidance
        # Range: 1 to 20
        'guidance_scale': 7.5,

        # Choose a scheduler. If you use an init image, PNDM will be used
        'scheduler': "DPMSolverMultistep",

        # Random seed. Leave blank to randomize the seed
        # 'seed': ...,
    }

    # https://replicate.com/cjwbw/stable-diffusion/versions/4049a9d8947a75a0b245e3937f194d27e3277bab4a9c1d6f81922919b65175fd#output-schema
    output = version.predict(**inputs)    
    return output

async def image_deforum(prompt, seed = None, steps = 30, cgf_scale = 0.75, width = 512, height = 512, samples = 1):

    model = replicate.models.get("deforum/deforum_stable_diffusion")
    version = model.versions.get("e22e77495f2fb83c34d5fae2ad8ab63c0a87b6b573b6208e1535b23b89ea66d6")

    # https://replicate.com/deforum/deforum_stable_diffusion/versions/e22e77495f2fb83c34d5fae2ad8ab63c0a87b6b573b6208e1535b23b89ea66d6#input
    inputs = {
        # Number of frames for animation
        # Range: 100 to 1000
        'max_frames': 100,

        # Prompt for animation. Provide 'frame number : prompt at this frame',
        # separate different prompts with '|'. Make sure the frame number does
        # not exceed the max_frames.
        'animation_prompts': f"0: {prompt}",

        # angle parameter for the motion
        'angle': "0:(0)",

        # zoom parameter for the motion
        'zoom': "0: (1.04)",

        # translation_x parameter for the motion
        'translation_x': "0: (0)",

        # translation_y parameter for the motion
        'translation_y': "0: (0)",

        'color_coherence': "Match Frame 0 LAB",

        'sampler': "plms",

        # Choose fps for the video.
        # Range: 10 to 60
        'fps': 15,

        # Random seed. Leave blank to randomize the seed
        # 'seed': ...,
    }
    # https://replicate.com/cjwbw/stable-diffusion/versions/4049a9d8947a75a0b245e3937f194d27e3277bab4a9c1d6f81922919b65175fd#output-schema
    output = version.predict(**inputs)    
    return output

@tree.command(name = "sd", description = "Run Stable Diffusion", guild=discord.Object(guild_id)) #Add the guild ids in which the slash command will appear. If it should be in all, remove the argument, but note that it will take some time (up to an hour) to register the command if it's for all guilds.
async def sd(interaction: discord.Interaction, prompt: str, seed: int = None, steps: int = 30, cgf_scale: float = 7.5, width: int = 512, height: int = 512, samples: int = 1):
    await interaction.response.defer()
    image_urls, warning_messages = await image_sd(prompt, seed, steps, cgf_scale, width, height, samples)
    if len(image_urls) > 0:
        for image_url in image_urls:
            file = discord.File(image_url)
            await interaction.followup.send("SDXL 0.9:")
            await interaction.followup.send(file=file)
    elif len(warning_messages)> -1:
        for warning in warning_messages:
            await interaction.followup.send(warning)
    else:
        await interaction.followup.send("Oh no! Could process this prompt.")


@tree.command(name = "replicate", description = "Run Stable Diffusion", guild=discord.Object(guild_id)) #Add the guild ids in which the slash command will appear. If it should be in all, remove the argument, but note that it will take some time (up to an hour) to register the command if it's for all guilds.
async def replicate_command(interaction: discord.Interaction, prompt: str, seed: int = None, steps: int = 30, cgf_scale: float = 7.5, width: int = 512, height: int = 512, samples: int = 1):
    await interaction.response.defer()
    image_urls = await image_replicate(prompt, seed, steps, cgf_scale, width, height, samples)
    if len(image_urls) > 0:
        for image_url in image_urls:
            loop = asyncio.get_event_loop()
            try:
                future1 = loop.run_in_executor(None, requests.get, image_url)
                response = await future1
                filename = image_url.split('/')[-1] + '.png'
                # create temporary file to store downloaded file
                with open(filename, 'wb') as f:
                    f.write(response.content)
                file = discord.File(filename)
                await interaction.followup.send("Replicate:")
                await interaction.followup.send(file=file)
            except requests.exceptions.RequestException as e:
                print("Error fetching URL:", e)
                await interaction.followup.send(warning)
                await interaction.followup.send("Request timed out. Maybe try again?")
            for image_url in image_urls:
                file = discord.File(image_url)
                await interaction.followup.send(file=file)
    else:
        await interaction.followup.send("Oh no! Could process this prompt.")


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
                view = ButtonView(result)


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
