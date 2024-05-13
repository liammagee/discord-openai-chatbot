import os
import sys
import json
from datetime import datetime

from discord.ext import tasks, commands
from discord.ui import Button, View
from discord import app_commands,  Member, Permissions, Status, Emoji
from discord.ui import Button, View
import discord

import openai
from transformers import AutoTokenizer

import sqlite3
from sqlite3 import Error
import pandas as pd

import asyncio

conn = sqlite3.connect('messages.db')  # create a new database file named messages.db
c = conn.cursor()

# Create table
c.execute('''CREATE TABLE IF NOT EXISTS entries
             (id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT, author TEXT, timestamp TEXT)''')

conn.commit()  # save (commit) the changes
conn.close()  # close the connection


all_models = [
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


# Load a pre-trained tokenizer (for example, the GPT-2 tokenizer)
tokenizer = AutoTokenizer.from_pretrained('gpt2')


INITIAL_MESSAGE = "Welcome to Alter Ego! I'm here to help you write down your thoughts and maintain a journal! No pressure - you can leave at any time. Shall we get started? Click the 'Add Reaction' button and give a thumbs up - or just type 'yes' - to begin."
CHANNEL_ID = 1092954017321201784
EMOJIS = [":thumbs_up:", ":thumbs_down:", ":shrug:", ":thinking:", ""]



class ButtonView(discord.ui.View):
    def __init__(self, bot, content, author, created_at):
        super().__init__()
        self.bot = bot
        self.content = content
        self.author = author
        self.created_at = created_at


    # Put this inside of the View
    @discord.ui.button(label="Save", style=discord.ButtonStyle.primary)
    async def button_save(self, interaction: discord.Interaction, button: discord.ui.Button):

        await self.bot.save_entry(self.content, self.author, self.created_at)
        await interaction.response.send_message("Saved!", ephemeral=True)






class JournallingBot(commands.Bot):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params_gpt = params_gpt
        self.channel_ids = []
        for id in channel_ids:
            self.channel_ids.append(int(id))
        self.user_refs = []
        self.bot_running_dialog = []
        self.last_message = ""
        self.prompt_suffix = """{0}: {1}
{2}:"""
        self.synced = False
        self.storing = False
        self.confirming = False
        
        @self.tree.command(name = "testing123")
        async def testing123(interaction: discord.Interaction):
            await interaction.response.send_message(f"Hello {interaction.user.name}!")
        
        @self.tree.command(name = "list")
        async def entry_list(interaction: discord.Interaction):
            await self.get_journal_entries_text(interaction)
        
        @self.tree.command(name = "entry")
        async def entry_detail(interaction: discord.Interaction, entry_id: int):
            await self.get_journal_entry_detail(interaction, entry_id)
        
        @self.tree.command(name = "patterns")
        async def patterns(interaction: discord.Interaction):
            await interaction.response.defer(thinking=True)  # Send a deferred response to the interaction
            result = await self.analyse_journal_entries(interaction)
            await interaction.followup.send(result[0:1990])
        
        @self.tree.command(name = "export")
        async def export(interaction: discord.Interaction):
            # await interaction.response.defer(thinking=True)  # Send a deferred response to the interaction
            # loop = asyncio.get_event_loop()
            filename = await self.export_journal_entries(interaction)
            file = discord.File(filename)
            await interaction.response.send_message(file=file)

        @self.tree.command(name = "save")
        async def save(interaction: discord.Interaction):
            await self.save_entry(self.last_message, interaction.user.name, interaction.created_at)
            await interaction.response.send_message("Saved!")

        @self.tree.command(name = "help")
        async def help(interaction: discord.Interaction):
            help = await self.llm_help(interaction.user.name)
            await interaction.response.send_message(help)

    async def save_entry(self, content, author_name, created_at):

        conn = sqlite3.connect('messages.db')
        c = conn.cursor()

        journal_entry = self.extract_journal_entry(content)


        # Insert a row of data
        c.execute("INSERT INTO entries (content, author, timestamp)  VALUES (?, ?, ?)",
                (journal_entry, str(author_name), str(created_at)))

        # Save (commit) the changes
        conn.commit()

        # Close the connection
        conn.close()
        self.confirming = False


    async def llm_help(self, user_name):

        SYS_PROMPT = "I provide detailed assistance based on a minimal 'help' prompt. "
        help_instructions = f"""
Welcome to Alter Ego, **{user_name}**! 

There are a few commands to help work with Alter Ego (type '/' to see these listed):
 - **/list** - list your journal entries
 - **/entry** - view a specific journal entry
 - **/patterns** - view the most common patterns in your journal entries
 - **/export** - export your journal entries to a CSV file
 - **/save** - save your last message as a journal entry
 - **/help** - view this help message

You can also type **'help'** at any time to get help with what to say next.

At certain times you may need to react to a message to save or continue the conversation. You can do this by clicking the 'Add Reaction' button and selecting the appropriate 'Alter Ego' emoji.

I'm still learning, so please be patient with me! If you have any feedback, please let me know in the #feedback channel.

"""

        return help_instructions

        # messages = []
        # messages.append({"role": "system", "content": SYS_PROMPT})
        # messages.append({"role": "user", "content": help_instructions})

        # try:
        #     response = openai.ChatCompletion.create(
        #         model='gpt-3.5-turbo',
        #         messages=messages,
        #         max_tokens=250,
        #         temperature=0.0,
        #     )
        #     if response != 0:
        #         return response.choices[0].message.content
        #     return "No response"
        # except Exception as e:
        #     print(e)
        #     return "Can't help just now, sorry."


    def get_journal_entries(self, interaction):
        # Connect to the database
        conn = sqlite3.connect('messages.db')

        df = pd.read_sql_query(f"SELECT id, SUBSTR(content, 1, 80) as content, author, timestamp FROM entries WHERE author = '{interaction.user.name}' ORDER BY timestamp DESC LIMIT 100", conn)

        # Close the connection
        conn.close()

        return df



    async def get_journal_entries_text(self, interaction):

        await interaction.response.defer(thinking=True)
        df = self.get_journal_entries(interaction)
    
        markdown = ''
        for index, row in df.iterrows():
            # Convert the timestamp (seconds since epoch) to a datetime object
            # dt_object = datetime.fromtimestamp(row['timestamp'])
            dt_object = datetime.strptime(row['timestamp'], "%Y-%m-%d %H:%M:%S.%f%z")

            # Format the datetime object as a string
            date_string = dt_object.strftime("%Y-%m-%d")
            markdown_local = ''
            markdown_local += f"**Entry ID:** {row['id']}\n" 
            markdown_local += f"**Entry created:** {date_string}\n" 
            markdown_local += f"**Authored by:** {row['author']}\n"
            markdown_local += f"**Journal Title**:\n {row['content']}\n\n"
            markdown_local += "---\n\n"
            if len(markdown + markdown_local) > 1990:
                await interaction.followup.send(markdown[0:1990])
                markdown = ''
            markdown += markdown_local

        await interaction.followup.send(markdown[0:1990])





    async def get_journal_entry_detail(self, interaction, entry_id):

        await interaction.response.defer(thinking=True)
        # Connect to the database
        conn = sqlite3.connect('messages.db')
        conn.row_factory = sqlite3.Row

        # Create a cursor object
        cur = conn.cursor()

        # Execute the SELECT statement
        cur.execute(f"SELECT content, author, timestamp FROM entries WHERE author = '{interaction.user.name}' AND id = {entry_id}")

        # Fetch one row
        row = cur.fetchone()
 
        markdown = ''
        # Convert the timestamp (seconds since epoch) to a datetime object
        # dt_object = datetime.fromtimestamp(row['timestamp'])
        dt_object = datetime.strptime(row['timestamp'], "%Y-%m-%d %H:%M:%S.%f%z")

        # Format the datetime object as a string
        date_string = dt_object.strftime("%Y-%m-%d")
        markdown += f"**Entry created:** {date_string}\n" 
        markdown += f"**Authored by:** {row['author']}\n"
        markdown += f"**Journal Title**:\n {row['content']}\n\n"
        markdown += "---\n\n"

        await interaction.followup.send(markdown)


    async def analyse_journal_entries(self, interaction):

        ANALYSIS_PROMPT = "Based on my knowledge of psychology, psychotherapy, hermeneutics, linguistics and textual analysis, I look deeply for insights in the entry texts. I try not to make simplistic, naive or 'pop psychology'-style observations, but look for subtle clues and changes in the discourse of the entries over time. I analyse a series of journal entries and tell you what I think about it. I can also tell you about patterns in your journal entries. "

        messages = []
        messages.append({"role": "system", "content": ANALYSIS_PROMPT})

        prompt = 'Below are the journal entries to analyse. Each entry includes the journal entry date, author and entry contents. Each entry is separated by three dashes ("---"). Entries are organised from newest to oldest. \n\n'

        journal_entries_text = self.get_journal_entries_text(interaction)
        prompt += journal_entries_text

        prompt += '\n\nNow I will analyse the journal entries. \n\n'

        messages.append({"role": "user", "content": prompt})

        try:
            response = openai.ChatCompletion.create(
                model='gpt-4',
                messages=messages,
                max_tokens=1000,
                temperature=0.8,
            )
            if response != 0:
                return response.choices[0].message.content
            return "No response"
        except Exception as e:
            print(e)
            return "Looks like ChatGPT is down!"


    async def export_journal_entries(self, interaction):
        df = self.get_journal_entries()

        # Get the current date and time
        now = datetime.now()

        # Format the date and time as a string
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        file =  f"journal_entries_{interaction.user}_timestamp.csv"
        df.to_csv(file)

        return file

    async def on_ready(self): # override the on_ready event
        await self.wait_until_ready()
        print(f'Logged in as {self.user.name}')
        print('------')
        
        if not self.synced:
            for server in self.guilds: # Should only be one
                self.tree.copy_global_to(guild=discord.Object(id=server.id))
                await self.tree.sync(guild=discord.Object(id=server.id))
            self.synced = True

        # for slash_command in self.tree.walk_commands():
        #     print(f"Slash command: {slash_command.name}")
        #     print(slash_command)

        await self.welcome()
        


    def build_history(self, author_name, prompt, prompt_for_bot, prompt_suffix, summary = ''):
        # GPT limit, minus max response token size
        token_limit = 8000 - params_gpt["max_tokens"]

        # Prompt suffix formatting
        ps = prompt_suffix.format(author_name, prompt, bot_name).rstrip()

        # Tokenize the string to get the number of tokens
        tmp = f"{prompt_for_bot}\n\nCurrent conversation: {summary}\n\n\n{ps}"
        tokens_running = tokenizer(tmp, return_tensors='pt')
        tokens_len = tokens_running.input_ids.size(1)
        
        # Iterate through the list in reverse order
        bot_running_dialog_limited = []
        for item in reversed(self.bot_running_dialog):
            tokens_item = tokenizer(str(item) + "\n", return_tensors='pt')
            tokens_item_len = tokens_item.input_ids.size(1)
            if tokens_len + tokens_item_len < token_limit:
                tokens_len = tokens_len + tokens_item_len
                bot_running_dialog_limited.insert(0, item)
            else:
                break
        # Construct the dialog history
        return bot_running_dialog_limited


    async def chat_prompt(self, author_name, prompt, model='gpt-4'):
        try:
            prompt_for_bot = self.params_gpt["prompt_for_bot"]
            if type(prompt_for_bot) == list:
                prompt_for_bot = "\n".join(prompt_for_bot).strip()
            prompt_for_bot = eval(f'f"""{prompt_for_bot}"""')

            messages = []
            messages.append({"role": "system", "content": prompt_for_bot})
            history = self.build_history(author_name, prompt, prompt_for_bot, self.prompt_suffix)
            for item in history:
                messages.append(item)
            prompt = f"{author_name} says {prompt}"
            messages.append({"role": "user", "content": prompt})
            print("*** Messages ***")
            for m in messages:
                print(m)
            print("*** End Messages ***")
            response = openai.ChatCompletion.create(
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


    async def confirm_message(self, bot, message):
        self.confirming = True
        # check = lambda m: m.author == message.author and m.channel == message.channel
        check = lambda m: m != None
        await message.channel.send('Do you want to store this journal entry? (yes/no)')

        try:
            msg = await self.wait_for('message', timeout=20.0, check=check)
            if msg.content.lower() == 'yes':
                return True
            else:
                return False
        except asyncio.TimeoutError:
            await message.channel.send('Confirmation timed out.')
            return False
        

    async def do_llm(self, channel, author_name, data):
        async with channel.typing():
            model = all_models[self.params_gpt['model_id']]
            member_stops = [x + ":" for x in self.user_refs]
            if model.startswith("gpt-3.5") or model.startswith("gpt-4"):
                try:
                    reply = await self.chat_prompt(author_name, data)
                    result = reply['content'].strip()

                    self.bot_running_dialog.append({"role": "user", "content": data})
                    self.bot_running_dialog.append({"role": "assistant", "content": result})

                except Exception as e:
                    print(e)
                    result = 'So sorry, dear User! ChatGPT is down.'

            self.last_message = result
            await channel.send(result)
            return result


    async def on_message(self, message):
        data = message.content
        channel = message.channel
        author = message.author
        mentions = message.mentions
        author_name = author.name

        if author == self.user:
            return

        if message.channel.id not in self.channel_ids:
            return        

        if self.confirming == True:
            return

        # Special commands
        if data == "help" or data == "Help" or data == "HELP" or data == "h" or data == "H" or data == "?":
            help_instructions = await self.llm_help(author.name)
            await channel.send(help_instructions)
        else:
            result = await self.do_llm(channel, author_name, data)
            if 'draft journal entry' in result.lower():
                view = ButtonView(self, result, author_name, message.created_at)
                await message.reply("Save the journal entry", view=view)

            

        # Start the task when the bot starts
        # self.initiated_prompt = False
        # self.spontaneous_prompt.start()

        await self.process_commands(message)



    @tasks.loop(minutes=5)  # run every 5 minutes
    async def spontaneous_prompt(self):
        channel = self.get_channel(self.channel_ids[0])  # replace with channel ID that task should post in
        message = "Hi! Feel like talking?"
        if self.last_message != message and self.initiated_prompt == True:
            await channel.send(message)
        self.last_message = message
        self.initiated_prompt = True

    async def welcome(self):
        channel = self.get_channel(self.channel_ids[0])  # replace with channel ID that task should post in

        await channel.send(INITIAL_MESSAGE)

        online_members = []
        for member in channel.guild.members:
            if isinstance(member, Member) and not member.bot:  # exclude bots
                permissions = channel.permissions_for(member)
                if permissions.read_messages and member.status == Status.online:
                    online_members.append(member.name)
        online_members_str = ", ".join(online_members)

        if len(online_members) > 0:
            await channel.send(f"It looks like we have the following members online: {online_members_str}. Welcome everyone!")

        # await channel.send("\U0001F44D\U0001F3FD")
        await self.send_emoji()

        # Start the task when the bot starts
        self.initiated_prompt = False
        if not self.spontaneous_prompt.is_running():
            self.spontaneous_prompt.start()
    
    
    @spontaneous_prompt.before_loop
    async def before_spontaneous_prompt(self):
        await self.wait_until_ready()  # wait until the bot logs in

    
    async def send_emoji(self):
        channel = self.get_channel(self.channel_ids[0])  # replace with channel ID that task should post in
        emoji = "<:alterego:1122337236705349632>"
        await channel.send(emoji)


    def extract_journal_entry(self, text):
        # Split the text by the known delimiter
        parts = text.split('---')

        # If there's no delimiter, return the entire text
        if len(parts) == 1:
            return parts[0].strip()

        # Otherwise, return the second part, after the first delimiter
        else:
            return parts[1].strip()

    async def on_reaction_add(self, reaction, user):
        data = reaction.emoji
        message = reaction.message
        channel = reaction.message.channel
        author_name = user.name

        emoji_list = [f"{emoji.name}: {emoji.id}" for emoji in channel.guild.emojis]
        # print(emoji_list)
        # print(data)
    
           # self.spontaneous_prompt.cancel()
        if isinstance(data, Emoji) and data.name == "alterego":
            confirm = await self.confirm_message(bot, message)
            if confirm:
                await self.save_entry(message.content, author_name, message.created_at)

        else:
            await self.do_llm(channel, author_name, data)

            # Start the task when the bot starts
            # self.initiated_prompt = False
            # self.spontaneous_prompt.start()

def str_to_bool(s: str) -> bool:
    if s is None:
        return False
    elif s.lower() == 'true':
        return True
    elif s.lower() == 'false':
        return False
    else:
        raise ValueError(f"Cannot convert '{s}' to a boolean value")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 auto_subject_simplified.py --subject <name-of-subject>")
        sys.exit(1)
    subject = f'settings_{sys.argv[2]}.json'
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

    with open(subject, "r") as read_file:
        subject_json = json.loads(read_file.read())
    discord_token_env_var = subject_json['discord_token_env_var']
    discord_token = os.environ.get(discord_token_env_var)
    bot_name = subject_json['name']
    guild_id = subject_json['guild_id']
    channel_ids = subject_json['channel_ids']

    if len(channel_ids) <= 0:
        print("No channels specified in settings file")
        sys.exit(1)

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
    
    intents = discord.Intents.all()
    bot = JournallingBot(command_prefix='$', intents=intents, params_gpt=params_gpt, channel_ids=channel_ids)
    bot.run(discord_token)
    

    
