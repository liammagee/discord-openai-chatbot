# Discord-OpenAI-chatbot

Code for a minimal Discord chatbot interfacing with OpenAI's GPT-3 service. It has been developed for testing and research, and is not intended for use in production systems. The many caveats about real-world use of large language models apply.

The bot has a limited set of parameters that can be configured via a script (see *settings* examples).

It supports two features that can improve bot performance:

 - Memorisation of prior conversation.
 - Summarisation of channel messages.

Both features can be configured, as described below.


## Instructions

To setup and run a bot:


1. Open a terminal and go to this repository's directory.
2. Create an OpenAI account, and copy the API secret key value.
3. Set the `OPENAI_API_KEY` variable to the secret key value.

    export OPENAI_API_KEY=<insert_secret_key>

4. Create a Discord account, and create or join a Discord server with 'Administrator' permissions.
5. Follow the detailed instructions here to create a bot account:
   <https://discordpy.readthedocs.io/en/stable/discord.html>
   
6. On the Discord server, add the bot account create a Discord channel, e.g. `gpt-3`.
7. On the terminal, set the `DISCORD_BOTOKEN` environment variable:

    export DISCORD_BOTOKEN=<insert_bot_token>

9. [Optional but advisable] Create and activate a virtual environment.
10. Install dependencies:

    pip3 install -r requirements.txt

11. Run the bot server, specifying a name that matches one of the `settings_` files:

    python3 auto_subject.py --subject x > bot-x.out 2>&1

12. Visit the channel (e.g. `gpt-3`), and interact with the bot.


From time to time the bot can get stuck - not responding to a request. Usually this is due to an OpenAI API timeout. Repeating or simplifying the request usually fixes the problem.



## Parameters


The `settings_` files contain options for configuring the bot. 

 - `name`: is included in bot responses.
 - `prompt_for_bot`: is the *seed* prompt for the bot. Included in all OpenAI requests.
 - `channels`: is an array of Discord server channels the bot will respond to.
 - `summarise_level`: on initialisation, the bot will try to summarise (using OpenAI) a number of previous conversations on the channel it is responding to, and include these summaries in its OpenAI requests.
 - `max_size_dialog`: is the number of chat interactions the bot will "memorise" and include in each OpenAI request.
 - `discord_token_env_var`: (by default `DISCORD_TOKEN`). Multiple environment variables can be specified (e.g. `DISCORD_TOKEN_X`, `DISCORD_TOKEN_Y`), to allow multiple concurrent bots. 
 - `gpt_settings`: OpenAI API hyperparameters . For creative conversations, a `temperature` setting of 0.6-0.9 is recommended.
