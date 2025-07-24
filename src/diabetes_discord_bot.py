import discord
import joblib
import numpy as np
import os
from dotenv import load_dotenv
import logging
import shap

# Load environment variables
load_dotenv()
TOKEN = os.getenv('DISCORD_BOT_TOKEN')
ADMIN_USER_IDS = os.getenv('ADMIN_USER_IDS', '').split(',')  # Comma-separated list of admin Discord user IDs

# Set up logging
logging.basicConfig(filename='bot.log', level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'diabetes_model.pkl')
model = joblib.load(MODEL_PATH)

# Define the features expected by the model
FEATURES = ['Gender', 'AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']

intents = discord.Intents.default()
client = discord.Client(intents=intents)

WELCOME_MESSAGE = (
    "👋 **Welcome to the Diabetes Prediction Bot!**\n"
    "Type `!help` for instructions or use `!predict` followed by your data."
)

HELP_MESSAGE = (
    "**Diabetes Prediction Bot Help**\n"
    "Use the following command to get a prediction:\n"
    f"`!predict {' '.join(FEATURES)}`\n"
    "Example:\n"
    "`!predict 0 50 4.7 46 4.9 4.2 0.9 2.4 1.4 0.5 24.0`\n\n"
    "To get an explanation of the prediction, use `!explain` with the same input format.\n\n"
    "**Feature order:**\n"
    + '\n'.join([f"- {f}" for f in FEATURES])
)

async def send_welcome_message():
    for guild in client.guilds:
        for channel in guild.text_channels:
            try:
                await channel.send(WELCOME_MESSAGE)
                return
            except Exception:
                continue
        break

@client.event
async def on_ready():
    print(f'Logged in as {client.user}')
    await send_welcome_message()
    logging.info('Bot started and ready.')

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    user_id = str(message.author.id)
    content = message.content.strip()
    logging.info(f'Command from {message.author}: {content}')

    if content.startswith('!help'):
        await message.channel.send(HELP_MESSAGE)
        return

    if content.startswith('!status') and user_id in ADMIN_USER_IDS:
        await message.channel.send('✅ Bot is running.')
        return

    if content.startswith('!shutdown') and user_id in ADMIN_USER_IDS:
        await message.channel.send('Shutting down bot...')
        await client.close()
        return

    if content.startswith('!predict') or content.startswith('!explain'):
        try:
            parts = content.split()
            if len(parts) != len(FEATURES) + 1:
                await message.channel.send(
                    f'❗ Please provide {len(FEATURES)} values in order: {" ".join(FEATURES)}\nType `!help` for details.'
                )
                return
            # Validate input types
            try:
                values = [float(x) for x in parts[1:]]
            except ValueError:
                await message.channel.send('❗ All input values must be numbers. Please check your input.')
                return
            values = np.array(values).reshape(1, -1)
            pred = model.predict(values)[0]
            if content.startswith('!predict'):
                await message.channel.send(f'✅ Predicted diabetes class: **{pred}**')
            elif content.startswith('!explain'):
                # SHAP explainability
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(values)
                    # For multiclass, pick the predicted class
                    if isinstance(shap_values, list):
                        class_idx = int(pred)
                        shap_val = shap_values[class_idx][0]
                    else:
                        shap_val = shap_values[0]
                    # Get top 3 features
                    top_idx = np.argsort(np.abs(shap_val))[::-1][:3]
                    explanation = '\n'.join([
                        f"- {FEATURES[i]}: {shap_val[i]:.3f}" for i in top_idx
                    ])
                    await message.channel.send(
                        f'🔎 **Top features impacting this prediction:**\n{explanation}'
                    )
                except Exception as e:
                    await message.channel.send(f'⚠️ SHAP explanation error: {e}')
                    logging.error(f'SHAP error: {e}')
        except Exception as e:
            await message.channel.send(f'⚠️ Error: {e}')
            logging.error(f'Prediction error: {e}')

if __name__ == '__main__':
    if not TOKEN:
        print('Error: DISCORD_BOT_TOKEN not set in .env')
    else:
        client.run(TOKEN) 