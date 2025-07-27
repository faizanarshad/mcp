import discord
import joblib
import numpy as np
import pandas as pd
import os
import sqlite3
from dotenv import load_dotenv
import logging
import shap
import datetime
import json

# Load environment variables
load_dotenv()
TOKEN = os.getenv('DISCORD_BOT_TOKEN')
ADMIN_USER_IDS = os.getenv('ADMIN_USER_IDS', '').split(',')

# Set up logging
logging.basicConfig(filename='bot.log', level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# SQLite setup
DB_PATH = os.path.join(os.path.dirname(__file__), 'user_history.db')
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
c.execute('''
    CREATE TABLE IF NOT EXISTS history (
        user_id TEXT,
        timestamp TEXT,
        command TEXT,
        input TEXT,
        prediction TEXT,
        explanation TEXT
    )
''')
conn.commit()

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'diabetes_model.pkl')
model = joblib.load(MODEL_PATH)

# Define the features expected by the model
FEATURES = ['Gender', 'AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']

# Feature validation ranges (min, max)
FEATURE_RANGES = {
    'Gender': (0, 1),
    'AGE': (18, 100),
    'Urea': (1.0, 50.0),
    'Cr': (5, 1000),
    'HbA1c': (3.0, 15.0),
    'Chol': (1.0, 10.0),
    'TG': (0.1, 50.0),
    'HDL': (0.1, 5.0),
    'LDL': (0.1, 10.0),
    'VLDL': (0.1, 50.0),
    'BMI': (15.0, 50.0)
}

intents = discord.Intents.default()
intents.message_content = True
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
    "To get an explanation of the prediction, use `!explain` with the same input format.\n"
    "To see your prediction history, use `!history`.\n"
    "To see your statistics, use `!stats`.\n"
    "To validate your data before prediction, use `!validate`.\n\n"
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

def validate_input(values):
    """Validate input values against expected ranges"""
    errors = []
    for i, (feature, value) in enumerate(zip(FEATURES, values)):
        min_val, max_val = FEATURE_RANGES[feature]
        if value < min_val or value > max_val:
            errors.append(f"{feature}: {value} (should be between {min_val} and {max_val})")
    return errors

def get_user_stats(user_id):
    """Get user statistics"""
    c.execute("SELECT COUNT(*) FROM history WHERE user_id=?", (user_id,))
    total_predictions = c.fetchone()[0]
    
    c.execute("SELECT prediction FROM history WHERE user_id=?", (user_id,))
    predictions = [row[0] for row in c.fetchall()]
    
    if predictions:
        class_counts = {}
        for pred in predictions:
            class_counts[pred] = class_counts.get(pred, 0) + 1
        
        most_common = max(class_counts.items(), key=lambda x: x[1])
        stats = f"Total predictions: {total_predictions}\nMost common class: {most_common[0]} ({most_common[1]} times)"
    else:
        stats = "No predictions yet"
    
    return stats

async def notify_admins(message):
    for admin_id in ADMIN_USER_IDS:
        if admin_id:
            try:
                user = await client.fetch_user(int(admin_id))
                await user.send(message)
            except Exception:
                continue

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

    if content.startswith('!stats'):
        stats = get_user_stats(user_id)
        await message.channel.send(f'📊 **Your Statistics:**\n{stats}')
        return

    if content.startswith('!validate'):
        try:
            parts = content.split()
            if len(parts) != len(FEATURES) + 1:
                await message.channel.send(
                    f'❗ Please provide {len(FEATURES)} values in order: {" ".join(FEATURES)}\nType `!help` for details.'
                )
                return
            try:
                values = [float(x) for x in parts[1:]]
            except ValueError:
                await message.channel.send('❗ All input values must be numbers. Please check your input.')
                return
            
            errors = validate_input(values)
            if errors:
                error_msg = '❌ **Validation Errors:**\n' + '\n'.join(errors)
                await message.channel.send(error_msg)
            else:
                await message.channel.send('✅ **All values are within expected ranges!**')
        except Exception as e:
            await message.channel.send(f'⚠️ Error: {e}')
            logging.error(f'Validation error: {e}')
        return

    if content.startswith('!history'):
        c.execute("SELECT timestamp, command, input, prediction, explanation FROM history WHERE user_id=? ORDER BY timestamp DESC LIMIT 5", (user_id,))
        rows = c.fetchall()
        if not rows:
            await message.channel.send("No history found for you.")
        else:
            history_msg = "**Your last 5 predictions:**\n"
            for row in rows:
                history_msg += f"\n- `{row[0][:19]}` `{row[1]}`\n  Input: {row[2]}\n  Prediction: {row[3]}\n"
                if row[4]:
                    history_msg += f"  Explanation: {row[4]}\n"
            await message.channel.send(history_msg)
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
            
            # Validate input ranges
            errors = validate_input(values)
            if errors:
                error_msg = '❌ **Validation Errors:**\n' + '\n'.join(errors[:3])  # Show first 3 errors
                if len(errors) > 3:
                    error_msg += f'\n... and {len(errors) - 3} more errors'
                await message.channel.send(error_msg + '\nUse `!validate` to check your data before predicting.')
                return
            
            # Create DataFrame with feature names to avoid warning
            values_df = pd.DataFrame([values], columns=FEATURES)
            pred = model.predict(values_df)[0]
            explanation = ""
            
            if content.startswith('!predict'):
                await message.channel.send(f'✅ Predicted diabetes class: **{pred}**')
            elif content.startswith('!explain'):
                # SHAP explainability
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(values_df)
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
            # Log history
            log_history(user_id, parts[0], " ".join(parts[1:]), str(pred), explanation)
        except Exception as e:
            await message.channel.send(f'⚠️ Error: {e}')
            logging.error(f'Prediction error: {e}')
            await notify_admins(f'Critical error for user {user_id}: {e}')

def log_history(user_id, command, input_str, prediction, explanation):
    c.execute(
        "INSERT INTO history (user_id, timestamp, command, input, prediction, explanation) VALUES (?, ?, ?, ?, ?, ?)",
        (user_id, datetime.datetime.now().isoformat(), command, input_str, prediction, explanation)
    )
    conn.commit()

if __name__ == '__main__':
    if not TOKEN:
        print('Error: DISCORD_BOT_TOKEN not set in .env')
    else:
        client.run(TOKEN) 