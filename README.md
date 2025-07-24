# Diabetes Prediction Discord Bot

This project is a professional Python application that uses AI to predict diabetes class from user input via a Discord bot. It includes model training, deployment, user history, explainability, admin commands, and a user-friendly interface for predictions.

## Features
- Trains a machine learning model on the Multiclass Diabetes Dataset
- Exports the trained model for use in a Discord bot
- Discord bot accepts user input and returns diabetes class predictions
- SHAP explainability: `!explain` command shows top features impacting the prediction
- Persistent user history: `!history` command shows your last 5 predictions
- Admin commands: `!status`, `!shutdown`
- Logging of all commands and errors
- Docker deployment support
- Easy setup and extensible codebase

## Project Structure
```
├── src/
│   ├── train_model.py
│   ├── diabetes_discord_bot.py
│   └── diabetes_model.pkl
├── requirements.txt
├── .gitignore
├── .env.example
├── README.md
├── Multiclass_Diabetes_Dataset.csv
└── Dockerfile
```

## Setup
1. Clone the repository and navigate to the project folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and add your Discord bot token and admin user ID(s).
4. Train the model:
   ```bash
   python src/train_model.py
   ```
5. Run the Discord bot:
   ```bash
   python src/diabetes_discord_bot.py
   ```

## Usage
- In your Discord server, use the following commands:
  - `!help` — Show instructions and feature order
  - `!predict <Gender> <AGE> <Urea> <Cr> <HbA1c> <Chol> <TG> <HDL> <LDL> <VLDL> <BMI>` — Get a diabetes class prediction
  - `!explain <Gender> <AGE> <Urea> <Cr> <HbA1c> <Chol> <TG> <HDL> <LDL> <VLDL> <BMI>` — Get top features impacting the prediction
  - `!history` — See your last 5 predictions and explanations
  - `!status` — (admin only) Check if the bot is running
  - `!shutdown` — (admin only) Shut down the bot

Example:
```
!predict 0 50 4.7 46 4.9 4.2 0.9 2.4 1.4 0.5 24.0
!explain 0 50 4.7 46 4.9 4.2 0.9 2.4 1.4 0.5 24.0
```

## Environment Variables
- `DISCORD_BOT_TOKEN`: Your Discord bot token (set in `.env`)
- `ADMIN_USER_IDS`: Comma-separated list of Discord user IDs with admin privileges

## Docker Deployment
1. Build the Docker image:
   ```bash
   docker build -t diabetes-discord-bot .
   ```
2. Run the container:
   ```bash
   docker run --env-file .env diabetes-discord-bot
   ```

## License
MIT