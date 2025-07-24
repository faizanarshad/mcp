# Diabetes Prediction Discord Bot

This project is a professional Python application that uses AI to predict diabetes class from user input via a Discord bot. It includes model training, deployment, and a user-friendly interface for predictions.

## Features
- Trains a machine learning model on the Multiclass Diabetes Dataset
- Exports the trained model for use in a Discord bot
- Discord bot accepts user input and returns diabetes class predictions
- Easy setup and extensible codebase

## Project Structure
```
├── src/
│   ├── train_model.py
│   └── diabetes_discord_bot.py
├── requirements.txt
├── .gitignore
├── .env.example
├── README.md
└── Multiclass_Diabetes_Dataset.csv
```

## Setup
1. Clone the repository and navigate to the project folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and add your Discord bot token.
4. Train the model:
   ```bash
   python src/train_model.py
   ```
5. Run the Discord bot:
   ```bash
   python src/diabetes_discord_bot.py
   ```

## Usage
- In your Discord server, use the command:
  ```
  !predict <Gender> <AGE> <Urea> <Cr> <HbA1c> <Chol> <TG> <HDL> <LDL> <VLDL> <BMI>
  ```
  Example:
  ```
  !predict 0 50 4.7 46 4.9 4.2 0.9 2.4 1.4 0.5 24.0
  ```

## Environment Variables
- `DISCORD_BOT_TOKEN`: Your Discord bot token (set in `.env`)

## License
MIT