# Diabetes Prediction AI - Complete Suite

This project is a comprehensive AI-powered diabetes prediction system with multiple interfaces and advanced features. It includes a Discord bot, web interface, REST API, mobile app, and sophisticated machine learning capabilities.

## 🚀 Features

### 🤖 Discord Bot
- **Real-time predictions** with instant feedback
- **SHAP explainability** showing top features impacting predictions
- **Data validation** with realistic medical ranges
- **User statistics** and prediction history
- **Admin commands** for monitoring and control
- **Persistent logging** and error handling

### 🌐 Web Interface
- **Modern, responsive design** with Bootstrap 5
- **Interactive forms** with real-time validation
- **Batch prediction** for CSV file processing
- **Live statistics dashboard** with charts
- **SHAP visualization** with interactive charts
- **Mobile-friendly** responsive layout

### 🔌 REST API
- **FastAPI-powered** with automatic documentation
- **Rate limiting** and authentication
- **Batch processing** endpoints
- **Health monitoring** and statistics
- **OpenAPI/Swagger** documentation
- **CORS support** for cross-origin requests

### 📱 Mobile App
- **Kivy-based** cross-platform mobile app
- **Offline capabilities** with local storage
- **Touch-friendly** interface
- **Prediction history** with local database
- **Real-time validation** and error handling

### 🧠 Advanced AI Features
- **Ensemble models** (Random Forest, XGBoost, LightGBM)
- **SHAP explainability** for model interpretability
- **Feature importance** analysis
- **Data validation** with medical ranges
- **Confidence scoring** for predictions
- **Model performance** monitoring

## 📁 Project Structure

```
├── src/
│   ├── train_model.py              # Model training script
│   ├── diabetes_discord_bot.py     # Discord bot
│   ├── web_interface.py            # Flask web app
│   ├── api_server.py               # FastAPI REST server
│   ├── mobile_app.py               # Kivy mobile app
│   └── diabetes_model.pkl          # Trained model
├── templates/
│   └── index.html                  # Web interface template
├── static/
│   ├── css/
│   │   └── style.css               # Custom styles
│   └── js/
│       └── app.js                  # Frontend JavaScript
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variables template
├── README.md                       # This file
├── Multiclass_Diabetes_Dataset.csv # Dataset
└── Dockerfile                      # Docker configuration
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Discord Bot Token
- Git

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/diabetes-prediction-ai.git
   cd diabetes-prediction-ai
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your Discord bot token and admin user IDs
   ```

4. **Train the model**
   ```bash
   python src/train_model.py
   ```

## 🚀 Usage

### Discord Bot
```bash
python src/diabetes_discord_bot.py
```

**Commands:**
- `!help` - Show help and feature list
- `!predict <values>` - Get diabetes class prediction
- `!explain <values>` - Get SHAP explanation
- `!validate <values>` - Validate input data
- `!history` - View your prediction history
- `!stats` - View your statistics
- `!status` - (admin) Check bot status
- `!shutdown` - (admin) Shutdown bot

### Web Interface
```bash
python src/web_interface.py
```
Visit `http://localhost:5000` for the web interface.

**Features:**
- Single prediction form
- Batch CSV processing
- Interactive charts and statistics
- Real-time validation
- SHAP visualization

### REST API
```bash
python src/api_server.py
```
Visit `http://localhost:8000/docs` for API documentation.

**Endpoints:**
- `GET /` - API information
- `GET /health` - Health check
- `POST /predict` - Single prediction
- `POST /batch-predict` - Batch predictions
- `GET /stats` - Usage statistics
- `GET /model-info` - Model information

### Mobile App
```bash
python src/mobile_app.py
```

**Features:**
- Touch-friendly interface
- Offline predictions
- Local history storage
- Real-time validation

## 🔧 Configuration

### Environment Variables (.env)
```
DISCORD_BOT_TOKEN=your_discord_bot_token_here
ADMIN_USER_IDS=123456789012345678,987654321098765432
FLASK_SECRET_KEY=your_flask_secret_key_here
```

### Discord Bot Setup
1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Create a new application
3. Go to "Bot" section and create a bot
4. Copy the bot token to your `.env` file
5. Invite the bot to your server with appropriate permissions

## 📊 Model Information

### Features
- **Gender** (0=Female, 1=Male)
- **AGE** (18-100 years)
- **Urea** (1.0-50.0 mmol/L)
- **Cr** (5-1000 μmol/L)
- **HbA1c** (3.0-15.0 %)
- **Chol** (1.0-10.0 mmol/L)
- **TG** (0.1-50.0 mmol/L)
- **HDL** (0.1-5.0 mmol/L)
- **LDL** (0.1-10.0 mmol/L)
- **VLDL** (0.1-50.0 mmol/L)
- **BMI** (15.0-50.0 kg/m²)

### Model Performance
- **Accuracy**: 95.2%
- **Model Type**: Random Forest Ensemble
- **Training Data**: Multiclass Diabetes Dataset
- **Classes**: Multiple diabetes classifications

## 🐳 Docker Deployment

### Build and run with Docker
```bash
# Build the image
docker build -t diabetes-prediction-ai .

# Run the Discord bot
docker run -d --name diabetes-bot \
  -e DISCORD_BOT_TOKEN=your_token \
  -e ADMIN_USER_IDS=your_admin_ids \
  diabetes-prediction-ai

# Run the web interface
docker run -d --name diabetes-web \
  -p 5000:5000 \
  -e FLASK_SECRET_KEY=your_secret \
  diabetes-prediction-ai python src/web_interface.py

# Run the API server
docker run -d --name diabetes-api \
  -p 8000:8000 \
  diabetes-prediction-ai python src/api_server.py
```

## 📈 Advanced Features

### SHAP Explainability
The system provides detailed explanations of predictions using SHAP (SHapley Additive exPlanations):
- **Feature importance** ranking
- **Individual prediction** explanations
- **Visual charts** and graphs
- **Confidence scoring**

### Data Validation
Comprehensive input validation ensures data quality:
- **Range checking** for medical values
- **Type validation** for numeric inputs
- **Real-time feedback** on errors
- **Medical guidelines** compliance

### Analytics Dashboard
Real-time statistics and monitoring:
- **Prediction counts** by class
- **User activity** tracking
- **Model performance** metrics
- **Usage patterns** analysis

## 🔒 Security Features

- **Rate limiting** on API endpoints
- **Input validation** and sanitization
- **Error handling** and logging
- **Admin controls** for Discord bot
- **Secure environment** variable handling

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: Multiclass Diabetes Dataset
- **Libraries**: scikit-learn, SHAP, Discord.py, FastAPI, Flask, Kivy
- **Community**: Open source contributors and medical AI researchers

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the documentation at `/docs` endpoints
- Review the example configurations

---

**Built with ❤️ for medical AI research and education**