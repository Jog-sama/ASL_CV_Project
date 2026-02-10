# VisibleVoices 🤟

**Learn Sign Language, Amplify Every Voice**

An AI-powered platform for learning and practicing American Sign Language with real-time feedback.

---

## 🤖 AI Assistance Acknowledgment

This project was developed with assistance from **Claude Sonnet 4.5** by Anthropic for:
- Code generation and debugging
- Architecture design and brainstorming
- Frontend component development
- API integration patterns
- Documentation

---

## 📖 About the Product

VisibleVoices is a comprehensive ASL (American Sign Language) learning platform that combines education with AI-powered practice tools. The platform helps users:

- **Learn ASL Fundamentals**: Study the ASL alphabet and basic signs through interactive visual guides
- **Practice with AI**: Use your webcam to practice signs and receive real-time recognition powered by deep learning
- **Get Instant Feedback**: Receive confidence scores and accuracy metrics to track your progress
- **Build Skills**: Develop proficiency in ASL through repeated practice and immediate feedback

### Key Features
- 🎓 Comprehensive ASL alphabet learning grid
- 📹 Real-time webcam-based sign recognition
- 🧠 EfficientNet-B3 deep learning model with 95%+ accuracy
- 📊 Confidence scoring and top-5 predictions
- 🎨 Modern, accessible UI with beige and mauve aesthetics
- 📱 Fully responsive design (mobile, tablet, desktop)

### Tech Stack
- **Frontend**: React 18, Vite, Tailwind CSS, Framer Motion
- **Backend**: FastAPI, Python 3.9+
- **ML Model**: EfficientNet-B3 trained on ASL dataset
- **Deployment**: Vercel (Frontend), HuggingFace Spaces (Backend)

---

## 📁 Project Structure

```
ASL_CV_PROJECT/
├── models/                          # ML model files
│   ├── baseline.py                    
│   ├── classical.py
│   ├── deep_learning.py 
│   ├── explainability.py                  
│-- saved_models/                
│       └── baseline_model.pkl
│       ├── classical_model.pkl                
│
├── scripts/                         # Training and preprocessing scripts
│   ├── data_preprocessing.py
│   ├── train_baseline.py
│   ├── train_efficientnet.py
│   └── evaluate_models.py
│
├── visible-voices/                  # Frontend React application
│   ├── src/
│   │   ├── components/             # React components
│   │   │   ├── NavBar.jsx
│   │   │   ├── BackgroundASL.jsx
│   │   │   └── ASLAlphabetGrid.jsx
│   │   ├── pages/                  # Page components
│   │   │   ├── LearnPage.jsx       # Landing/learning page
│   │   │   ├── PracticePage.jsx    # Camera practice page
│   │   │   └── NotFound.jsx        # 404 page
│   │   ├── App.jsx                 # Main app with routing
│   │   ├── main.jsx                # Entry point
│   │   └── index.css               # Global styles
│   ├── package.json
│   ├── vite.config.js
│   └── tailwind.config.js
│
├── app.py                          # FastAPI backend application
├── config.py                       # Backend configuration
├── main.py                         # Model inference logic
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Docker configuration for HF Space
├── README.md                      
└── start.ipynb                     # Jupyter notebook for experimentation
```

### About `saved_models/`

The `saved_models/` directory contains trained deep learning models. Due to GitHub's file size restrictions, the EfficientNet-B3 model (`.h5` file) is **not stored in this repository**. Instead:

- **Storage Location**: Azure Blob Storage
- **Model Format**: `.h5` (Keras/TensorFlow)
- **Access**: The backend (`app.py`) downloads the model from Azure at runtime
- **Baseline/Classical Models**: Smaller `.pkl` files for baseline and classical models are committed to Git

**HuggingFace Space Configuration** (in `saved_models/README.md`):
```yaml
---
title: ASL Translation API
emoji: 🤟
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---
```

---

## 🚀 Backend Deployment (HuggingFace Spaces)

The backend is deployed as a **FastAPI application** on HuggingFace Spaces using Docker.

### API Endpoints

**Base URL**: `https://YOUR-USERNAME-asl-api.hf.space`

#### `GET /`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "message": "ASL Translation API is running"
}
```

#### `POST /predict`
Upload an image to get ASL sign prediction.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `file` (image file: jpg, jpeg, png)

**Response:**
```json
{
  "predicted_class": 21,
  "predicted_letter": "V",
  "confidence": 0.3447,
  "top5_predictions": [
    {
      "class": 21,
      "letter": "V",
      "confidence": 0.3447
    },
    {
      "class": 26,
      "letter": "del",
      "confidence": 0.3087
    },
    ...
  ]
}
```

### Usage Example

```bash
# Using cURL
curl -X POST "https://YOUR-USERNAME-asl-api.hf.space/predict" \
  -F "file=@image.jpg"

# Using Python requests
import requests

url = "https://YOUR-USERNAME-asl-api.hf.space/predict"
files = {"file": open("image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

### Local Development

1. **Install dependencies:**
```bash
pip install -r requirements.txt --break-system-packages
```

2. **Set environment variables:**
```bash
export AZURE_STORAGE_CONNECTION_STRING="your_connection_string"
export MODEL_CONTAINER_NAME="your_container_name"
export MODEL_BLOB_NAME="efficientnet_b3_asl.h5"
```

3. **Run the server:**
```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Deployment Steps

1. Create a new Space on HuggingFace
2. Choose **Docker** as the SDK
3. Push your code to the Space's Git repository
4. HuggingFace will automatically build and deploy using the `Dockerfile`

---

## 🌐 Frontend Deployment (Vercel)

The frontend is a **React + Vite** application deployed on Vercel.

### Setup Instructions

1. **Install dependencies:**
```bash
cd visible-voices
npm install
```

2. **Update backend URL:**

Edit `src/pages/PracticePage.jsx`:
```javascript
const BACKEND_URL = 'https://YOUR-HUGGINGFACE-SPACE.hf.space/predict'
```

3. **Run development server:**
```bash
npm run dev
```

Visit `http://localhost:5173`

4. **Build for production:**
```bash
npm run build
```

### Deployment to Vercel

**Option 1: Automatic (Recommended)**

1. Connect your GitHub repository to Vercel
2. Import the project
3. Set build settings:
   - **Framework Preset**: Vite
   - **Root Directory**: `visible-voices`
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
4. Deploy!

**Option 2: Manual via CLI**

```bash
# Install Vercel CLI
npm install -g vercel

# Deploy
cd visible-voices
vercel --prod
```

### Environment Variables (Vercel)

No environment variables needed for the frontend. The backend URL is hardcoded in the source.

---

## 🎨 Design System

### Color Palette
- **Background**: `#F5F1ED` (Beige)
- **Accent**: `#E8D5D3` (Mauve)
- **Text**: Stone shades (600-800)

### Typography
- **Headers**: Georgia (serif)
- **Body**: Calibri, Inter (sans-serif)

### Components
- Minimalist, clean aesthetic
- Soft shadows and rounded corners
- Translucent ASL hand decorations
- Smooth animations with Framer Motion

---

## 📊 Model Performance

- **Architecture**: EfficientNet-B3
- **Training Data**: ASL alphabet dataset (A-Z + space, delete)
- **Accuracy**: 95%+ on test set
- **Inference Time**: <100ms per image
- **Input Size**: 224x224 RGB images

---

## 🛠️ Development

### Prerequisites
- **Backend**: Python 3.9+, pip
- **Frontend**: Node.js 18+, npm
- **Model Training**: CUDA-compatible GPU (optional but recommended)

### Training the Model

1. **Prepare dataset:**
```bash
python scripts/data_preprocessing.py
```

2. **Train EfficientNet:**
```bash
python scripts/train_efficientnet.py
```

3. **Evaluate:**
```bash
python scripts/evaluate_models.py
```

### Running Full Stack Locally

**Terminal 1 - Backend:**
```bash
uvicorn app:app --reload --port 7860
```

**Terminal 2 - Frontend:**
```bash
cd visible-voices
npm run dev
```

---

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Dataset**: ASL Alphabet Dataset (Kaggle)
- **Model Architecture**: EfficientNet by Google Research
- **Icons**: React Icons, Font Awesome
- **Fonts**: Google Fonts (Inter, Georgia)
- **AI Assistant**: Claude Sonnet 4.5 by Anthropic

---

**Live Demo**:
- Frontend: [https://visible-voices.vercel.app](https://visible-voices.vercel.app)
- API: [https://your-space.hf.space](https://your-space.hf.space)

---

## 🔮 Future Enhancements

- [ ] Support for full ASL words and phrases
- [ ] Progress tracking and user profiles
- [ ] Multi-language support (BSL, LSF, etc.)
- [ ] Mobile app versions (iOS, Android)
- [ ] Community features and challenges
- [ ] Video-based sentence recognition
- [ ] Integration with educational platforms

---

**Made with ❤️ for the ASL community**

*Empowering communication, one sign at a time.*
