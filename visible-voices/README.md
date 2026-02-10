# VisibleVoices - ASL Learning Platform

A modern, AI-powered platform for learning and practicing American Sign Language.

## Features

- 🎓 **Learn ASL**: Comprehensive guide to ASL alphabet and basics
- 📹 **Practice with AI**: Real-time sign recognition using your webcam
- 🎯 **Instant Feedback**: Get confidence scores and accuracy ratings
- 📱 **Responsive Design**: Works on desktop, tablet, and mobile
- 🎨 **Modern UI**: Clean, minimalist design with beige and mauve tones

## Tech Stack

- **Frontend**: React 18 with Vite
- **Routing**: React Router v6
- **Styling**: Tailwind CSS
- **Animations**: Framer Motion
- **Camera**: react-webcam
- **Backend**: HuggingFace Space (FastAPI + EfficientNet-B3)

## Setup Instructions

### 1. Install Dependencies

```bash
npm install
```

### 2. Configure Backend Endpoint

Open `src/pages/PracticePage.jsx` and update the `BACKEND_URL`:

```javascript
const BACKEND_URL = 'https://YOUR-HUGGINGFACE-SPACE.hf.space/predict'
```

### 3. Run Development Server

```bash
npm run dev
```

Visit `http://localhost:5173` to see the app!

### 4. Build for Production

```bash
npm run build
```

## Deployment

### Deploy to Vercel (Recommended)

1. Install Vercel CLI:
```bash
npm install -g vercel
```

2. Deploy:
```bash
vercel --prod
```

### Deploy to Netlify

1. Build the project:
```bash
npm run build
```

2. Deploy the `dist` folder to Netlify

## Project Structure

```
visible-voices/
├── src/
│   ├── components/
│   │   ├── NavBar.jsx           # Navigation component
│   │   ├── BackgroundASL.jsx    # Decorative background
│   │   └── ASLAlphabetGrid.jsx  # Alphabet learning grid
│   ├── pages/
│   │   ├── LearnPage.jsx        # Landing/learning page
│   │   ├── PracticePage.jsx     # Practice with camera
│   │   └── NotFound.jsx         # 404 page
│   ├── App.jsx                  # Main app with routing
│   ├── main.jsx                 # Entry point
│   ├── index.css                # Global styles
│   └── App.css                  # Component styles
├── index.html
├── package.json
├── vite.config.js
├── tailwind.config.js
└── postcss.config.js
```

## Color Palette

- **Background**: `#F5F1ED` (Beige)
- **Accent**: `#E8D5D3` (Mauve)
- **Text**: Stone shades (600-800)

## Camera Permissions

The app requires camera access for the practice feature. Browsers will prompt users for permission when they click "Start Camera".

## Tips for Best Results

1. **Good Lighting**: Ensure your hand is well-lit and clearly visible
2. **Plain Background**: Use a simple, contrasting background
3. **Center Frame**: Position your hand in the center of the camera view
4. **Steady Position**: Hold your hand steady when capturing

## Future Enhancements

- [ ] Support for full ASL words and phrases
- [ ] Progress tracking and history
- [ ] User accounts and profiles
- [ ] Multi-language support
- [ ] Mobile app versions
- [ ] Community features and challenges

## License

MIT License - feel free to use this for learning purposes!

## Credits

Built with ❤️ for the ASL community
