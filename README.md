# 🌾 Agricultural AI Platform - Proof of Concept

A working web application with three main AI modules for agriculture:
- **Disease Detection**: Upload plant images for AI disease analysis
- **Terrain Quality**: Input sensor data to assess soil conditions
- **Plants Analysis**: Monitor plant health with custom AI predictions

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Web App
```bash
python web_app.py
```

The app will start at: **http://localhost:5000**

## Features

### 🦠 Disease Detection
- Upload plant images for analysis
- AI-powered disease detection using U-Net model
- Treatment recommendations
- Visual analysis overlay

### 🌍 Terrain Quality  
- Input sensor data (pH, moisture, temperature, humidity)
- Monitor nutrient levels (NPK)
- Quality score assessment
- Actionable recommendations

### 🌱 Plants Analysis
- Track plant health and growth
- Support for multiple plant types
- Health scoring system
- 30-day growth projections

## Project Structure
```
web_app.py              # Main Flask application
requirements.txt        # Python dependencies
templates/
  ├── index.html       # Main menu page
  ├── disease_detection.html
  ├── terrain_quality.html
  └── plants_analysis.html
```

## API Endpoints

- `GET /` - Main menu
- `GET /disease-detection` - Disease detection page
- `POST /api/predict-disease` - Disease prediction API
- `GET /terrain-quality` - Terrain quality page
- `POST /api/analyze-terrain` - Terrain analysis API
- `GET /plants-analysis` - Plants analysis page
- `POST /api/analyze-plants` - Plants analysis API
- `GET /health` - Health check

## Notes

This is a **proof of concept** demonstrating:
- ✅ Working web interface with 3 main modules
- ✅ Drag-and-drop image upload
- ✅ Sensor data input forms
- ✅ Mock AI analysis (ready for real model integration)
- ✅ Real-time results display

**Ready for integration with:**
- Custom disease detection models
- Real sensor data streams
- Advanced plant analysis algorithms

## Customization

To integrate your own AI models:
1. Replace mock predictions in `/api/predict-disease` with your model
2. Update sensor data processing in `/api/analyze-terrain`
3. Implement custom logic in `/api/analyze-plants`

Enjoy! 🚀
