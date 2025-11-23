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
- ✅ Real-time results display
