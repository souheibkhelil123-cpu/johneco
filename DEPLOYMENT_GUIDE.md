# 🚀 Deployment Guide - Plant Disease Detector

Complete guide for deploying your plant disease detection system to production.

## 📋 Pre-Deployment Checklist

### Local Testing
- [ ] Server runs without errors
- [ ] Model loads successfully
- [ ] Web interface accessible on http://localhost:5000
- [ ] API endpoints working
- [ ] All tests passing (see TESTING_GUIDE.md)

### Code Quality
- [ ] No debug mode enabled
- [ ] Error handling implemented
- [ ] Logging configured
- [ ] Requirements.txt up to date
- [ ] Dependencies pinned to specific versions

### Security
- [ ] Remove debug prints
- [ ] Add input validation
- [ ] Implement rate limiting
- [ ] Use environment variables for config
- [ ] Add CORS headers if needed

### Documentation
- [ ] README complete
- [ ] API documentation written
- [ ] Setup instructions clear
- [ ] Deployment steps documented
- [ ] Troubleshooting guide included

## 🌐 Option 1: Local Network Deployment

**Use this to share with others on your network.**

### Step 1: Configure Firewall
```bash
# Windows: Allow Python through firewall
# Settings → Firewall & Network Protection → Allow an app through firewall
# Select python.exe

# Or via PowerShell:
New-NetFirewallRule -DisplayName "Python Flask" -Direction Inbound -Program "C:\Python311\python.exe" -Action Allow
```

### Step 2: Find Your IP Address
```bash
# Windows:
ipconfig

# Look for "IPv4 Address: 192.168.x.x"

# Linux/Mac:
ifconfig | grep "inet "
```

### Step 3: Update Flask Config
```python
# web_interface.py
if __name__ == '__main__':
    app.run(
        host='0.0.0.0',  # Listen on all interfaces
        port=5000,
        debug=False,
        threaded=True
    )
```

### Step 4: Start Server
```bash
python web_interface.py

# Output shows:
# Running on http://0.0.0.0:5000
# Running on http://192.168.x.x:5000
```

### Step 5: Access from Other Machines
```
Open browser on another device:
http://192.168.x.x:5000
```

## ☁️ Option 2: Heroku Deployment

### Step 1: Install Heroku CLI
```bash
# Windows: Download from heroku.com/download

# Or using npm:
npm install -g heroku

# Login:
heroku login
```

### Step 2: Create Heroku App
```bash
# Create app
heroku create your-plant-disease-detector

# Or use existing app
heroku apps
```

### Step 3: Create Procfile
```bash
# Create file: Procfile
cat > Procfile << EOF
web: gunicorn web_interface:app
EOF
```

### Step 4: Create Runtime File
```bash
# Create file: runtime.txt
echo "python-3.11.4" > runtime.txt
```

### Step 5: Update Requirements
```bash
pip install gunicorn
pip freeze > requirements.txt

# Add to requirements.txt if missing:
# gunicorn==21.2.0
```

### Step 6: Initialize Git Repository
```bash
git init
git add .
git commit -m "Initial plant disease detector deployment"
```

### Step 7: Deploy to Heroku
```bash
heroku git:remote -a your-app-name
git push heroku main

# Monitor deployment:
heroku logs --tail
```

### Step 8: Access Your App
```
https://your-app-name.herokuapp.com
```

**Note**: Free tier has 30-second request timeout. Model loading might exceed this.

## ☁️ Option 3: AWS EC2 Deployment

### Step 1: Launch EC2 Instance
```bash
# AWS Console → EC2 → Launch Instance
# Select: Ubuntu 22.04 LTS
# Instance type: t3.medium (or larger)
# Storage: 50GB (for model and OS)
# Security group: Allow 5000 inbound
```

### Step 2: Connect to Instance
```bash
# Download key pair
# Then:
ssh -i "your-key.pem" ubuntu@your-instance-ip
```

### Step 3: Install Dependencies
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python
sudo apt install python3-pip python3-venv -y

# Install system packages
sudo apt install libsm6 libxext6 -y  # For OpenCV
```

### Step 4: Clone Code
```bash
cd ~
git clone https://github.com/yourrepo/plant-disease-detector
cd plant-disease-detector
```

### Step 5: Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 6: Download Model
```bash
# Copy model checkpoint to server
# Either:
# 1. Clone entire repo with model
# 2. Or download separately

# Verify:
ls PlantSeg/work_dirs/segnext_mscan-l_test/iter_1000.pth
```

### Step 7: Install Supervisor
```bash
sudo apt install supervisor -y
```

### Step 8: Create Supervisor Config
```bash
sudo nano /etc/supervisor/conf.d/plant-disease.conf

# Add:
[program:plant-disease]
directory=/home/ubuntu/plant-disease-detector
command=/home/ubuntu/plant-disease-detector/venv/bin/python web_interface.py
autostart=true
autorestart=true
stderr_logfile=/var/log/plant-disease.err.log
stdout_logfile=/var/log/plant-disease.out.log
user=ubuntu
environment=PATH="/home/ubuntu/plant-disease-detector/venv/bin",CUDA_VISIBLE_DEVICES="0"
```

### Step 9: Install Nginx
```bash
sudo apt install nginx -y
```

### Step 10: Configure Nginx
```bash
sudo nano /etc/nginx/sites-available/plant-disease

# Add:
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Handle large file uploads
        client_max_body_size 16M;
    }
}
```

### Step 11: Enable Site
```bash
sudo ln -s /etc/nginx/sites-available/plant-disease /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### Step 12: Start Services
```bash
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start plant-disease

# Check status:
sudo supervisorctl status
```

### Step 13: Setup SSL (Optional)
```bash
sudo apt install certbot python3-certbot-nginx -y
sudo certbot --nginx -d your-domain.com
```

### Access Your App
```
http://your-domain.com (or https:// if SSL enabled)
```

## ☁️ Option 4: Google Cloud Run (Serverless)

### Step 1: Install Google Cloud SDK
```bash
# Download from cloud.google.com/sdk/install

# Or using package manager
# macOS: brew install --cask google-cloud-sdk
# Windows: choco install gcloudsdk -y

# Initialize:
gcloud init
```

### Step 2: Create Dockerfile
```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 5000

# Run application
CMD exec gunicorn --bind 0.0.0.0:${PORT:-5000} web_interface:app
```

### Step 3: Create .dockerignore
```
__pycache__
*.pyc
.git
.gitignore
results/
*.jpg
*.png
.env
```

### Step 4: Build and Test Locally
```bash
docker build -t plant-disease-detector .
docker run -p 5000:5000 plant-disease-detector
```

### Step 5: Push to Container Registry
```bash
# Configure Docker
gcloud auth configure-docker

# Build and push
docker build -t gcr.io/your-project/plant-disease-detector .
docker push gcr.io/your-project/plant-disease-detector
```

### Step 6: Deploy to Cloud Run
```bash
gcloud run deploy plant-disease-detector \
    --image gcr.io/your-project/plant-disease-detector \
    --platform managed \
    --region us-central1 \
    --memory 4Gi \
    --cpu 2 \
    --timeout 600 \
    --set-env-vars "FLASK_ENV=production"
```

### Access Your App
```
https://plant-disease-detector-xxxxx.run.app
```

## 🐳 Option 5: Docker Deployment (Any Server)

### Step 1: Create Dockerfile
```dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-runtime-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

EXPOSE 5000

CMD ["python", "web_interface.py"]
```

### Step 2: Build Image
```bash
docker build -t plant-disease:latest .
```

### Step 3: Run Container
```bash
docker run -d \
    --name plant-disease \
    --gpus all \
    -p 5000:5000 \
    -v $(pwd)/uploads:/app/uploads \
    plant-disease:latest
```

### Step 4: Check Status
```bash
docker ps
docker logs plant-disease
```

### Step 5: Stop Container
```bash
docker stop plant-disease
docker rm plant-disease
```

## 🔐 Security Hardening

### 1. Add Input Validation
```python
# web_interface.py
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'bmp'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    if file.content_length > MAX_FILE_SIZE:
        return jsonify({'error': 'File too large'}), 413
```

### 2. Add Rate Limiting
```python
from flask_limiter import Limiter

limiter = Limiter(
    app=app,
    key_func=lambda: request.remote_addr,
    default_limits=["200 per day", "50 per hour"]
)

@app.route('/api/predict', methods=['POST'])
@limiter.limit("10 per minute")
def predict():
    ...
```

### 3. Add CORS Headers
```python
from flask_cors import CORS

CORS(app, resources={
    r"/api/*": {
        "origins": ["yourdomain.com"],
        "methods": ["POST", "GET"],
        "max_age": 3600
    }
})
```

### 4. Use Environment Variables
```python
import os
from dotenv import load_dotenv

load_dotenv()

DEBUG = os.getenv('FLASK_DEBUG', 'False') == 'True'
MODEL_PATH = os.getenv('MODEL_PATH', 'PlantSeg/work_dirs/segnext_mscan-l_test/iter_1000.pth')
SECRET_KEY = os.getenv('SECRET_KEY', 'change-me-in-production')

app.config['SECRET_KEY'] = SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
```

### 5. Add Logging
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        logger.info(f"Prediction request from {request.remote_addr}")
        # ... prediction logic ...
        logger.info(f"Prediction successful: {disease}")
        return jsonify({'success': True, 'disease': disease})
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return jsonify({'error': 'Prediction failed'}), 500
```

## 📊 Monitoring & Maintenance

### Log Monitoring
```bash
# Real-time logs
tail -f app.log

# Search for errors
grep "ERROR" app.log

# Count requests
grep "Prediction request" app.log | wc -l
```

### Health Checks
```bash
# Monitor endpoint
while true; do
    curl -s http://localhost:5000/health | jq '.'
    sleep 60
done
```

### Automatic Restarts
```bash
# Systemd service (Linux)
sudo systemctl status plant-disease
sudo systemctl restart plant-disease
sudo journalctl -u plant-disease -f
```

## 🔄 CI/CD Pipeline

### GitHub Actions Example
```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Deploy to server
        uses: appleboy/ssh-action@master
        with:
          host: ${{ secrets.HOST }}
          username: ${{ secrets.USERNAME }}
          key: ${{ secrets.SSH_KEY }}
          script: |
            cd plant-disease-detector
            git pull
            source venv/bin/activate
            pip install -r requirements.txt
            sudo supervisorctl restart plant-disease
```

## 📈 Scaling Considerations

### For High Traffic:
1. Use load balancer (Nginx, HAProxy)
2. Run multiple Flask instances
3. Use caching (Redis)
4. Implement request queuing
5. Scale horizontally with Kubernetes

### For Batch Processing:
```python
# Process many images at once
from celery import Celery

celery = Celery(app.name)

@celery.task
def predict_async(image_path):
    disease, conf, idx = predict_disease(image_path)
    return {'disease': disease, 'confidence': conf}
```

## ✅ Post-Deployment Checklist

- [ ] Server running stable
- [ ] Model loading without errors
- [ ] API responding correctly
- [ ] Web UI functional
- [ ] Logs being written
- [ ] Health check passing
- [ ] HTTPS/SSL enabled (if applicable)
- [ ] Firewall rules correct
- [ ] Backup strategy in place
- [ ] Monitoring alerts set up
- [ ] Documentation updated
- [ ] Team trained on management

## 🆘 Troubleshooting

### Port Already in Use
```bash
# Find process on port 5000
lsof -i :5000  # Linux/Mac
netstat -ano | findstr :5000  # Windows

# Kill process
kill -9 <PID>  # Linux/Mac
taskkill /PID <PID> /F  # Windows
```

### Model Not Found
```bash
# Check path
ls -la PlantSeg/work_dirs/segnext_mscan-l_test/

# Download model if missing
# Instructions in main README
```

### GPU Not Detected
```bash
# Check NVIDIA drivers
nvidia-smi

# If not working:
# Install CUDA toolkit from nvidia.com
# Reinstall PyTorch: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Memory Issues
```bash
# Monitor memory
watch -n 1 'free -h'
nvidia-smi

# Reduce batch size in web_interface.py
# Or use CPU: device = 'cpu'
```

## 📚 Additional Resources

- [Flask Deployment](https://flask.palletsprojects.com/deployment/)
- [Docker Documentation](https://docs.docker.com/)
- [AWS EC2 Guide](https://docs.aws.amazon.com/ec2/)
- [Google Cloud Run](https://cloud.google.com/run/docs)
- [Nginx Documentation](https://nginx.org/en/docs/)
- [PyTorch Production Deployment](https://pytorch.org/serve/)

---

**Last Updated**: [Current Date]
**Status**: Ready for Production Deployment ✅
