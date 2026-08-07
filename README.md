# AI-Based Early Cancer Detection and Risk Prediction for Erode District

A research-oriented healthcare web application leveraging artificial intelligence for early cancer detection and risk prediction.

## Technology Stack

- **Backend**: Python, Flask
- **Frontend**: Bootstrap 5, HTML5, CSS3
- **Database**: SQLite
- **AI/ML**: PyTorch, Hugging Face Transformers
- **Image Processing**: OpenCV, Pillow
- **Version Control**: Git

## Project Structure

```
AI-Early-Cancer-Detection/
├── app/
│   ├── __init__.py          # Application factory
│   ├── models/              # Database models
│   ├── controllers/         # Route handlers
│   ├── views/               # Templates
│   ├── templates/           # HTML templates
│   └── static/              # CSS, JS, images
├── config/                  # Configuration files
├── migrations/              # Database migrations
├── tests/                   # Unit tests
├── requirements.txt         # Python dependencies
├── run.py                   # Application entry point
└── README.md               # This file
```

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Initialize the database:
```bash
flask init-db
```

4. Run the application:
```bash
python run.py
```

The application will be available at `http://localhost:5000`

## Features

- **Lung Cancer Detection**: AI-powered analysis of chest X-rays and CT scans
- **Breast Cancer Detection**: Mammography analysis using deep learning
- **Skin Cancer Detection**: Dermatological image analysis
- **Risk Prediction**: Comprehensive risk assessment based on patient data
- **Medical Records**: Secure storage of patient medical history
- **Expert Consultation**: Connect with healthcare professionals

## Architecture

The application follows the Model-View-Controller (MVC) pattern:

- **Models**: Database schema and data relationships
- **Views**: HTML templates with Bootstrap 5
- **Controllers**: Flask blueprints handling HTTP requests

## License

This project is for research purposes.
