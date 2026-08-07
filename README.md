# AI-Based Early Cancer Detection and Risk Prediction for Erode District

A research-oriented healthcare web application leveraging artificial intelligence for early cancer detection and risk prediction.

## Technology Stack

- **Backend**: Python 3.13+, Flask 3.0.0
- **Frontend**: Bootstrap 5.3.2, HTML5, CSS3
- **Database**: SQLite with Flask-SQLAlchemy
- **Database Migrations**: Flask-Migrate
- **Form Handling**: Flask-WTF, WTForms
- **Environment Management**: python-dotenv
- **AI/ML**: PyTorch, Hugging Face Transformers (to be added in later modules)
- **Image Processing**: OpenCV, Pillow (to be added in later modules)
- **Version Control**: Git

## Project Structure

```
AI-Early-Cancer-Detection/
├── app/
│   ├── __init__.py              # Application factory with error handling
│   ├── models/
│   │   └── __init__.py          # Database models (User, Prediction, MedicalRecord)
│   ├── controllers/
│   │   ├── __init__.py
│   │   └── main.py              # Route handlers for main pages
│   ├── views/
│   │   └── __init__.py          # Views package (templates stored in templates/)
│   ├── templates/
│   │   ├── base.html            # Base template with Bootstrap 5
│   │   ├── index.html           # Home page
│   │   ├── about.html           # About page
│   │   ├── services.html        # Services page
│   │   └── errors/              # Custom error pages
│   │       ├── 404.html
│   │       ├── 403.html
│   │       └── 500.html
│   └── static/
│       ├── css/                 # Custom CSS files
│       ├── js/                  # Custom JavaScript files
│       └── img/                 # Image assets
├── config/
│   └── config.py                # Configuration classes (dev, prod, testing)
├── migrations/                  # Database migration files (Flask-Migrate)
├── tests/                       # Unit and integration tests
├── .env                         # Environment variables
├── .gitignore                   # Git ignore patterns
├── requirements.txt             # Python dependencies
├── run.py                       # Application entry point
└── README.md                    # This file
```

## Prerequisites

- Python 3.13 or higher
- pip (Python package manager)
- Git (for version control)

## Installation Instructions

### Step 1: Clone or Navigate to Project Directory

If cloning from Git:
```bash
git clone <repository-url>
cd AI-Early-Cancer-Detection
```

### Step 2: Create Python Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Why Virtual Environment?**
- Isolates project dependencies from system Python
- Prevents version conflicts between projects
- Makes dependency management easier
- Ensures reproducible builds

### Step 3: Upgrade pip (Recommended)
```bash
python -m pip install --upgrade pip
```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

This will install:
- Flask and Flask extensions
- Database tools (SQLAlchemy, Flask-Migrate)
- Form handling libraries
- Environment variable management

### Step 5: Configure Environment Variables

The `.env` file contains:
```
FLASK_APP=run.py
FLASK_ENV=development
SECRET_KEY=your-secret-key-change-in-production
```

**Important**: Change `SECRET_KEY` to a secure random string in production.

### Step 6: Initialize the Database

**Option 1: Using Flask CLI**
```bash
flask init-db
```

**Option 2: Using Python**
```bash
python -c "from run import app; app.app_context().push(); from app import db; db.create_all(); print('Database initialized!')"
```

This creates the SQLite database with tables for:
- Users
- Predictions
- Medical Records

### Step 7: Run the Application

**Development Mode:**
```bash
python run.py
```

**Or using Flask CLI:**
```bash
flask run
```

The application will be available at:
- http://localhost:5000
- http://127.0.0.1:5000

### Step 8: Verify Installation

Open your browser and navigate to:
- Home: http://localhost:5000
- About: http://localhost:5000/about
- Services: http://localhost:5000/services

## How to Run the Project

### Quick Start (Development)

```bash
# Activate virtual environment
venv\Scripts\activate

# Run the application
python run.py
```

### Production Deployment

For production, use a WSGI server like Gunicorn:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 run:app
```

### Stopping the Application

Press `Ctrl+C` in the terminal where the application is running.

## File Explanations

### Configuration Files

**`.env`**
- Stores environment variables
- Keeps sensitive data out of code
- Contains: FLASK_APP, FLASK_ENV, SECRET_KEY

**`.gitignore`**
- Specifies files to exclude from Git
- Ignores: virtual environment, database files, Python cache, IDE files

**`requirements.txt`**
- Lists all Python dependencies with versions
- Ensures consistent installations across environments
- Used by `pip install -r requirements.txt`

**`config/config.py`**
- Configuration classes for different environments
- DevelopmentConfig: Debug mode, SQLite database
- ProductionConfig: Production settings
- TestingConfig: In-memory database for tests

### Application Files

**`run.py`**
- Application entry point
- Creates Flask app using factory pattern
- Provides CLI commands for database initialization
- Configures shell context for interactive debugging

**`app/__init__.py`**
- Application factory pattern implementation
- Initializes Flask extensions (database, migrations)
- Registers blueprints (route handlers)
- Configures error handlers and logging
- Returns configured Flask app

**`app/models/__init__.py`**
- Database models using SQLAlchemy ORM
- User model: Authentication and patient data
- Prediction model: Cancer detection results
- MedicalRecord model: Patient medical history
- Defines relationships between models

### Controller Files

**`app/controllers/main.py`**
- Flask blueprint for main routes
- Route handlers: index(), about(), services()
- Returns rendered HTML templates
- Can be extended with additional routes

### Template Files

**`app/templates/base.html`**
- Base template using Jinja2 inheritance
- Bootstrap 5 CDN integration
- Navigation bar with responsive design
- Footer with contact information
- CSS styling for consistent look

**`app/templates/index.html`**
- Home page extending base template
- Hero section with gradient background
- Feature cards for cancer detection services
- Statistics section
- Call-to-action buttons

**`app/templates/about.html`**
- About page with mission statement
- Technology stack overview
- Team information
- Project objectives

**`app/templates/services.html`**
- Services page detailing detection capabilities
- Feature cards for each service type
- Step-by-step process explanation
- Service descriptions with icons

**`app/templates/errors/404.html`**
- Custom 404 Not Found error page
- User-friendly error message
- Navigation buttons to home/back

**`app/templates/errors/500.html`**
- Custom 500 Internal Server error page
- Professional error message
- Contact information

**`app/templates/errors/403.html`**
- Custom 403 Forbidden error page
- Access denied message
- Navigation options

## Features

### Current Features (Module 1)

- **Professional Web Interface**: Modern, responsive design with Bootstrap 5
- **MVC Architecture**: Clean separation of concerns
- **Database Integration**: SQLite with SQLAlchemy ORM
- **Error Handling**: Custom error pages for 404, 403, 500
- **Configuration Management**: Environment-based configuration
- **Logging**: Application logging for debugging
- **Navigation**: Responsive navigation with mobile support

### Planned Features (Future Modules)

- **User Authentication**: Registration, login, password management
- **Lung Cancer Detection**: AI-powered X-ray analysis
- **Breast Cancer Detection**: Mammography analysis
- **Skin Cancer Detection**: Image-based detection
- **Risk Prediction**: Comprehensive risk assessment
- **Medical Records**: Secure record management
- **Expert Consultation**: Doctor-patient communication

## Architecture

The application follows the Model-View-Controller (MVC) pattern:

### Models (app/models/)
- Represent database tables
- Handle data validation
- Define relationships between entities
- Business logic for data operations

### Views (app/templates/)
- HTML templates with Jinja2
- Bootstrap 5 for styling
- Responsive design
- User interface components

### Controllers (app/controllers/)
- Flask blueprints for route organization
- Handle HTTP requests/responses
- Coordinate between models and views
- Implement business logic

## Development Guidelines

### Code Style
- Follow PEP 8 for Python code
- Use descriptive variable names
- Add docstrings to functions
- Keep functions focused and small

### Git Workflow
- Create feature branches for new features
- Commit frequently with descriptive messages
- Pull before pushing
- Use meaningful branch names

### Testing
- Write unit tests for models
- Test controller routes
- Validate form inputs
- Test error handling

## Troubleshooting

### Common Issues

**Issue: Module not found error**
```bash
Solution: Ensure virtual environment is activated
venv\Scripts\activate
```

**Issue: Database locked**
```bash
Solution: Delete the .db file and reinitialize
del cancer_detection_dev.db
python -c "from run import app; app.app_context().push(); from app import db; db.create_all()"
```

**Issue: Port already in use**
```bash
Solution: Use a different port
python run.py  # Change port in run.py or use: flask run --port 5001
```

**Issue: Template not found**
```bash
Solution: Ensure templates folder is in app/ directory
```

## Security Considerations

- Change SECRET_KEY in production
- Use HTTPS in production
- Implement user authentication (future module)
- Validate all user inputs
- Use parameterized queries (SQLAlchemy handles this)
- Keep dependencies updated

## License

This project is for research purposes.

## Contact

For questions or support, contact the development team.
