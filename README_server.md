# Screen Correction FastAPI Server

## Features
- Supports multiple file uploads.
- Automatically performs screen detection and perspective correction.
- Results are saved in a timestamped output directory.
- Supported formats: JPG, JPEG, PNG, HEIC.

## Installation
```bash
pip install -r requirements.txt
```

## Start the Server
```bash
python fastapi_server.py
```

## API Usage

### 1. API Documentation
Open your browser and visit: http://localhost:8000/docs

### 2. File Upload
Send a POST request to: http://localhost:8000/upload
- Supports multiple file uploads.
- Field name: `files`

### 3. Health Check
Send a GET request to: http://localhost:8000/health

## Example curl Commands
```bash
# Upload a single file
curl -X POST "http://localhost:8000/upload" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "files=@your_image.jpg"

# Upload multiple files
curl -X POST "http://localhost:8000/upload" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "files=@image1.jpg" \
     -F "files=@image2.png"
```

## Output Description
- Results are saved in: `output/YYYYMMDD_HHMMSS/`
- Corrected image filenames: `originalname_corrected_ratio.jpg`
- API responses include processing status and output file paths.