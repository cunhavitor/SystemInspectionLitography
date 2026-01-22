# Inspection Vision Camera

A python-based system for visual inspection and dataset preparation.

## Features
- **Inspection Mode**: Real-time camera feed (logic to be added).
- **Dataset Mode**: Capture images from camera and save them to `data/raw`.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure settings in `config/settings.yaml`.

## Usage

### Run Application (Recommended)
Use the provided script which automatically handles camera drivers (e.g., on Raspberry Pi):
```bash
./run.sh
```

### Run Manually
```bash
source venv/bin/activate
python main.py
```
- Press `s` to save an image.
- Press `q` to quit.
