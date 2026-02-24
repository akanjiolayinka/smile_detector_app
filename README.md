# 😊 Smile Detector

Real-time face and smile detection using OpenCV Haar Cascade classifiers. Detects faces and smiles from your webcam or image files with visual feedback.

## Features

- ✅ **Real-time detection** from webcam feed
- ✅ **Static image processing** for photos
- ✅ **Accurate face localization** using Haar Cascades
- ✅ **Smile detection** within detected faces (proper ROI coordinate mapping)
- ✅ **FPS display** for performance monitoring
- ✅ **Configurable detection parameters** via command-line arguments
- ✅ **Robust error handling** and model validation
- ✅ **Performance optimization** with optional frame resizing

## Installation

### Prerequisites

- Python 3.7 or higher
- A webcam (for real-time detection)

### Setup

1. Clone or download this repository:
```bash
cd smile_detector.py
``

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure the cascade XML files are present in the project directory:
   - `haarcascade_frontalface_default.xml` (face detection)
   - `haarcascade_smile.xml` (smile detection)

   These files are included in the repository. If missing, download them from the [OpenCV GitHub repository](https://github.com/opencv/opencv/tree/master/data/haarcascades).

## Usage

### Real-time Webcam Detection (Default)

Run the script with default settings:
```bash
python smile_detector.py
```

Press **`q`** or **`ESC`** to quit.

### Process a Static Image

Detect faces and smiles in an image file:
```bash
python smile_detector.py --image smile.jpg
```

Save the output to a file:
```bash
python smile_detector.py --image smile.jpg --output result.jpg
```

### Advanced Options

#### Specify Camera Device
Use a different camera (e.g., external webcam):
```bash
python smile_detector.py --camera 1
```

#### Adjust Detection Sensitivity
Fine-tune face detection:
```bash
python smile_detector.py --face-scale-factor 1.2 --face-min-neighbors 3
```

Fine-tune smile detection (higher `min-neighbors` = stricter):
```bash
python smile_detector.py --smile-scale-factor 1.5 --smile-min-neighbors 25
```

#### Improve Performance
Resize frames to a smaller width for faster processing:
```bash
python smile_detector.py --resize-width 640
```

#### Custom Model Paths
Use different cascade classifiers:
```bash
python smile_detector.py --face-cascade path/to/face.xml --smile-cascade path/to/smile.xml
```

### Full Command-Line Reference

```
usage: smile_detector.py [-h] [-c CAMERA | -i IMAGE] [--face-cascade FACE_CASCADE]
                         [--smile-cascade SMILE_CASCADE] [--face-scale-factor FACE_SCALE_FACTOR]
                         [--face-min-neighbors FACE_MIN_NEIGHBORS] [--smile-scale-factor SMILE_SCALE_FACTOR]
                         [--smile-min-neighbors SMILE_MIN_NEIGHBORS] [--resize-width RESIZE_WIDTH]
                         [-o OUTPUT]

Real-time smile detection using OpenCV cascade classifiers.

optional arguments:
  -h, --help            show this help message and exit
  -c CAMERA, --camera CAMERA
                        Camera device index (0 for default camera) (default: 0)
  -i IMAGE, --image IMAGE
                        Path to input image file (instead of webcam) (default: None)
  --face-cascade FACE_CASCADE
                        Path to face cascade classifier XML (default: haarcascade_frontalface_default.xml)
  --smile-cascade SMILE_CASCADE
                        Path to smile cascade classifier XML (default: haarcascade_smile.xml)
  --face-scale-factor FACE_SCALE_FACTOR
                        Face detection scale factor (default: 1.1)
  --face-min-neighbors FACE_MIN_NEIGHBORS
                        Face detection minimum neighbors (default: 5)
  --smile-scale-factor SMILE_SCALE_FACTOR
                        Smile detection scale factor (default: 1.7)
  --smile-min-neighbors SMILE_MIN_NEIGHBORS
                        Smile detection minimum neighbors (default: 20)
  --resize-width RESIZE_WIDTH
                        Resize frame width for better performance (e.g., 640) (default: None)
  -o OUTPUT, --output OUTPUT
                        Save output image to this path (image mode only) (default: None)
```

## How It Works

1. **Face Detection**: Uses Haar Cascade classifier to detect faces in each frame
2. **ROI Extraction**: For each detected face, extracts the face region (Region of Interest)
3. **Smile Detection**: Runs smile detection within each face ROI
4. **Coordinate Mapping**: Maps smile coordinates from face ROI back to the global frame
5. **Visualization**: Draws bounding boxes (green for faces, blue for smiles) and labels

## Key Improvements Over Original

### Bug Fixes
- ✅ Fixed ROI coordinate mapping bug (smiles now correctly positioned relative to faces)
- ✅ Added model load validation (graceful error if cascade files missing)
- ✅ Fixed redundant/conflicting detection calls

### Code Quality
- ✅ Restructured into modular functions for readability and maintainability
- ✅ Consistent variable naming conventions
- ✅ Added comprehensive docstrings and comments
- ✅ Removed unused/commented code

### Features Added
- ✅ Command-line argument parsing for flexibility
- ✅ FPS display for performance monitoring
- ✅ Clean exit handling (press 'q' or ESC)
- ✅ Static image mode with optional output saving
- ✅ Frame resizing for performance optimization
- ✅ Configurable detection thresholds

### Robustness
- ✅ Error handling for missing files and camera issues
- ✅ Input validation for all parameters
- ✅ Informative error messages

## Troubleshooting

### "Could not load face/smile detector" Error
- Ensure `haarcascade_frontalface_default.xml` and `haarcascade_smile.xml` are in the same directory as the script
- Or specify custom paths using `--face-cascade` and `--smile-cascade` arguments

### "Could not open camera" Error
- Check that your webcam is connected and not being used by another application
- Try a different camera index: `--camera 1` or `--camera 2`
- On Windows, you may need to grant camera permissions to Python

### Low FPS / Performance Issues
- Use `--resize-width 640` to process smaller frames
- Adjust detection parameters to be less strict (lower `min-neighbors`)
- Close other applications using the camera

### False Positives (Detecting Smiles When Not Smiling)
- Increase `--smile-min-neighbors` (e.g., `25` or `30`)
- Adjust `--smile-scale-factor` (try `1.8` or `2.0`)

### Missing Smiles (Not Detecting Actual Smiles)
- Decrease `--smile-min-neighbors` (e.g., `15` or `18`)
- Try different lighting conditions (avoid harsh shadows)

## Future Enhancements

### Short-term
- [ ] Temporal smoothing to reduce flicker (average detections over N frames)
- [ ] Unit tests for detection functions
- [ ] Logging with verbosity levels

### Medium-term
- [ ] Replace Haar face detector with DNN-based detector (e.g., OpenCV DNN SSD)
- [ ] Use facial landmarks to isolate mouth region for better smile detection
- [ ] Multi-threaded processing (separate capture and inference)
- [ ] Simple web UI using Streamlit or Flask

### Long-term
- [ ] Train custom CNN classifier for smile detection (MobileNet-based)
- [ ] GPU acceleration support
- [ ] Real-time emotion classification beyond just smiles
- [ ] CI/CD pipeline with automated tests

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

This project uses OpenCV, which is licensed under the Apache 2.0 License.

## Acknowledgments

- Haar Cascade classifiers provided by the [OpenCV project](https://opencv.org/)
- Face detection model: `haarcascade_frontalface_default.xml`
- Smile detection model: `haarcascade_smile.xml`

---

**Enjoy detecting smiles! 😊**
