#!/usr/bin/env python3
"""
Smile Detector - Real-time face and smile detection using OpenCV
Detects faces and smiles from webcam or image file.
"""

import cv2
import argparse
import sys
import time
from pathlib import Path


def load_cascade_classifier(model_path, model_name):
    """Load a cascade classifier and validate it loaded successfully."""
    classifier = cv2.CascadeClassifier(model_path)
    if classifier.empty():
        raise SystemExit(f"Error: Could not load {model_name} from '{model_path}'. "
                         f"Please ensure the file exists and is a valid cascade XML.")
    return classifier


def detect_faces(gray_frame, face_detector, scale_factor=1.1, min_neighbors=5):
    """Detect faces in a grayscale frame."""
    faces = face_detector.detectMultiScale(
        gray_frame,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(30, 30)
    )
    return faces


def detect_smiles_in_face(face_roi_gray, smile_detector, scale_factor=1.7, min_neighbors=20):
    """Detect smiles within a face ROI (region of interest)."""
    smiles = smile_detector.detectMultiScale(
        face_roi_gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(25, 25)
    )
    return smiles


def draw_detections(frame, faces, smile_data):
    """Draw bounding boxes and labels on the frame."""
    for (fx, fy, fw, fh) in faces:
        # Draw face rectangle (green)
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (100, 200, 50), 2)
        
        # Check if this face has smiles detected
        face_has_smile = False
        for (face_coords, smiles) in smile_data:
            if face_coords == (fx, fy, fw, fh) and len(smiles) > 0:
                face_has_smile = True
                # Draw smile rectangles (blue) - map back to global coordinates
                for (sx, sy, sw, sh) in smiles:
                    # Convert smile coordinates from face ROI to global frame coordinates
                    global_sx = fx + sx
                    global_sy = fy + sy
                    cv2.rectangle(frame, (global_sx, global_sy), 
                                  (global_sx + sw, global_sy + sh), (255, 100, 100), 2)
                break
        
        # Add "Smiling!" label if smile detected
        if face_has_smile:
            cv2.putText(frame, 'Smiling!', (fx, fy + fh + 30),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.9, color=(0, 255, 0), thickness=2)


def process_frame(frame, face_detector, smile_detector, args):
    """Process a single frame: detect faces and smiles, draw results."""
    # Convert to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = detect_faces(frame_gray, face_detector,
                         scale_factor=args.face_scale_factor,
                         min_neighbors=args.face_min_neighbors)
    
    # For each face, detect smiles within the face ROI
    smile_data = []
    for (fx, fy, fw, fh) in faces:
        face_roi_gray = frame_gray[fy:fy+fh, fx:fx+fw]
        smiles = detect_smiles_in_face(face_roi_gray, smile_detector,
                                        scale_factor=args.smile_scale_factor,
                                        min_neighbors=args.smile_min_neighbors)
        smile_data.append(((fx, fy, fw, fh), smiles))
    
    # Draw all detections
    draw_detections(frame, faces, smile_data)
    
    return frame


def run_webcam_detection(face_detector, smile_detector, args):
    """Run real-time smile detection on webcam feed."""
    webcam = cv2.VideoCapture(args.camera)
    
    if not webcam.isOpened():
        raise SystemExit(f"Error: Could not open camera {args.camera}. "
                         f"Please check your camera connection.")
    
    print(f"Starting webcam detection (camera {args.camera})...")
    print("Press 'q' or ESC to quit.")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        successful_read, frame = webcam.read()
        if not successful_read:
            print("Warning: Failed to read frame from camera.")
            break
        
        # Resize frame for better performance if requested
        if args.resize_width:
            height, width = frame.shape[:2]
            new_width = args.resize_width
            new_height = int(height * (new_width / width))
            frame = cv2.resize(frame, (new_width, new_height))
        
        # Process frame
        frame = process_frame(frame, face_detector, smile_detector, args)
        
        # Calculate and display FPS
        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            frame_count = 0
            start_time = time.time()
        else:
            fps = 0
        
        if fps > 0:
            cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.7, color=(255, 255, 255), thickness=2)
        
        # Show frame
        cv2.imshow('Smile Detector - Press Q to quit', frame)
        
        # Handle key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC
            print("Exiting...")
            break
    
    webcam.release()
    cv2.destroyAllWindows()


def run_image_detection(face_detector, smile_detector, args):
    """Run smile detection on a static image."""
    image_path = Path(args.image)
    if not image_path.exists():
        raise SystemExit(f"Error: Image file '{args.image}' not found.")
    
    print(f"Loading image: {args.image}")
    frame = cv2.imread(str(image_path))
    
    if frame is None:
        raise SystemExit(f"Error: Could not read image '{args.image}'.")
    
    # Process frame
    frame = process_frame(frame, face_detector, smile_detector, args)
    
    # Show result
    cv2.imshow('Smile Detector - Press any key to quit', frame)
    print("Press any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Optionally save output
    if args.output:
        cv2.imwrite(args.output, frame)
        print(f"Saved result to: {args.output}")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Real-time smile detection using OpenCV cascade classifiers.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('-c', '--camera', type=int, default=0,
                             help='Camera device index (0 for default camera)')
    input_group.add_argument('-i', '--image', type=str,
                             help='Path to input image file (instead of webcam)')
    
    # Model paths
    parser.add_argument('--face-cascade', type=str,
                        default='haarcascade_frontalface_default.xml',
                        help='Path to face cascade classifier XML')
    parser.add_argument('--smile-cascade', type=str,
                        default='haarcascade_smile.xml',
                        help='Path to smile cascade classifier XML')
    
    # Detection parameters
    parser.add_argument('--face-scale-factor', type=float, default=1.1,
                        help='Face detection scale factor')
    parser.add_argument('--face-min-neighbors', type=int, default=5,
                        help='Face detection minimum neighbors')
    parser.add_argument('--smile-scale-factor', type=float, default=1.7,
                        help='Smile detection scale factor')
    parser.add_argument('--smile-min-neighbors', type=int, default=20,
                        help='Smile detection minimum neighbors')
    
    # Performance options
    parser.add_argument('--resize-width', type=int, default=None,
                        help='Resize frame width for better performance (e.g., 640)')
    
    # Output options
    parser.add_argument('-o', '--output', type=str,
                        help='Save output image to this path (image mode only)')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Load cascade classifiers with validation
    print("Loading models...")
    try:
        face_detector = load_cascade_classifier(args.face_cascade, "face detector")
        smile_detector = load_cascade_classifier(args.smile_cascade, "smile detector")
        print("✓ Models loaded successfully.")
    except SystemExit as e:
        print(e)
        sys.exit(1)
    
    # Run detection
    try:
        if args.image:
            run_image_detection(face_detector, smile_detector, args)
        else:
            run_webcam_detection(face_detector, smile_detector, args)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print("Done!")


if __name__ == '__main__':
    main()


