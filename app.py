from flask import Flask, request, jsonify
import cv2
from flask_cors import CORS
import numpy as np

app = Flask(__name__)

CORS(app)

def match_fingerprints(sample_image, fingerprint_image):
    # Convert images to grayscale
    sample_gray = cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)
    fingerprint_gray = cv2.cvtColor(fingerprint_image, cv2.COLOR_BGR2GRAY)

    # Initiate ORB detector
    orb = cv2.ORB_create()

    # Find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(sample_gray, None)
    kp2, des2 = orb.detectAndCompute(fingerprint_gray, None)

    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Consider a match if the distance is below a certain threshold
    good_matches = [m for m in matches if m.distance < 50]

    # Calculate the matching score
    score = len(good_matches) / max(len(kp1), len(kp2)) * 100

    return score

@app.route('/compare_fingerprints', methods=['POST'])
def compare_fingerprints():
    # Check if both sample_image and fingerprint_image are present in the request
    if 'sample_image' not in request.files or 'fingerprint_image' not in request.files:
        return jsonify({'error': 'Both sample_image and fingerprint_image are required in the request'}), 400

    # Get the file objects for sample_image and fingerprint_image from the request
    sample_image = request.files['sample_image']
    fingerprint_image = request.files['fingerprint_image']

    # Print the filenames of the uploaded images for debugging
    print("Sample image uploaded:", sample_image.filename)
    print("Fingerprint image uploaded:", fingerprint_image.filename)

    try:
        # Read the images
        sample_image_np = np.frombuffer(sample_image.read(), np.uint8)
        fingerprint_image_np = np.frombuffer(fingerprint_image.read(), np.uint8)

        # Decode the images using OpenCV
        sample_cv2 = cv2.imdecode(sample_image_np, cv2.IMREAD_COLOR)
        fingerprint_cv2 = cv2.imdecode(fingerprint_image_np, cv2.IMREAD_COLOR)

        # Check if the images are decoded properly
        if sample_cv2 is None or fingerprint_cv2 is None:
            raise ValueError("One or both images could not be decoded")

        # Print shapes of the images for debugging
        print("Sample image shape:", sample_cv2.shape)
        print("Fingerprint image shape:", fingerprint_cv2.shape)
    except Exception as e:
        return jsonify({'error': f'Error loading images: {str(e)}'}), 400

    # Perform fingerprint matching
    score = match_fingerprints(sample_cv2, fingerprint_cv2)

    return jsonify({'match_score': score})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
