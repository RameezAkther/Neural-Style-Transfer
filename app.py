import os
import uuid
import sys
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO
import subprocess
import cv2
from werkzeug.utils import secure_filename
from src.nst import perform_style_transfer
from src.magenta_style_transfer import stylize_image

app = Flask(__name__)
socketio = SocketIO(app)

UPLOAD_FOLDER_RNST = './static/uploads/rnst'
RESULT_FOLDER_RNST = './static/results/rnst'

UPLOAD_FOLDER_MAGENTA = './static/uploads/magenta'
RESULT_FOLDER_MAGENTA = './static/results/magenta'

UPLOAD_FOLDER_FNST = './static/uploads/fnst'
RESULT_FOLDER_FNST = './static/results/fnst'

MODEL_FOLDER = './models/fast_style_transfer_model'

os.makedirs(UPLOAD_FOLDER_RNST, exist_ok=True)
os.makedirs(RESULT_FOLDER_RNST, exist_ok=True)
os.makedirs(UPLOAD_FOLDER_MAGENTA, exist_ok=True)
os.makedirs(RESULT_FOLDER_MAGENTA, exist_ok=True)
os.makedirs(UPLOAD_FOLDER_FNST, exist_ok=True)
os.makedirs(RESULT_FOLDER_FNST, exist_ok=True)

@app.route('/')
def index():
    return render_template('regular_nst.html')

@app.route('/magenta-style-transfer')
def magenta_style_transfer_page():
    return render_template('magenta_model_nst.html')

@app.route("/magenta-style-transfer", methods=["POST"])
def magenta_style_transfer():
    """Perform style transfer using Google's Magenta model."""
    if "content_image" not in request.files or "style_image" not in request.files:
        return jsonify({"error": "Both content and style images are required"}), 400

    content_file = request.files["content_image"]
    style_file = request.files["style_image"]

    if content_file.filename == "" or style_file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Save uploaded files
    content_path = os.path.join(UPLOAD_FOLDER_MAGENTA, secure_filename(content_file.filename))
    style_path = os.path.join(UPLOAD_FOLDER_MAGENTA, secure_filename(style_file.filename))

    content_file.save(content_path)
    style_file.save(style_path)

    # Apply style transfer with unique filename
    try:
        output_path, unique_filename = stylize_image(content_path, style_path, RESULT_FOLDER_MAGENTA)
        return jsonify({"image_url": f"/static/results/magenta/{unique_filename}"})
    except Exception as e:
        print(e)
        return jsonify({"error": f"Style transfer failed: {str(e)}"}), 500

@app.route('/fast-style-transfer')
def fast_style_transfer_page():
    return render_template('fast_nst.html')

@app.route('/fast-style-transfer', methods=['POST'])
def fast_style_transfer():
    """Handles fast NST by running the neural style script."""
    if "content_image" not in request.files or "style_model" not in request.form:
        return jsonify({"error": "Content image and style model are required"}), 400

    content_file = request.files["content_image"]
    style_model = request.form["style_model"]

    if content_file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Secure filename and save
    unique_id = uuid.uuid4().hex  # Generate a unique ID
    content_filename = secure_filename(content_file.filename)
    content_path = os.path.join(UPLOAD_FOLDER_FNST, content_filename)

    output_filename = f"styled_{unique_id}.jpg"  # Unique output filename
    output_path = os.path.join(RESULT_FOLDER_FNST, output_filename)

    content_file.save(content_path)

    # Run the fast style transfer command
    model_path = os.path.join(MODEL_FOLDER, style_model)
    command = [
        sys.executable, "src/fast_neural_style/neural_style.py", "eval",
        "--content-image", content_path,
        "--model", model_path,
        "--output-image", output_path,
        "--cuda", "0"
    ]

    try:
        subprocess.run(command, check=True)
        return jsonify({"image_url": f"/static/results/fnst/{output_filename}"})
    except subprocess.CalledProcessError as e:
        return jsonify({"error": "Style transfer failed"}), 500
    
# @app.route("/static/results/<path:filename>")
# def send_styled_image(filename):
#     """Serve the generated styled image."""
#     return send_from_directory(RESULT_FOLDER, filename)

@app.route("/upload", methods=["POST", "GET"])
def upload():
    print(request.files)  # Debugging: Print the files received
    if "content_img" not in request.files or "style_img" not in request.files:
        return jsonify({"error": "Both content and style images are required"}), 400

    content_file = request.files["content_img"]
    style_file = request.files["style_img"]

    if content_file.filename == "" or style_file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Secure filenames and save them
    content_path = os.path.join(UPLOAD_FOLDER_RNST, secure_filename(content_file.filename))
    style_path = os.path.join(UPLOAD_FOLDER_RNST, secure_filename(style_file.filename))
    
    content_file.save(content_path)
    style_file.save(style_path)

    content_weight = request.form.get('content_weight')
    style_weight = request.form.get('style_weight')
    preserve_color = request.form.get('preserve_color')
    num_iterations = request.form.get('num_iterations')

    # Generate a unique ID for this session
    session_id = uuid.uuid4().hex 
    print(content_path, style_path, session_id)
    # Start style transfer in the background
    socketio.start_background_task(run_nst, content_path, style_path, session_id, content_weight, style_weight, preserve_color, num_iterations)

    return jsonify({"message": "Style transfer started!", "session_id": session_id})

def run_nst(content_path, style_path, session_id, content_weight, style_weight, preserve_color, num_iterations):
    """Runs NST and emits updates to the frontend."""
    content_img = cv2.imread(content_path)
    style_img = cv2.imread(style_path)

    output_dir = os.path.join(RESULT_FOLDER_RNST, session_id)
    os.makedirs(output_dir, exist_ok=True)
    final_path = perform_style_transfer(content_img, style_img, socketio, output_dir, session_id, content_weight, style_weight, preserve_color, num_iterations)

    # Emit final image
    socketio.emit("final_image", {"image_url": "/" + final_path, "session_id": session_id})

# @app.route("/static/output/<path:filename>")
# def send_result(filename):
#     """Serve the generated images."""
#     return send_from_directory(RESULT_FOLDER, filename)

if __name__ == "__main__":
    socketio.run(app, debug=True)
