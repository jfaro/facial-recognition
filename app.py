from flask import Flask, request
from flask_cors import CORS
from PIL import Image
import base64
import io
import os
import face_recognition

app = Flask(__name__)
CORS(app)


@app.route('/api', methods=['POST', 'GET'])
def api():
    data = request.get_json()
    known_faces_dir = './known_faces'
    unknown_faces_dir = './unknown_faces'

    # Record of all known faces
    known_faces = [
        { 'name': 'Julien', 'image': 'julien.jpg' },
        { 'name': 'Barack', 'image': 'barack.jpg' }
    ]

    # Get image from request    
    result = data['data']
    b = bytes(result, 'utf-8')
    image = b[b.find(b'/9'):]
    image = Image.open(io.BytesIO(base64.b64decode(image)))

    # Save unknown image
    unknown_img_filepath = os.path.join(unknown_faces_dir, 'unknown.jpeg')
    image.save(unknown_img_filepath)

    # Find matches
    matched_names = []

    for face in known_faces:
        known_img_filepath = os.path.join(known_faces_dir, face['image'])
        known_img = face_recognition.load_image_file(known_img_filepath)
        unknown_img = face_recognition.load_image_file(unknown_img_filepath)

        known_encoding = face_recognition.face_encodings(known_img)
        unknown_encoding = face_recognition.face_encodings(unknown_img)

        if len(known_encoding) == 0 or len(unknown_encoding) == 0:
            continue
        
        known_encoding = known_encoding[0]
        unknown_encoding = unknown_encoding[0]

        results = face_recognition.compare_faces([known_encoding], unknown_encoding)
        if len(results) > 0:
            if results[0]:
                matched_names.append(face['name'])
   
    return { 'results': matched_names }


if __name__ == '__main__':
    app.run()
