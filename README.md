# Face Recognition REST API

Face recognition using the Flask and the [face_recognition](https://github.com/ageitgey/face_recognition) python package.

## Getting started

```console
$ python -m venv env        # Create virtual environment
$ source env/bin/activate   # Source virtual environment
$ make install              # Install dependencies
$ make run                  # Start in dev mode (not production build)
```

## Included

```
facial-recognition
|- app.py               - Flask application entry-point
|- models/              - FaceNet face recognition model
|- Makefile             - For easy setup
|- README.md            - This file
|- requirements.txt     - Dependencies (see above)
```