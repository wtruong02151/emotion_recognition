# emotion_recognition
CS295K: Deep Learning Final Project 

To crop the original Chicago face data call:
python image_crop.py

To train, cd into /tensor and run:
python train.py path_to_jpgs T

The second parameter is 'T' or 'F' which tells the program to train the net based on the images found in path_to_file, or load from a checkpoint. The path to the checkpoint is statically defined in the /static directory.

To run the server, cd in to the main directory and run python temp_server.py. Go to http://127.0.0.1:5000.
