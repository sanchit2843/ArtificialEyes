import cv2
import requests
import os

def send_data_to_server(image_path):  
    form_data = open(image_path, 'rb')
    print(form_data)
    files = {'file': form_data}
    print(files['file'])
    response = requests.post('http://6819d7ff.ngrok.io/result', files=files)
    print(response)

send_data_to_server('./upload/test_mnist.png')