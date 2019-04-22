import cv2
import requests
import os
import keyboard
ngrok_url = 'http://f07b6690.ngrok.io'
def send_data_to_server(image_path):
    form_data = open(image_path, 'rb')
    print(form_data)
    files = {'file': form_data}
    print(files['file'])
    response = requests.post(ngrok_url, files=files)
    print(response)
camera = cv2.VideoCapture(0)

while True:  # making a loop
    try:  # used try so that if user pressed other than the given key error will not be shown
        if keyboard.is_pressed('q'):  # if key 'q' is pressed
            _, image = camera.read()
            cv2.imwrite('./image.jpg', image)
            send_data_to_server('./image.jpg')
