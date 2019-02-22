import torch
from model import Decoder,Encoder
import cv2
import numpy as np
#path to encoder and decoder state dict files
path_encoder_dict = ''
path_decoder_dict = ''

# Creating models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = Encoder().to(device)
decoder = Decoder(num_words).to(device)

#loading state dict
encoder.load_state_dict(torch.load(path_encoder_dict))
decoder.load_state_dict(torch.load(path_decoder_dict))

#change models to evaluation mode
encoder.eval()
decoder.eval()

#Load image
path_image = ''
image = cv2.imread(path_image)
image = cv2.resize(image,(224,224))
image = np.reshape(image,(1,3,224,224))
image = torch.from_numpy(image)

#Predict function
def predict(encoder,decoder , image):
    images = images.to(device)
    features = encoder(images)
    outputs = decoder.test(features)
    outputs = outputs.cpu().detach().numpy()
    caption = []
    for i in range(len(caption)):
        caption.append(np.argmax(outputs[i]))
        sampled_caption = []
        for word_id in caption:
            word = reverse_word_map[int(word_id)]
            sampled_caption.append(word)
        if word == '<end>':
            break
        sentence = ' '.join(sampled_caption)
    return sentence

sentence = predict(encoder , decoder, image)
print(sentence)
