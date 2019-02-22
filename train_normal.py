import os
import Textpreprocessing
from model import Decoder,Encoder
import cv2
from torch.nn.utils.rnn import pack_padded_sequence
from dataloader import data_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = Encoder().to(device)
decoder = Decoder(num_words).to(device)
total_step = len(data_loader)

for epoch in range(10):
  for i, (images, captions, lengths) in enumerate(data_loader):
    images = images.to(device)
    captions = captions.to(device)
    targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
    features = encoder(images)
    outputs = decoder(features, captions, lengths)
    loss = criterion(outputs, targets)
    decoder.zero_grad()
    encoder.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 50 == 0:
      print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'.format(epoch, 10, i, total_step, loss.item(), np.exp(loss.item())))
    if i%500==0:
      outputs = outputs.cpu().detach().numpy()
      caption = []
      for i in range(len(outputs)):
        caption.append(np.argmax(outputs[i]))

      sampled_caption = []
      for word_id in caption:
        word = reverse_word_map[int(word_id)]
        sampled_caption.append(word)
        if word == '<end>':
            break
      sentence = ' '.join(sampled_caption)
      print(sentence)
