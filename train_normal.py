total_step = len(data_loader)
import os
import cv2
from torch.nn.utils.rnn import pack_padded_sequence
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
    #torch.nn.utils.rnn.pad_packed_sequence(outputs,lengths, batch_first=True)[0]
    if i % 500 == 0:
      print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'.format(epoch, 10, i, total_step, loss.item(), np.exp(loss.item())))
    if i%5000==0:
      outputs = outputs.cpu().detach().numpy()
      caption = []
      for i in range(len(outputs)):
        caption.append(np.argmax(outputs[i]))
        #targets = targets.cpu().detach().numpy()
      sampled_caption = []
      for word_id in caption:
        word = reverse_word_map[int(word_id)]
        sampled_caption.append(word)
        if word == '<end>':
            break
      sentence = ' '.join(sampled_caption)
      print(sentence)
