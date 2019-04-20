import torchvision.models as models
import torch
import torch.nn as nn

class Encoder_attention(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        resnet = models.resnet50(pretrained=True)
        f_extractor = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*f_extractor)
    def forward(self, x):
        return self.resnet(x)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        resnet = models.resnet50(pretrained=True)
        f_extractor = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*f_extractor)
        self.linear =  nn.Linear(2048,128)
    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        print(x.unsqueeze(1))
        x = self.linear(x)
        return x
class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=64, num_layers=20, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        self.softmax =  nn.LogSoftmax(dim=1)
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        outputs = self.softmax(outputs)
        return outputs
    def test(features):
        hiddens, _ = self.lstm(features)
        outputs = self.linear(hiddens[0])
        return outputs
