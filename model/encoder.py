import torch
from torch import nn
from transformers import AutoModel
from speechtokenizer import SpeechTokenizer

def get_audio_encoder(name, finetune_encoder):
    if name == "facebook/hubert-xlarge-ll60k":
        return TransformerAudioEnoder(model_name='facebook/hubert-xlarge-ll60k', finetune=finetune_encoder)
    elif name == "microsoft/wavlm-large":
        return TransformerAudioEnoder(model_name='microsoft/wavlm-large', finetune=finetune_encoder)
    elif name == "openai/whisper-small":
        return WhisperAudioEncoder(finetune=finetune_encoder)
    elif name == 'speech-tokenizer':
        return SpeechTokenizerEnoder(finetune=finetune_encoder)
    elif name == 'audio-clip':
        return AudioCLIPEncoder(finetune=finetune_encoder)
    else:
        raise NotImplementedError
    
class TransformerAudioEnoder(nn.Module):
    def __init__(self, model_name='facebook/hubert-xlarge-ll60k', finetune=False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        for param in self.encoder.parameters():
            param.requires_grad = finetune
            
        for param in self.encoder.encoder.layers[-15:].parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.encoder(x).last_hidden_state


if __name__ == "__main__":
    model = SpeechTokenizerEnoder()
    # print(model)

    x = torch.randn(2, 1, 16000)
    z = model(x)
    print(z.shape)