import torch
from torch import nn
import torchaudio
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer, HubertModel, AutoProcessor, AutoConfig, AutoModel, AutoFeatureExtractor
from .config import SpeechLLMModelConfig
from peft import LoraConfig, get_peft_model

class TransformerAudioEnoder(nn.Module):
    def __init__(self, model_name='microsoft/wavlm-large', finetune=False):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.encoder =  AutoModel.from_config(config)  

    def forward(self, x):
        return self.encoder(x).last_hidden_state
    
    def return_device(self):
        return next(self.parameters()).device


class CNNConnector(nn.Module):
    def __init__(self, in_channels, out_channels, k=2):
        super().__init__()
        self.layer = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(in_channels, out_channels//2, kernel_size=5,
                      stride=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(out_channels//2, out_channels, kernel_size=5,
                      stride=k, padding=0),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=5,
                      stride=1, padding=0),
        )

    def forward(self, x):
        return self.layer(x.transpose(1,2)).transpose(1,2)
    
    
class SpeechLLMModel(PreTrainedModel):
    config_class = SpeechLLMModelConfig

    def __init__(self, config):
        super().__init__(config)
        self.audio_processor = AutoFeatureExtractor.from_pretrained(config.audio_processor_name)
        self.audio_encoder = TransformerAudioEnoder(config.audio_encoder_name)
        self.connector = CNNConnector(config.audio_enc_dim, config.llm_dim)
        
        llm_config = AutoConfig.from_pretrained(config.llm_model_name)
        self.llm_model =  AutoModelForCausalLM.from_config(llm_config)  
        self.llm_tokenizer = AutoTokenizer.from_pretrained(config.llm_model_name)
        self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj', 'gate_proj'],
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
        )
        self.llm_model = get_peft_model(self.llm_model, peft_config)
        self.llm_model = self.llm_model.merge_and_unload()

    def encode(self, speech, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids):
        batch_size = speech.shape[0]

        with torch.no_grad():
            speech_embeds = self.audio_encoder(speech)
            speech_embeds = self.connector(speech_embeds)
            
        embedder = self.llm_model.model.embed_tokens
        pre_prompt_embeds = embedder(pre_tokenized_ids)
        post_prompt_embeds = embedder(post_tokenized_ids)
        output_prompt_embeds = embedder(output_tokenized_ids)

        combined_embeds = torch.cat([pre_prompt_embeds, speech_embeds, post_prompt_embeds, output_prompt_embeds], dim=1)
        atts = torch.ones(combined_embeds.size()[:-1], dtype=torch.long).to(combined_embeds.device)

        input_token_length = pre_tokenized_ids.shape[1] + speech_embeds.shape[1] + post_tokenized_ids.shape[1]
        label_ids = torch.cat([
            torch.ones([batch_size, input_token_length], device=combined_embeds.device) * -100,
            output_tokenized_ids
        ], 1).to(combined_embeds.device).to(torch.int64)
        return combined_embeds, atts, label_ids

    def forward(self, audio_tensor, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids, attention_mask=None):
        combined_embeds, atts, label_ids = self.encode(audio_tensor, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids)
        outputs = self.llm_model(inputs_embeds=combined_embeds, attention_mask=attention_mask)
        return outputs
    
    def generate_meta(self, audio_path=None, audio_tensor=None, instruction="Give me the following information about the audio [Transcript]", max_new_tokens=2000):
        device = self.audio_encoder.return_device()
        pre_speech_prompt = f'''Instruction:
{instruction}

Input: 
<speech>'''
        post_speech_prompt = f'''</speech>

Output:'''
        output_prompt = '\n<s>'

        with torch.no_grad():
            if audio_tensor == None and audio_path != None:
                audio_tensor, sr = torchaudio.load(audio_path)
            audio_tensor = self.audio_processor(audio_tensor.squeeze(), return_tensors="pt", sampling_rate=16000).input_values

            pre_tokenized_ids = self.llm_tokenizer(pre_speech_prompt, padding="do_not_pad", return_tensors='pt', truncation=False, add_special_tokens=False)["input_ids"]
            post_tokenized_ids = self.llm_tokenizer(post_speech_prompt, padding="do_not_pad", return_tensors='pt', truncation=False, add_special_tokens=False)["input_ids"]
            output_tokenized_ids = self.llm_tokenizer(output_prompt, padding="do_not_pad", return_tensors='pt', truncation=False, add_special_tokens=False)["input_ids"]

            combined_embeds, atts, label_ids = self.encode(audio_tensor.to(device), pre_tokenized_ids.to(device), post_tokenized_ids.to(device), output_tokenized_ids.to(device))

            out = self.llm_model.generate(
                inputs_embeds=combined_embeds,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.llm_tokenizer.pad_token_id
            ).cpu().tolist()[0]

            output_text = self.llm_tokenizer.decode(out, skip_special_tokens=True)
            return output_text

