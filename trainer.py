import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

import wandb
import pytorch_lightning as pl
import numpy as np
from jiwer import wer
import torchmetrics
import random
import re
import json

from model.encoder import get_audio_encoder, WhisperAudioEncoder, TransformerAudioEnoder, SpeechTokenizerEnoder, AudioCLIPEncoder
from model.connector import get_connector, LinearConnector, LinearPoolConnector, CNNConnector
from model.llm import get_llm

class SpeechLLMLightning(pl.LightningModule):
    def __init__(self, 
                 audio_enc_dim=512, 
                 llm_dim=2048, 
                 audio_encoder_name="speech-tokenizer",
                 connector_name='linear-pool',
                 llm_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                 finetune_encoder=False,
                 connector_k=5,
                 use_lora=True,
                 lora_r=32,
                 lora_alpha=2,
                 max_lr=3e-4,
                 total_training_step=500000,
                 warmup_steps=1000,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters()

        self.audio_enc_dim = audio_enc_dim
        self.llm_dim = llm_dim
        self.llm_name = llm_name
        self.finetune_encoder = finetune_encoder
        self.use_lora = use_lora

        self.audio_encoder = get_audio_encoder(audio_encoder_name, finetune_encoder)
        self.connector = get_connector(connector_name, audio_enc_dim, llm_dim, connector_k)
        self.llm_tokenizer, self.llm_model = get_llm(llm_name, use_lora, lora_r, lora_alpha)
        
        self.max_lr = max_lr
        self.total_training_step = total_training_step
        self.warmup_steps = warmup_steps
        self.use_embedding_loss = False
        self.num_validation_samples = 5000

    def configure_optimizers(self):
        opt = [
            {"params": self.audio_encoder.parameters(), "lr": 1e-5},
            {"params": self.connector.parameters(), "lr": self.max_lr},
            {"params": self.llm_model.parameters(), "lr": self.max_lr},
        ]
        optimizer = Adam(opt, lr=self.max_lr)
        return optimizer

    def encode(self, mel, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids, return_embedding_loss=False):
        batch_size = mel.shape[0]

        speech_embeds = self.audio_encoder(mel)
        speech_embeds = self.connector(speech_embeds)
        
        embedder = self.llm_model.model.model.embed_tokens
        pre_prompt_embeds = embedder(pre_tokenized_ids)
        post_prompt_embeds = embedder(post_tokenized_ids)
        output_prompt_embeds = embedder(output_tokenized_ids)

        combined_embeds = torch.cat([pre_prompt_embeds, speech_embeds, post_prompt_embeds, output_prompt_embeds], dim=1)
        atts = torch.ones(combined_embeds.size()[:-1], dtype=torch.long).to(combined_embeds.device)

        input_token_length = pre_tokenized_ids.shape[1] + speech_embeds.shape[1] + post_tokenized_ids.shape[1]
        label_ids = torch.cat([
            torch.ones([batch_size, input_token_length], device=combined_embeds.device)*-100,
            output_tokenized_ids
        ], 1).to(combined_embeds.device).to(torch.int64)
        return combined_embeds, atts, label_ids

    def forward(self, embeds, atts, label_ids):
        out = self.llm_model(
            inputs_embeds=embeds,
            attention_mask=atts,
            labels=label_ids,
        )
        return out
    
    def training_step(self, batch, batch_idx):
        mel, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids = batch
        embeds, atts, label_ids = self.encode(mel, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids)
        outputs = self.forward(embeds, atts, label_ids)
        loss =  outputs["loss"]
        self.log("train/loss", loss, on_epoch=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
            mel, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids = batch
            embeds, atts, label_ids = self.encode(mel, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids)
            outputs = self.forward(embeds, atts, label_ids)
            loss = outputs["loss"]
            self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            
            logits = outputs.logits
            predicted_ids = torch.argmax(logits, dim=-1).cpu()

            generated_output_text = self.llm_tokenizer.decode(predicted_ids[0], skip_special_tokens=False)
            target_text = self.llm_tokenizer.decode(output_tokenized_ids[0], skip_special_tokens=False)
            
            extracted_pred = self.extract_prediction_values(generated_output_text)
            extracted_target = self.extract_prediction_values(target_text)

            keys = extracted_target.keys()
            pred_keys = extracted_pred.keys()

            for key in keys:
                if key not in pred_keys:
                    extracted_pred[key] = "NA"

            if 'Transcript' in keys:
                target_transcript = extracted_target['Transcript']
                predicted_transcript = extracted_pred['Transcript']
                wer_metric = wer(target_transcript.lower(), predicted_transcript.lower())
                self.log("val/wer", wer_metric, on_step=False, on_epoch=True, prog_bar=True, logger=True)

            if 'Response' in keys:
                target_transcript = extracted_target['Response']
                predicted_transcript = extracted_pred['Response']
                wer_metric = wer(target_transcript.lower(), predicted_transcript.lower())
                self.log("val/response_wer", wer_metric, on_step=False, on_epoch=True, prog_bar=True, logger=True)

            if 'SpeechActivity' in keys:
                target_isspeech = extracted_target['SpeechActivity']
                predicted_isspeech = extracted_pred['SpeechActivity']
                self.log("val/speech_activity", float(target_isspeech.lower()==predicted_isspeech.lower()), on_step=False, on_epoch=True, prog_bar=True, logger=True)

            if 'Gender' in keys:
                target_gender = extracted_target['Gender']
                predicted_gender = extracted_pred['Gender']
                self.log("val/gender", float(target_gender.lower()==predicted_gender.lower()), on_step=False, on_epoch=True, prog_bar=True, logger=True)

            if 'Emotion' in keys:
                target_emotion = extracted_target['Emotion']
                predicted_emotion = extracted_pred['Emotion']
                self.log("val/emotion", float(target_emotion.lower()==predicted_emotion.lower()), on_step=False, on_epoch=True, prog_bar=True, logger=True)

            if 'Age' in keys:
                target_age = extracted_target['Age']
                predicted_age = extracted_pred['Age']
                self.log("val/age", float(target_age.lower()==predicted_age.lower()), on_step=False, on_epoch=True, prog_bar=True, logger=True)

            if 'Accent' in keys:
                target_accent = extracted_target['Accent']
                predicted_accent = extracted_pred['Accent']
                self.log("val/accent", float(target_accent.lower()==predicted_accent.lower()), on_step=False, on_epoch=True, prog_bar=True, logger=True)

            if batch_idx in self.selected_samples_for_logging:
                sample_idx = self.selected_samples_for_logging.index(batch_idx)
                # Use wandb.log to log prediction and truth texts
                wandb.log({
                    f"val_sample_{sample_idx}_pred": wandb.Html(f"<pre>{str(extracted_pred)}</pre>"), 
                    f"val_sample_{sample_idx}_target": wandb.Html(f"<pre>{str(target_text).replace('<s>', '').replace('</s>', '')}</pre>"),
                    f"val_sample_{sample_idx}_gen": wandb.Html(f"<pre>{generated_output_text.replace('<s>', '').replace('</s>', '')}</pre>"),
                }, commit=False)

            return {"val_loss": loss}
    
    def test_step(self, batch, batch_idx):
            mel, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids = batch
            embeds, atts, label_ids = self.encode(mel, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids)
            outputs = self.forward(embeds, atts, label_ids)
            loss = outputs["loss"]
            self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            
            logits = outputs.logits
            predicted_ids = torch.argmax(logits, dim=-1)

            input_token_length = output_tokenized_ids.shape[1]
            generated_output_text = self.llm_tokenizer.decode(predicted_ids[0], skip_special_tokens=False)
            target_text = self.llm_tokenizer.decode(output_tokenized_ids[0], skip_special_tokens=False)
            
            extracted_pred = self.extract_prediction_values(generated_output_text)
            extracted_target = self.extract_prediction_values(target_text)

            keys = extracted_target.keys()
            pred_keys = extracted_pred.keys()

            for key in keys:
                if key not in pred_keys:
                    extracted_pred[key] = "NA"

            if 'Transcript' in keys:
                target_transcript = extracted_target['Transcript']
                predicted_transcript = extracted_pred['Transcript']
                wer_metric = wer(target_transcript.lower(), predicted_transcript.lower())
                self.log("val/wer", wer_metric, on_step=False, on_epoch=True, prog_bar=True, logger=True)

            if 'Response' in keys:
                target_transcript = extracted_target['Response']
                predicted_transcript = extracted_pred['Response']
                wer_metric = wer(target_transcript.lower(), predicted_transcript.lower())
                self.log("val/response_wer", wer_metric, on_step=False, on_epoch=True, prog_bar=True, logger=True)

            if 'SpeechActivity' in keys:
                target_isspeech = extracted_target['SpeechActivity']
                predicted_isspeech = extracted_pred['SpeechActivity']
                self.log("val/speech_activity", float(target_isspeech.lower()==predicted_isspeech.lower()), on_step=False, on_epoch=True, prog_bar=True, logger=True)

            if 'Gender' in keys:
                target_gender = extracted_target['Gender']
                predicted_gender = extracted_pred['Gender']
                self.log("val/gender", float(target_gender.lower()==predicted_gender.lower()), on_step=False, on_epoch=True, prog_bar=True, logger=True)

            if 'Emotion' in keys:
                target_emotion = extracted_target['Emotion']
                predicted_emotion = extracted_pred['Emotion']
                self.log("val/emotion", float(target_emotion.lower()==predicted_emotion.lower()), on_step=False, on_epoch=True, prog_bar=True, logger=True)

            if 'Age' in keys:
                target_age = extracted_target['Age']
                predicted_age = extracted_pred['Age']
                self.log("val/age", float(target_age.lower()==predicted_age.lower()), on_step=False, on_epoch=True, prog_bar=True, logger=True)

            if 'Accent' in keys:
                target_accent = extracted_target['Accent']
                predicted_accent = extracted_pred['Accent']
                self.log("val/accent", float(target_accent.lower()==predicted_accent.lower()), on_step=False, on_epoch=True, prog_bar=True, logger=True)

            return {"val_loss": loss}
    
    def on_validation_epoch_start(self):
        """Select two random validation samples to log for each epoch."""
        self.selected_samples_for_logging = random.sample(range(self.num_validation_samples), 2)

    
    def extract_dictionary(self, input_string):
        pattern = r'<s>\s*(\{.*?\})\s*</s>'
        match = re.search(pattern, input_string, re.DOTALL)
        if match:
            dict_string = match.group(1)
            dict_string = re.sub(r',\s*}', '}', dict_string)
            try:
                return json.loads(dict_string)
            except json.JSONDecodeError as e:
                return {}
        else:
            return {}
    
    def extract_prediction_values(self, input_string):
        json_str_match = re.search(r'<s>\s*\{.*?\}\s*</s>', input_string)
        try:
            json_str = json_str_match.group(0)
        except:
            json_str = '{}'
        return self.extract_dictionary(json_str)