from transformers import PretrainedConfig

class SpeechLLMModelConfig(PretrainedConfig):
    model_type = "custom_model"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.audio_enc_dim = 1024
        self.llm_dim = 2048

        self.audio_processor_name = "microsoft/wavlm-base"
        self.audio_encoder_name = 'microsoft/wavlm-large'
        self.llm_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.llm_model_checkpoint = "hf_repo/llm_model_checkpoint"
