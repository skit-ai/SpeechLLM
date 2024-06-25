from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from trainer import SpeechLLMLightning
from dataset import InstructionalAudioDataset

import torch.utils.data as data_utils
from dataset import InstructionalAudioDataset, MyCollator

if __name__ == "__main__":
    
    model_config = {
                'audio_enc_dim': 1024, 
                'llm_dim': 2048, 
                'audio_encoder_name': "microsoft/wavlm-large", #"facebook/hubert-xlarge-ll60k",
                'connector_name': 'cnn',
                'llm_name': "TinyLlama/TinyLlama-1.1B-Chat-v1.0", #"google/gemma-2b-it", #"TinyLlama/TinyLlama-1.1B-Chat-v1.0", #"microsoft/phi-2", 
                'finetune_encoder': False,
                'connector_k': 2,
                'use_lora': True,
                'lora_r': 8,
                'lora_alpha': 16,
                'max_lr': 3e-4,
                'total_training_step': 1000000,
                'warmup_steps': 100,
                'train_batch_per_epoch': 10000,
                'grad_accumulate_steps': 8
        }  

    model = SpeechLLMLightning.load_from_checkpoint("./path-to-checkpoint-dir/best_checkpoint.ckpt")
    tokenizer = model.llm_tokenizer

    test_dataset = InstructionalAudioDataset(
        csv_file='./path-to-test-dir/librispeech-test-clean.csv', # same train.csv and dev.csv
        mode='test'
        )
    
    my_collator = MyCollator(model_config['audio_encoder_name'], tokenizer)
    test_loader = data_utils.DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=my_collator, num_workers=3)
    
    trainer = Trainer(
        accelerator='gpu', devices=1
    )
    trainer.test(model=model, dataloaders=test_loader)
    