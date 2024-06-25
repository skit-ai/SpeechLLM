

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from trainer import SpeechLLMLightning
from dataset import InstructionalAudioDataset, MyCollator
from pytorch_lightning.strategies import DDPStrategy

import torch.utils.data as data_utils
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import wandb

if __name__ == "__main__":
    log_path = 'WavLM-CNN-tinyllama-run1'
    wandb.init(project="mmllm", name=log_path)
    logger = WandbLogger(project="mmllm", name=log_path)

    model_config = {
                'audio_enc_dim': 1024, 
                'llm_dim': 2048, 
                'audio_encoder_name': "microsoft/wavlm-large", 
                'connector_name': 'cnn',
                'llm_name': "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                'finetune_encoder': False,
                'connector_k': 2,
                'use_lora': True,
                'lora_r': 8,
                'lora_alpha': 16,
                'max_lr': 1e-4,
                'total_training_step': 10000000,
                'warmup_steps': 100,
                'train_batch_per_epoch': 10000,
                'grad_accumulate_steps': 8
        }   
    
    model = SpeechLLMLightning(**model_config)
    tokenizer = model.llm_tokenizer

    train_dataset = InstructionalAudioDataset(
        csv_file = './data_samples/train.csv',
        mode='train', 
        random_keys_prob=0.2,
        )

    val_dataset = InstructionalAudioDataset(
        csv_file='./data_samples/dev.csv', 
        mode='test'
        )

    print(len(train_dataset), len(val_dataset))

    my_collator = MyCollator(model_config['audio_encoder_name'], tokenizer)
    train_loader = data_utils.DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=my_collator, num_workers=3)
    val_loader = data_utils.DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=my_collator, num_workers=3)

    checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints", filename=log_path+'-{epoch}', save_top_k=1, monitor="val/loss", save_last=True)
    early_stop_callback = EarlyStopping(monitor="val/loss", min_delta=0.00, patience=10, verbose=False, mode="min")

    trainer = Trainer(
            max_epochs=model_config['total_training_step']//model_config['train_batch_per_epoch'], gpus=2, 
            strategy=DDPStrategy(find_unused_parameters=True),
            limit_train_batches=model_config['train_batch_per_epoch'], 
            limit_val_batches=model_config['train_batch_per_epoch'], 
            log_every_n_steps=model_config['train_batch_per_epoch'], 
            enable_checkpointing=True, 
            callbacks=[checkpoint_callback],
            fast_dev_run=False, logger=logger, 
            accumulate_grad_batches=model_config['grad_accumulate_steps'],
            resume_from_checkpoint=None
    )
    trainer.fit(model, train_loader, val_loader)

