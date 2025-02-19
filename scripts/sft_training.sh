set -x

torchrun -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.prompt_key=question \
    data.response_key=answer \
    data.micro_batch_size_per_gpu=8 \
    model.partial_pretrain=deepseek-ai/deepseek-coder-6.7b-instruct \
    trainer.default_hdfs_dir=hdfs://user/verl/experiments/gsm8k/deepseek-coder-6.7b-instruct/ \
    trainer.project_name=gsm8k-sft \
    trainer.experiment_name=gsm8k-sft-deepseek-coder-6.7b-instruct \
    trainer.total_epochs=4 \
    trainer.logger=['console','wandb']
