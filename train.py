import yaml
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from data.text_image_dm import TextImageDataModule
from models import CLIPWrapper
import torch


def main(hparams):
    config_dir = 'models/configs/ViT.yaml' if 'ViT' in hparams.model_name else 'models/configs/RN.yaml'
    with open(config_dir) as fin:
        config = yaml.safe_load(fin)[hparams.model_name]

    if hparams.minibatch_size < 1:
        hparams.minibatch_size = hparams.batch_size

    model = CLIPWrapper(hparams.model_name, config, hparams.minibatch_size)
    del hparams.model_name
    dm = TextImageDataModule.from_argparse_args(hparams)
    trainer = Trainer.from_argparse_args(hparams, precision=16, max_epochs=100, weights_save_path="/content/drive/Shareddrives/DiffusionTraining/archive/lightening-clip")
    trainer.fit(model, dm)
    torch.save(model.state_dict(), "/content/drive/Shareddrives/DiffusionTraining/archive/lightening-clip/model.pt")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--minibatch_size', type=int, default=0)
    parser = TextImageDataModule.add_argparse_args(parser)


    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
