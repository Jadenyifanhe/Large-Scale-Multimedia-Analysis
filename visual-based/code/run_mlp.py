import os.path as osp
from argparse import ArgumentParser
import time

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from modules import FeatureDataModule, MlpClassifier


def parse_args(argv=None):
    parser = ArgumentParser(__file__, add_help=False)
    parser.add_argument('name')
    parser = FeatureDataModule.add_argparse_args(parser)
    parser = MlpClassifier.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--earlystop_patience', type=int, default=15)
    parser = ArgumentParser(parents=[parser])
    parser.set_defaults(gpus=1, default_root_dir=osp.abspath(
        osp.join(osp.dirname(__file__), '../data/mlp')))
    args = parser.parse_args(argv)
    return args


def main(args):
    data_module = FeatureDataModule(args)
    model = MlpClassifier(args)
    logger = TensorBoardLogger(args.default_root_dir, args.name)

    checkpoint_callback = ModelCheckpoint(
        filename='{epoch}-{step}-{val_acc:.4f}', monitor='val_acc',
        mode='max', save_top_k=-1)

    early_stop_callback = EarlyStopping(
        'val_acc', patience=args.earlystop_patience, mode='max', verbose=True)

    trainer = pl.Trainer.from_argparse_args(
        args, logger=logger, 
        callbacks=[checkpoint_callback, early_stop_callback])

    trainer.fit(model, data_module)
    predictions = trainer.predict(datamodule=data_module, ckpt_path='best')
    df = data_module.test_df.copy()
    df['Category'] = torch.concat(predictions).numpy()
    prediction_path = osp.join(logger.log_dir, 'test_prediction.csv')
    df.to_csv(prediction_path, index=False)
    print('Output file:', prediction_path)

    train_predictions = trainer.predict(dataloaders=data_module.train_dataloader(), ckpt_path='best')
    save_csv(data_module.train_df.copy(), train_predictions, logger, 'train_prediction.csv')

    val_predictions = trainer.predict(dataloaders=data_module.val_dataloader(), ckpt_path='best')
    save_csv(data_module.val_df.copy(), val_predictions, logger, 'val_prediction.csv')
    return


def save_csv(df, predictions, logger, fname=None):
    df['Pred'] = torch.concat(predictions).numpy()
    prediction_path = osp.join(logger.log_dir, fname)
    df.to_csv(prediction_path, index=False)
    print('Output file:', prediction_path)
    return


if __name__ == '__main__':
    start = time.time()
    main(parse_args())
    end = time.time()
    print("Time for computation: ", (end - start))
    
