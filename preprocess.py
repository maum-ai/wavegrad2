#This code is adopted from
#https://github.com/ming024/FastSpeech2
import argparse

from omegaconf import OmegaConf as OC

from preprocessor.preprocessor import Preprocessor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help="path to preprocess.yaml")
    args = parser.parse_args()

    hparams = OC.load(args.config)
    preprocessor = Preprocessor(hparams)
    preprocessor.build_from_path()
