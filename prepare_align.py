import argparse

from omegaconf import OmegaConf as OC

from preprocessor import ljspeech


def main(args):
    hparams = OC.load(args.config)
    if "LJSpeech" in hparams.dataset:
        ljspeech.prepare_align(hparams)
    # if "AISHELL3" in config["dataset"]:
    #     aishell3.prepare_align(config)
    # if "LibriTTS" in config["dataset"]:
    #     libritts.prepare_align(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help="path to preprocess.yaml")
    args = parser.parse_args()

    main(args)
