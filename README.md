# WaveGrad 2 &mdash; Unofficial PyTorch Implementation

**WaveGrad 2: Iterative Refinement for Text-to-Speech Synthesis**<br>
Unofficial PyTorch+[Lightning](https://github.com/PyTorchLightning/pytorch-lightning) Implementation of Chen *et al.*(JHU, Google Brain), [WaveGrad2](https://arxiv.org/abs/2106.09660).<br>

##TODO
- [ ] MT + SpecAug


## Requirements
- [Pytorch](https://pytorch.org/) 
- [Pytorch-Lightning](https://github.com/PyTorchLightning/pytorch-lightning)==1.2.10
- The requirements are highlighted in [requirements.txt](./requirements.txt).<br>
- We also provide docker setup [Dockerfile](./Dockerfile).<br>


## Preprocessing

## Training
- run `trainer.py`
```shell script
$ python trainer.py
```

- During training, tensorboard logger is logging loss, spectrogram and audio.
```shell script
$ tensorboard --logdir=./tensorboard --bind_all
```
## Evaluation

## Author
This code is implemented by
- Seungu Han at mindslab [hansw0326@mindslab.ai](mailto:hansw0326@mindslab.ai)
- Junhyeok Lee at mindslab [jun3518@mindslab.ai](mailto:jun3518@mindslab.ai)

## References
- Chen *et al.*, [WaveGrad 2: Iterative Refinement for Text-to-Speech Synthesis](https://arxiv.org/abs/2106.09660)
- Chen *et al.*,[WaveGrad: Estimating Gradients for Waveform Generation](https://arxiv.org/abs/2009.00713)
- Ho *et al.*, [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)

This implementation uses code from following repositories:
- [J.Ho's Official DDPM Implementation](https://github.com/hojonathanho/diffusion)
- [lucidrains' DDPM Pytorch Implementation](https://github.com/lucidrains/denoising-diffusion-pytorch)
- [ivanvovk's WaveGrad Pytorch Implementation](https://github.com/ivanvovk/WaveGrad)
- [lmnt-com's DiffWave Pytorch Implementation](https://github.com/lmnt-com/diffwave)
- [ming024's FastSpeech2 Pytorch Implementation](https://github.com/ming024/FastSpeech2)
- [yanggeng1995's EATS Pytorch Implementation](https://github.com/yanggeng1995/EATS)
- [mindslab's NU-Wave](https://github.com/mindslab-ai/nuwave)
- [Keith Ito's Tacotron implementation](https://github.com/keithito/tacotron)
- [NVIDIA's Tacotron2 implementation](https://github.com/NVIDIA/tacotron2)

