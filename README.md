# ToP-ViM

[NeurIPS 2024] Exploring Token Pruning in Vision State Space Models

## Envs Setup

```bash
#Python 3.10
conda create -n ToP-ViM python=3.10 -y
conda activate ToP-ViM

# torch 2.1.1 + cu118
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
# python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
# conda install -c conda-forge cudatoolkit-dev

pip install -r vim/vim_requirements.txt

pip install -e causal-conv1d
pip install -e mamba-1p1p1
```

### Model Weights

| Model | #param. | Top-1 Acc. | Top-5 Acc. | Hugginface Repo |
|:------------------------------------------------------------------:|:-------------:|:----------:|:----------:|:----------:|
| [Vim-tiny](https://huggingface.co/hustvl/Vim-tiny-midclstok)    |       7M       |   76.1   | 93.0 | https://huggingface.co/hustvl/Vim-tiny-midclstok |
| [Vim-tiny<sup>+</sup>](https://huggingface.co/hustvl/Vim-tiny-midclstok)    |       7M       |   78.3   | 94.2 | https://huggingface.co/hustvl/Vim-tiny-midclstok |
| [Vim-small](https://huggingface.co/hustvl/Vim-small-midclstok)    |       26M       |   80.5   | 95.1 | https://huggingface.co/hustvl/Vim-small-midclstok |
| [Vim-small<sup>+</sup>](https://huggingface.co/hustvl/Vim-small-midclstok)    |       26M       |   81.6   | 95.4 | https://huggingface.co/hustvl/Vim-small-midclstok |
| [Vim-base](https://huggingface.co/hustvl/Vim-base-midclstok)    |       98M       |   81.9   | 95.8 | https://huggingface.co/hustvl/Vim-base-midclstok |


### Snapshot Download
```bash
pip install huggingface_hub hf_transfer

python hf-snapshot-download.py
```

### Evaluation on Provided Weights
```bash
sh vim/scripts/eval-vim-s.sh 
```

### Train Your Vim / at Finer Granularity
```bash
sh vim/scripts/pt-vim-t.sh

sh vim/scripts/ft-vim-t.sh
```

## Acknowledgement :heart:
This project is based on Mamba ([paper](https://arxiv.org/abs/2312.00752), [code](https://github.com/state-spaces/mamba)), Causal-Conv1d ([code](https://github.com/Dao-AILab/causal-conv1d)), DeiT ([paper](https://arxiv.org/abs/2012.12877), [code](https://github.com/facebookresearch/deit)), Vision Mamba ([paper](https://arxiv.org/abs/2401.09417), [code](https://github.com/hustvl/Vim)). Thanks for their wonderful works.

## Citation
If you find Vim is useful in your research or applications, please consider giving us a star 🌟 and citing it by the following BibTeX entry.

```bibtex
@misc{zhan2024exploringtokenpruningvision,
      title={Exploring Token Pruning in Vision State Space Models}, 
      author={Zheng Zhan and Zhenglun Kong and Yifan Gong and Yushu Wu and Zichong Meng and Hangyu Zheng and Xuan Shen and Stratis Ioannidis and Wei Niu and Pu Zhao and Yanzhi Wang},
      year={2024},
      eprint={2409.18962},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.18962}, 
}
```
