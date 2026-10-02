# Fine-tuning do AST em PyTorch

Projeto para fine-tuning do modelo [`MIT/ast-finetuned-audioset-10-10-0.4593`](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) em arquivos locais. Os pesos do backbone AST ficam congelados e somente a cabeça `Linear + Softmax` é treinada para as classes do dataset.

## Formato dos dados

O treino usa o `FMADataset` e espera o seguinte layout:

```text
datasets/
├── fma_small/
│   └── 000/000002.mp3
└── fma_tracks.csv
```

O `fma_tracks.csv` é usado para obter o gênero (`genre_top`) e a divisão oficial de treino/avaliação do FMA. Use `--metadata` para informar outro caminho.

## Instalação e treino

No Windows, é preciso instalar o ffmpeg para executar o script.

```powershell
winget install "FFmpeg (Shared)"
```

```powershell
uv sync
uv run python -m ast_finetuning.train --data-dir datasets --metadata datasets/fma_tracks.csv --subset small --epochs 10 --batch-size 16
```

O PyTorch e o TorchAudio são obtidos do índice CUDA 13.0 configurado no `pyproject.toml`. O melhor checkpoint fica em `checkpoints/ast-finetuned`. A cada época, as losses e acurácias de treino e validação são impressas e registradas em `checkpoints/ast-finetuned/history.csv`. O treino imprime também a lista de parâmetros que permanecem treináveis, que deve conter apenas `classifier.weight` e `classifier.bias`.


Durante o treinamento e a avaliação, o `FMADataset` retorna diretamente o espectrograma Mel, nunca a forma de onda. A extração usa o mesmo `torchaudio.compliance.kaldi.fbank` do AST original: 128 bins, hop de 10 ms, janela Hanning de 25 ms, pré-ênfase, remoção de offset DC e FFT de 512 pontos. O log-Mel é normalizado de acordo com o `ASTFeatureExtractor` e ajustado para 1024 frames, o tamanho esperado pelo AST. O treino usa recorte aleatório como augmentation; validação e inferência usam recorte determinístico.
