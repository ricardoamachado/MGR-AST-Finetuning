# Fine-tuning do AST em PyTorch

Projeto para fine-tuning do modelo [`MIT/ast-finetuned-audioset-10-10-0.4593`](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) em arquivos locais. O backbone AST permanece totalmente treinável e recebe uma cabeça `Linear + Softmax` para as classes do dataset.

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

```powershell
uv sync
uv run python -m ast_finetuning.train --data-dir datasets --metadata datasets/fma_tracks.csv --subset small --epochs 10 --batch-size 8
```

O PyTorch e o TorchAudio são obtidos do índice CUDA 13.0 configurado no `pyproject.toml`. O melhor checkpoint fica em `checkpoints/ast-finetuned`. A cada época, as losses e acurácias de treino e validação são impressas e registradas em `checkpoints/ast-finetuned/history.csv`.


Durante o treinamento e a avaliação, o `FMADataset` retorna diretamente o espectrograma Mel, nunca a forma de onda. Cada áudio é convertido para mono, reamostrado para 16 kHz e transformado com 128 bins de frequência, `hop_length=160` (10 ms), janela Hamming de 400 amostras (25 ms) e `n_fft=400`. O resultado é normalizado e ajustado para 1024 frames, o tamanho esperado pelo AST. A perda usa logits (`CrossEntropyLoss`); o Softmax é aplicado para expor as probabilidades na saída.
