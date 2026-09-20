# Fine-tuning do AST em PyTorch

Projeto para fine-tuning do modelo [`MIT/ast-finetuned-audioset-10-10-0.4593`](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) em arquivos locais. O backbone AST permanece totalmente treinável e recebe uma cabeça `Linear + Softmax` para as classes do dataset.

## Formato dos dados

A forma mais simples é organizar os áudios por classe:

```text
datasets/
├── rock/arquivo_01.wav
├── rock/arquivo_02.mp3
└── jazz/arquivo_03.flac
```

Alternativamente, use `--metadata caminho.csv`, com as colunas `path,label`. Caminhos relativos no CSV são resolvidos a partir da pasta do CSV.

## Instalação e treino

```powershell
uv sync
uv run python -m ast_finetuning.train --data-dir datasets --epochs 10 --batch-size 8
```

O PyTorch e o TorchAudio são obtidos do índice CUDA 13.0 configurado no `pyproject.toml`. O melhor checkpoint fica em `checkpoints/ast-finetuned`.

## Inferência

```powershell
uv run python -m ast_finetuning.predict datasets/rock/arquivo_01.wav
```

Durante o treinamento, a entrada é convertida para mono, reamostrada para 16 kHz e transformada pelo `ASTFeatureExtractor` no espectrograma Mel esperado pelo AST. A perda usa logits (`CrossEntropyLoss`); o Softmax é aplicado para expor as probabilidades na saída.
