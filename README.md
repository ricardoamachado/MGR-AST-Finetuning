# Fine-tuning do AST em PyTorch

Projeto para fine-tuning do modelo [`MIT/ast-finetuned-audioset-10-10-0.4593`](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) em arquivos locais. Há duas metodologias de ajuste fino, escolhidas pela flag `--finetuning`:

- `head` (padrão): os pesos do backbone AST ficam congelados e somente a cabeça `Linear + Softmax` é treinada para as classes do dataset.
- `lora`: os pesos originais do AST continuam congelados, mas adaptadores LoRA de baixo rank são injetados (via [`peft`](https://github.com/huggingface/peft)) nas projeções de atenção do backbone e treinados junto com a cabeça `Linear + Softmax`.

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

O PyTorch e o TorchAudio são obtidos do índice CUDA 13.0 configurado no `pyproject.toml`. O melhor checkpoint fica em `checkpoints/ast-finetuned`. A cada época, as losses e acurácias de treino e validação são impressas e registradas em `checkpoints/ast-finetuned/history.csv`. O treino imprime também a quantidade de parâmetros treináveis; no modo `head`, imprime a lista deles, que deve conter apenas `classifier.weight` e `classifier.bias`.

### Métricas de desempenho

Todas as métricas são calculadas no conjunto de validação e salvas em `--output-dir`:

| Arquivo | Conteúdo |
| --- | --- |
| `history.csv` | Por época: losses, acurácias e `validation_precision_micro`, `validation_recall_micro` e `validation_f1_micro`. |
| `per_class_metrics.csv` | Precision, recall, F1 e suporte por classe na época de melhor acurácia de validação. |
| `confusion_matrix.csv` | Matriz de confusão normalizada por linha da melhor época: a célula (`true_label`=i, coluna j) é P(predito = j \| verdadeiro = i). |

Como cada faixa tem um único gênero, precision, recall e F1 micro-averaged são iguais à acurácia. Os arquivos da melhor época são sobrescritos sempre que um novo checkpoint é salvo.

### Ajuste fino com LoRA

```powershell
uv run python -m ast_finetuning.train --data-dir datasets --metadata datasets/fma_tracks.csv --subset small --finetuning lora --output-dir checkpoints/ast-lora
```

Hiperparâmetros disponíveis:

| Flag | Padrão | Descrição |
| --- | --- | --- |
| `--lora-rank` | `8` | Rank `r` das matrizes `A` e `B`. |
| `--lora-alpha` | `16` | Escala `alpha`; a atualização é multiplicada por `alpha / r`. |
| `--lora-dropout` | `0.1` | Dropout aplicado na entrada dos adaptadores. |
| `--lora-target-modules` | `q_proj v_proj` | Camadas lineares do AST adaptadas. Opções: `q_proj`, `k_proj`, `v_proj`, `o_proj` (atenção) e `fc1`, `fc2` (MLP). |

Com os valores padrão, cerca de 300 mil parâmetros são treinados (≈0,35% do modelo). No modo `lora` o backbone fica em modo de treino para que o dropout dos adaptadores seja aplicado. A metodologia e os hiperparâmetros do LoRA são salvos no `config.json` do checkpoint, de modo que o `predict` reconstrói o modelo correto sem flags adicionais. Use `--output-dir` diferentes para não sobrescrever o checkpoint de uma metodologia com o da outra.


Durante o treinamento e a avaliação, o `FMADataset` retorna diretamente o espectrograma Mel, nunca a forma de onda. A extração usa o mesmo `torchaudio.compliance.kaldi.fbank` do AST original: 128 bins, hop de 10 ms, janela Hanning de 25 ms, pré-ênfase, remoção de offset DC e FFT de 512 pontos. O log-Mel é normalizado de acordo com o `ASTFeatureExtractor` e ajustado para 1024 frames, o tamanho esperado pelo AST. O treino usa recorte aleatório como augmentation; validação e inferência usam recorte determinístico.
