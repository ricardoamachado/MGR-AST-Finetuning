# Fine-tuning do AST em PyTorch

Projeto para fine-tuning do modelo [`MIT/ast-finetuned-audioset-10-10-0.4593`](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) para classificação de gêneros musicais nos datasets FMA e GTZAN. Como os gêneros dos datasets não coincidem, cada um gera um modelo separado, escolhido pela flag `--dataset {fma_small,fma_medium,gtzan}` (padrão `fma_small`).

Há duas metodologias de ajuste fino, escolhidas pela flag `--finetuning`:

- `head` (padrão): os pesos do backbone AST ficam congelados e somente a cabeça `Linear + Softmax` é treinada para as classes do dataset.
- `lora`: os pesos originais do AST continuam congelados, mas adaptadores LoRA de baixo rank são injetados (via [`peft`](https://github.com/huggingface/peft)) nas projeções de atenção do backbone e treinados junto com a cabeça `Linear + Softmax`.

## Formato dos dados

```text
datasets/
├── fma_small/
│   └── 000/000002.mp3
├── fma_medium/                   # opcional, para --dataset fma_medium
├── fma_tracks.csv
├── GTZAN/
│   ├── blues/blues.00000.wav
│   ├── ...
│   ├── rock/rock.00099.wav
│   ├── train_stratified.txt
│   ├── valid_stratified.txt
│   └── test_stratified.txt
└── GTZAN_pitch_shifted/          # gerado automaticamente (cache da augmentation)
```

**FMA:** o `fma_tracks.csv` é usado para obter o gênero (`genre_top`) e a divisão oficial do FMA (coluna `split`). O treino usa `training` e a validação usa apenas `validation`; o split `test` fica reservado para o teste e não influencia a escolha da melhor época. No FMA-Small isso resulta em 6400 faixas de treino e 800 de validação, balanceadas entre os 8 gêneros. Os subconjuntos são aninhados (o `fma_medium` contém todas as faixas do `fma_small`), então `--dataset fma_medium` usa as 25.000 faixas da pasta `datasets/fma_medium/` (19.922 de treino e 2505 de validação). Use `--metadata` para informar outro caminho.

**GTZAN:** não há arquivo de metadados; o gênero é o nome da pasta de cada faixa. O treino usa `train_stratified.txt` e a validação usa `valid_stratified.txt` (`test_stratified.txt` fica reservado para o teste). O arquivo `jazz/jazz.00054.wav`, corrompido na distribuição original do GTZAN, é detectado e ignorado automaticamente, deixando 499 faixas no treino.

### Data augmentation do GTZAN

Como o treino do GTZAN tem apenas 500 faixas, cada faixa de treino é incluída também com pitch shift de um semitom (meio tom) para baixo e para cima, triplicando o conjunto (1497 amostras). A augmentation é aplicada apenas ao treino do GTZAN, nunca à validação nem ao FMA.

O pitch shift (`torchaudio.functional.pitch_shift`, phase vocoder com FFT de 2048 pontos) é lento demais para rodar na CPU a cada época, então as versões deslocadas são geradas uma única vez, na GPU, e salvas como WAV de 16 kHz em `datasets/GTZAN_pitch_shifted/semitones_{-1,+1}/` (≈1 GB, cerca de 1 minuto). Execuções seguintes reutilizam o cache. Use `--pitch-shift-semitones` para escolher outros deslocamentos, ou passe a flag sem valores para desativar a augmentation.

## Instalação e treino

No Windows, é preciso instalar o ffmpeg para executar o script.

```powershell
winget install "FFmpeg (Shared)"
```

```powershell
uv sync
uv run python -m ast_finetuning.train --dataset fma_small --data-dir datasets --metadata datasets/fma_tracks.csv --epochs 30 --batch-size 16
uv run python -m ast_finetuning.train --dataset gtzan --data-dir datasets --epochs 30 --batch-size 16
```

O PyTorch e o TorchAudio são obtidos do índice CUDA 13.0 configurado no `pyproject.toml`. O melhor checkpoint fica em `checkpoints/ast-{dataset}-{finetuning}` (por exemplo `checkpoints/ast-gtzan-lora`), a menos que `--output-dir` seja informado, de modo que os modelos de cada dataset e metodologia não se sobrescrevem. A cada época, as losses e acurácias de treino e validação são impressas e registradas em `history.csv` dentro dessa pasta. O treino imprime também a quantidade de parâmetros treináveis; no modo `head`, imprime a lista deles, que deve conter apenas `classifier.weight` e `classifier.bias`.

O treino roda por até 30 épocas (`--epochs`) e usa early stopping: se a `validation_loss` não diminuir em 3 épocas seguidas (`--early-stopping-patience`), o treino é interrompido. Use `--early-stopping-patience 0` para desativar. O checkpoint salvo continua sendo o da época de maior acurácia de validação.

Para classificar um arquivo com um checkpoint treinado:

```powershell
uv run python -m ast_finetuning.predict datasets/GTZAN/jazz/jazz.00000.wav --checkpoint checkpoints/ast-gtzan-head
```

### Métricas de desempenho

Todas as métricas são calculadas no conjunto de validação e salvas em `--output-dir`:

| Arquivo | Conteúdo |
| --- | --- |
| `history.csv` | Por época: losses, acurácias e `validation_precision_macro`, `validation_recall_macro` e `validation_f1_macro`. |
| `per_class_metrics.csv` | Precision, recall, F1 e suporte por classe na época de melhor acurácia de validação. |
| `confusion_matrix.csv` | Matriz de confusão normalizada por linha da melhor época: a célula (`true_label`=i, coluna j) é P(predito = j \| verdadeiro = i). |

As métricas macro-averaged são a média simples das métricas por classe, de modo que todos os gêneros têm o mesmo peso independentemente do número de faixas; uma classe nunca prevista entra com valor 0. Os arquivos da melhor época são sobrescritos sempre que um novo checkpoint é salvo.

### Ajuste fino com LoRA

```powershell
uv run python -m ast_finetuning.train --dataset gtzan --data-dir datasets --finetuning lora
```

Hiperparâmetros disponíveis:

| Flag | Padrão | Descrição |
| --- | --- | --- |
| `--lora-rank` | `1` | Rank `r` das matrizes `A` e `B`. |
| `--lora-alpha` | `2` | Escala `alpha`; a atualização é multiplicada por `alpha / r`. |
| `--lora-dropout` | `0.2` | Dropout aplicado na entrada dos adaptadores. |
| `--lora-target-modules` | `q_proj v_proj` | Camadas lineares do AST adaptadas. Opções: `q_proj`, `k_proj`, `v_proj`, `o_proj` (atenção) e `fc1`, `fc2` (MLP). |

Com os valores padrão e as 8 classes do FMA-Small, 43.016 parâmetros são treinados (≈0,05% do modelo). No modo `lora` o backbone fica em modo de treino para que o dropout dos adaptadores seja aplicado. A metodologia e os hiperparâmetros do LoRA são salvos no `config.json` do checkpoint, de modo que o `predict` reconstrói o modelo correto sem flags adicionais.

## Pré-processamento

Durante o treinamento e a avaliação, os datasets retornam diretamente o espectrograma Mel, nunca a forma de onda. A extração usa o mesmo `torchaudio.compliance.kaldi.fbank` do AST original: 128 bins, hop de 10 ms, janela Hanning de 25 ms, pré-ênfase, remoção de offset DC e FFT de 512 pontos. O log-Mel é normalizado de acordo com o `ASTFeatureExtractor` e ajustado para 1024 frames, o tamanho esperado pelo AST. O treino usa recorte aleatório como augmentation; validação e inferência usam recorte determinístico.
