Diretrizes para Agentes de IA e Assistentes de Código.

Este documento serve como a referência para as restrições de arquitetura, stack tecnológico e padrões de código deste projeto de Music Information Retrieval (MIR). Todos os agentes de IA devem seguir estas regras ao gerar, refatorar ou analisar o código.

1. Gerenciamento de Pacotes e Ambiente

Gerenciador Oficial: Utilize exclusivamente o uv. Não utilize pip padrão, poetry, pipenv ou conda para instalar dependências ou gerenciar o ambiente virtual.

Instalação de Dependências: Ao sugerir comandos de instalação, use a sintaxe do uv, como uv pip install <pacote> ou uv add <pacote>.

PyTorch com CUDA 13.0: O projeto requer PyTorch configurado especificamente para CUDA 13.0. Veja exemplo do pytorch.toml dentro do repo.

GPU Utilizada: RTX 5070 Ti com 16 GB de VRAM.

2. Gerenciamento de dependências.

Manipulação de Dados:

Dar preferência a utilizar a biblioteca polars e pyarrow no lugar da biblioteca pandas.

Manipulação de Arquivos e Diretórios:

USAR: pathlib.Path (para navegação de diretórios, junção de caminhos e verificação de arquivos).

NÃO USAR: os.path.

Processamento de Áudio:

USAR: torchaudio e torchaudio.functional como prioridade. Caso seja necessário utilize a biblioteca librosa.

3. Padrões de Implementação.

Dataset: Estamos trabalhando com os subsets FMA-Small, FMA-Medium e FMA-Large do FreeMusicArchive. Para salvar espaço no computador, estou somente com o fma_small dentro da pasta de datasets, pois o objetivo inicial é fazer uma prova de conceito.

Train Validation Split: Coluna `split` dentro do csv de metadados do FMA contém separação entre `training`, `validation` e `test`.

Caminho: Todos os datasets estarão localizados na pasta /datasets/ dentro da raiz do projeto.

Representação do sinal de áudio: O sinal de áudio deve ser transformado num Mel Espectrograma com 128 bins de frequência, hop size de 10 ms e janelamento por Hamming com comprimento de 25 ms.
por
Modelo Pré-treinado a ser utilizado: https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593. Audio Spectrogram Transformer (AST).

4. Idioma

Os comentários e a documentação do código (docstrings) podem ser escritos em Inglês. O feedback para o usuário deve ser escrito em português.

5. Para executar o projeto.

uv run python -m ast_finetuning.train `
  --data-dir datasets `
  --metadata datasets/fma_tracks.csv `
  --subset small