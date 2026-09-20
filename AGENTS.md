Diretrizes para Agentes de IA e Assistentes de Código.

Este documento serve como a referência para as restrições de arquitetura, stack tecnológico e padrões de código deste projeto de Music Information Retrieval (MIR). Todos os agentes de IA devem seguir estas regras ao gerar, refatorar ou analisar o código.

1. Gerenciamento de Pacotes e Ambiente

Gerenciador Oficial: Utilize exclusivamente o uv. Não utilize pip padrão, poetry, pipenv ou conda para instalar dependências ou gerenciar o ambiente virtual.

Instalação de Dependências: Ao sugerir comandos de instalação, use a sintaxe do uv, como uv pip install <pacote> ou uv add <pacote>.

PyTorch com CUDA 13.0: O projeto requer PyTorch configurado especificamente para CUDA 13.0. Veja exemplo do pytorch.toml abaixo.
```
[project]
name = "estudos-pytorch"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.14"
dependencies = [
    "ipykernel>=7.1.0",
    "matplotlib>=3.10.8",
    "torch>=2.10.0",
    "torchvision>=0.25.0",
]

[tool.uv.sources]
torch = [
    { index = "pytorch-cu130" },
]
torchvision = [
    { index = "pytorch-cu130" },
]

[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true
```



2. Gerenciamento de dependências.

Manipulação de Dados:

Dar preferência a utilizar a biblioteca polars e pyarrow no lugar da biblioteca pandas.

Manipulação de Arquivos e Diretórios:

✅ USAR: pathlib.Path (para navegação de diretórios, junção de caminhos e verificação de arquivos).

❌ NÃO USAR: os.path.

Processamento de Áudio:

USAR: torchaudio e torchaudio.functional como prioridade. Caso seja necessário utilize a biblioteca librosa.

3. Padrões de Implementação.

Dataset: Estamos trabalhando primariamente com o subset fma_small do FreeMusicArchive.

Caminho: Todos os datasets estão localizados na pasta /datasets/ dentro da raiz do projeto.

Representação do sinal de áudio: O sinal de áudio deve ser transformado num Mel Espectrograma.

Modelo Pré-treinado a ser utilizado: https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593. Audio Spectrogram Transformer (AST).

4. Idioma

Os comentários e a documentação do código (docstrings) podem ser escritos em Inglês. O feedback para o usuário deve ser escrito em português.