Detecção de Sarna em Cachorros usando Inteligência Artificial

Este é um projeto de detecção de sarna em cachorros utilizando inteligência artificial. O projeto foi desenvolvido com o auxílio do Flask, um framework web em Python.
Pré-requisitos

    Python 3.x instalado no seu computador.
    Pacote pip instalado no Python.
    Conhecimento básico em Python e terminal/prompt de comando.

Instalação

Siga as etapas abaixo para configurar o projeto em seu computador:

    Clone ou baixe este repositório para o seu computador.

    No terminal ou prompt de comando, navegue até o diretório do projeto.

    Crie um ambiente virtual para isolar as dependências do projeto:
    mkdir myproject
    cd myproject
    py -3 -m venv .venv
    
    Após criado o projeto venv, mover os arquivos contidos neste repositorio para a pasta .venv
    
    Ative o ambiente virtual:

    No Windows:

    .venv\Scripts\activate

    No Linux/Mac:

    source .venv/bin/activate

Instale as dependências do projeto:

    pip install -r requirements.txt

Configuração

    Abra o arquivo function.py localizado na pasta src.

    Na linha 27 do arquivo, altere o caminho para o arquivo dogHealthClassifier.h5 para o caminho correto em seu sistema.
