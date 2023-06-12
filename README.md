# Detecção de Sarna em Cachorros usando Inteligência Artificial

Este é um projeto de detecção de sarna em cachorros utilizando inteligência artificial. O projeto consiste em um frontend que interage com uma API em JavaScript, desenvolvida com o framework Express.js, para enviar as requisições para a API em Python responsável por realizar a detecção.

## Pré-requisitos

- Python 3.x instalado no seu computador.
- Pacote pip instalado no Python.
- Conhecimento básico em Python e terminal/prompt de comando.

## Instalação

Siga as etapas abaixo para configurar o projeto em seu computador:

1. Clone ou baixe este repositório para o seu computador.

2. No terminal ou prompt de comando, navegue até o diretório do projeto.

3. Crie um ambiente virtual para isolar as dependências do projeto:

   No Windows:
   ```
   mkdir myproject
   cd myproject
   py -3 -m venv .venv
   ```

   No Linux/Mac:
   ```
   mkdir myproject
   cd myproject
   python3 -m venv .venv
   ```

4. Após criar o ambiente virtual, mova os arquivos contidos neste repositório para a pasta `.venv`.

5. Ative o ambiente virtual:

   No Windows:
   ```
   .venv\Scripts\activate
   ```

   No Linux/Mac:
   ```
   source .venv/bin/activate
   ```

6. Instale as dependências do projeto:

   ```
   pip install -r requirements.txt
   ```

## Configuração

1. Abra o arquivo `function.py` localizado na pasta `.venv`.

2. Na linha 27 do arquivo, altere o caminho para o arquivo `dogHealthClassifier.h5` para o caminho correto em seu sistema.

## Execução

Importante: não execute o arquivo main.py, pois isso fará com que o sistema não funcione corretamente.

Após a conclusão da instalação e configuração, execute o seguinte comando no terminal ou prompt de comando para iniciar o servidor Flask:

```
python function.py
```

O servidor Flask será iniciado e estará pronto para receber solicitações.

A seguir, execute o front-end para utilizar esta API.

