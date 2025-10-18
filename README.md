# Detecção de Viés Social via Programação Semidefinida e Análise de Grafos

Este repositório contém o código e os experimentos do artigo **"Detecção de Viés Social em Redes Sociais via Programação Semidefinida e Análise Estrutural de Grafos: Implementação e Validação Experimental"**.

## 📝 Descrição

O projeto apresenta uma metodologia para detectar câmaras de eco e comportamento coordenado em redes sociais. A abordagem combina otimização convexa (Programação Semidefinida - SDP) com análise estrutural de grafos para identificar comunidades que são, ao mesmo tempo, densamente conectadas e ideologicamente homogêneas.

As principais contribuições incluem:
- Uma formulação SDP que incorpora scores de viés na maximização de modularidade.
- Uma heurística eficiente baseada no algoritmo de Louvain que converge para a solução do SDP com um speedup de até 118x.
- Validação experimental em datasets clássicos (Karate Club) e sintéticos.

## 🚀 Como Executar os Experimentos

### 1. Pré-requisitos

- Python 3.8+
- Jupyter Notebook ou JupyterLab

### 2. Instalação

Clone o repositório e instale as dependências:

```bash
git clone [https://github.com/seu-usuario/deteccao-vies-social-sdp.git](https://github.com/seu-usuario/deteccao-vies-social-sdp.git)
cd deteccao-vies-social-sdp
pip install -r requirements.txt
```

### 3. Executando os Notebooks

Todos os experimentos, análises e visualizações estão disponíveis no notebook Jupyter. Para executá-lo, abra o ambiente Jupyter:

```bash
jupyter notebook
```
Em seguida, navegue até a pasta `notebooks/` e abra o arquivo `Experimentos.ipynb`. O notebook é autossuficiente e foi projetado para ser executado no Google Colab.

## 📂 Estrutura do Repositório

- **/notebooks**: Contém o notebook Jupyter com a execução de todos os experimentos.
- **/src**: Contém o código fonte modularizado:
    - `sdp_model.py`: Implementação do modelo SDP.
    - `heuristic.py`: Implementação da heurística baseada em Louvain.
    - `evaluation.py`: Funções para as métricas de avaliação.
    - `data_utils.py`: Scripts para geração de dados.
- **/data**: Armazena os datasets utilizados.
- **/figures**: Armazena as figuras e gráficos gerados.

## 📄 Licença

Este projeto está licenciado sob a Licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## ✍️ Autores

- Sergio A. Monteiro
- Ronaldo M. Gregorio
- Nelson Maculan
- Vitor S. Ponciano
- Axl S. Andrade