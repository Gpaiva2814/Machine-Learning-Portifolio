import base64
import json
from io import BytesIO
from pathlib import Path

import cv2 as cv
import numpy as np
import polars as pl
import requests
from PIL import Image, ImageDraw, ImageFont

import consulta_preco 

# Path definitions
FONTES_PATH = Path(__file__).resolve().parent / 'fontes'
INPUT_PATH = Path(__file__).resolve().parent / 'input'
OUTPUT_PATH = Path(__file__).resolve().parent / 'output'
TABELAS_PATH = Path(__file__).resolve().parent / 'tabelas'
 

def le_produtos(df: pl.DataFrame, consulta, precos: dict):
    produtos_info = []

    # Itera sobre os códigos de produto fornecidos em consulta.codprd
    for cod in consulta.codprd:
        # Filtra o DataFrame usando o método filter de Polars para encontrar o produto
        produto_df = df.filter(pl.col("IDPRODUTO") == cod)

        # Verifica se o DataFrame filtrado tem algum dado
        if produto_df.height > 0:
            # Extrai os valores do produto usando o método 'item' de Polars para acessar valores únicos
            url = produto_df.get_column("URL_IMAGEM").first()[:-5] + '100x'
            descricao = produto_df.get_column("DESCPRODUTO").first()

            # Obtém os preços do dicionário 'precos'
            preco = precos.get(cod, {}).get("preco", "preço não encontrado")
            preco_cx = precos.get(cod, {}).get("preco_cx", "preço não encontrado")

            # Monta o dicionário do produto
            produto_dict = {
                "Id": cod,
                "descricao": descricao,
                "preco": preco,
                "preco_cx": preco_cx,
                "url_imagem": url
            }
            produtos_info.append(produto_dict)

    return produtos_info

def adicionar_imagens_no_fundo(fundo_path, produtos_info, posicoes):
    fundo = cv.imread(str(fundo_path))
    imagens_adicionadas = 0  # Contador para acompanhar quantas imagens foram adicionadas

    for idx, produto in enumerate(produtos_info):
        if imagens_adicionadas >= 8:  # Se já foram adicionadas 8 imagens, sai do loop
            break

        url = produto['url_imagem']
        try:
            response = requests.get(url, timeout=10)  # Adicionando timeout

            if response.status_code == 200:
                img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                imagem = cv.imdecode(img_array, cv.IMREAD_COLOR)
                
                imagem_key = f'imagem_{imagens_adicionadas + 1}'  # Gera a chave da imagem

                if imagem_key in posicoes:  # Verifica se a chave está nas posições
                    coord_img = posicoes[imagem_key]
                    left, top = coord_img
                    width, height = 87, 99

                    imagem_redimensionada = cv.resize(imagem, (width, height))
                    fundo[left: left + height, top: top + width] = imagem_redimensionada
                    imagens_adicionadas += 1  # Incrementa o contador de imagens adicionadas
            else:
                print(f'Erro ao baixar a imagem. Status code: {response.status_code}')

        except requests.exceptions.Timeout:
            print(f'Timeout ao tentar baixar a imagem da URL: {url}')
        except requests.exceptions.RequestException as e:
            print(f'Erro ao fazer requisição para a URL: {url} - {e}')

    return fundo

def quebra_descricao(descricao, draw, fonte, largura_maxima=127):
    # Function remains unchanged as it doesn't use paths
    palavras = descricao.upper().split(' ')
    linhas = ['', '', '', '']
    linha_atual = 0

    def pode_adicionar(palavra, linha):
        texto_teste = (linha + ' ' + palavra).strip() if linha else palavra
        bbox = draw.textbbox((0, 0), texto_teste, font=fonte)
        return (bbox[2] - bbox[0]) <= largura_maxima

    for palavra in palavras:
        if pode_adicionar(palavra, linhas[linha_atual]):
            linhas[linha_atual] += ' ' + palavra if linhas[linha_atual] else palavra
        else:
            linha_atual += 1
            if linha_atual >= len(linhas):
                break
            linhas[linha_atual] = palavra

    return (linhas[0].strip(), linhas[1].strip(), linhas[2].strip(), linhas[3].strip())


def adicionar_textos_no_fundo(consulta, fundo, produtos_info, posicoes):
    preto = (0, 0, 0)
    branco = (255, 255, 255)

    # Use FONTES_PATH for font files
    fonte_obs_negrito = ImageFont.truetype(str(FONTES_PATH / 'calibrib.ttf'), size=11)
    fonte_obs = ImageFont.truetype(str(FONTES_PATH / 'calibrib.ttf'), size=11)
    fonte_produtos = ImageFont.truetype(str(FONTES_PATH / 'Rubik-Light.ttf'), size=11)
    fonte_preco = ImageFont.truetype(str(FONTES_PATH / 'segoe-ui-black.ttf'), size=14)

    draw = ImageDraw.Draw(fundo)

    # Rest of the function remains unchanged as it doesn't use paths
    draw.text((28, 824), 'Condição de Pagamento:', fill=branco, font=fonte_obs_negrito)
    draw.text((142, 824), f' {consulta.cndPgt}{consulta.desCndPgt}', fill=branco, font=fonte_obs)
    draw.text((28, 836), 'Vendido por', fill=branco, font=fonte_obs)
    draw.text((86, 836), f' {consulta.filEntrega}', fill=branco, font=fonte_obs_negrito)

    largura_texto = draw.textbbox((0, 0), f' {consulta.filEntrega}', font=fonte_obs_negrito)[2]
    inicio_proximo_texto = 86 + largura_texto

    draw.text((inicio_proximo_texto, 836), ' e faturado por', fill=branco, font=fonte_obs)
    draw.text(((inicio_proximo_texto + 67), 836), f' {consulta.filFaturamento}', fill=branco, font=fonte_obs)

    for idx, produto in enumerate(produtos_info):
        descricao = produto['Id'].split('_')[1] + ' - ' + produto['descricao']
        preco = produto['preco']
        preco_cx = produto['preco_cx'] if produto['preco_cx'] is not None else produto['preco']

        l_1, l_2, l_3, l_4 = quebra_descricao(descricao, draw, fonte_produtos)
        nome_key = f'nome_{idx + 1}'
        preco_key = f'preco_{idx + 1}'

        if nome_key in posicoes and preco_key in posicoes:
            coord_nome = posicoes[nome_key]
            coord_preco = posicoes[preco_key]

            draw.text(coord_nome, l_1, font=fonte_produtos, fill=preto)
            draw.text((coord_nome[0], coord_nome[1] + 15), l_2, font=fonte_produtos, fill=preto)
            draw.text((coord_nome[0], coord_nome[1] + 2 * 15), l_3, font=fonte_produtos, fill=preto)
            draw.text((coord_nome[0], coord_nome[1] + 3 * 15), l_4, font=fonte_produtos, fill=preto)

            if isinstance(produto['preco'], (int, float)):
                texto_preco = f'R${preco} und | R${preco_cx} cx'
            else:
                texto_preco = f'{preco}'

            tamanho_texto = draw.textbbox((0, 0), texto_preco, font=fonte_preco)
            tamanho_texto_largura, tamanho_texto_altura = tamanho_texto[2], tamanho_texto[3]
            x_centralizado = (540 - tamanho_texto_largura) // 2

            if idx % 2 == 0:
                draw.text(((x_centralizado - 149) + 37, coord_preco[1]), texto_preco, font=fonte_preco, fill=preto)
            else:
                draw.text(((x_centralizado + 149) - 37, coord_preco[1]), texto_preco, font=fonte_preco, fill=preto)

    return fundo


def construcao_panfleto(CODCLI): 
    # Define background paths using INPUT_PATH
    fundo_panfleto='1'
    fundos = {'1': INPUT_PATH / 'backgrounds' / 'background_mais_vendidos.png'}
    
    if fundo_panfleto == '1':
        fundo_path = fundos.get(str(fundo_panfleto), Path('Fundo não encontrado'))

        # Use INPUT_PATH for positions file
        posicoes_path = INPUT_PATH / 'posicoes_mais_vendidos.json'
        if not posicoes_path.exists():
            raise FileNotFoundError(f'Posições file not found at {posicoes_path}')

        with open(str(posicoes_path), 'r') as json_file:
            posicoes = json.load(json_file)

        posicoes = {key: tuple(value) for key, value in posicoes.items()}

        # Use TABELAS_PATH for parquet files
        df_mais_vendidos_path = TABELAS_PATH / 'MAIS_VENDIDOS.parquet'
        df_mais_vendidos = pl.read_parquet(str(df_mais_vendidos_path))
        codigo_produtos = df_mais_vendidos.filter(pl.col("CODCLI") == CODCLI).get_column("CODMERSRR").to_list()
    
    consulta = consulta_preco.Consulta(
        codCli=CODCLI,
        filEntrega=None,
        filFaturamento=None,
        cndPgt=None,
        desCndPgt=None,
        codprd=codigo_produtos,
    )
    
    # Use TABELAS_PATH for products parquet file
    df_img_prd_path = TABELAS_PATH / 'SUP_PRODUTOS.parquet'
    df_img_prd = pl.read_parquet(str(df_img_prd_path))
    precos = consulta_preco.get_precos_api(consulta)
    produtos_info = le_produtos(df_img_prd, consulta, precos)
    produtos_info_validos = [produto for produto in produtos_info if produto['preco'] != 'Preço não encontrado']
    
    resultado = adicionar_imagens_no_fundo(fundo_path, produtos_info_validos, posicoes)
    
    imagem_rgb = cv.cvtColor(resultado, cv.COLOR_BGR2RGB)
    fundo = Image.fromarray(imagem_rgb)

    fundo = adicionar_textos_no_fundo(consulta,fundo,produtos_info_validos,posicoes)
    fundo.save(str(OUTPUT_PATH / 'panfleto.png'))
    buffer = BytesIO()
    fundo.save(buffer, format="PNG")

    # end_time = time.time()
    # print("tempo execucao:", end_time - start_time, "segundos")
    return(base64.b64encode(buffer.getvalue()).decode('utf-8'))

if __name__ == '__main__':
    CODCLI = 1
    construcao_panfleto(CODCLI)
