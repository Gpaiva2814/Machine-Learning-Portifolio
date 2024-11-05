import requests


class Consulta:
    def __init__(self, codCli, filEntrega, filFaturamento, cndPgt, desCndPgt, codprd):
        self.codCli = codCli
        self.filEntrega = filEntrega
        self.filFaturamento = filFaturamento
        self.cndPgt = cndPgt
        self.desCndPgt = desCndPgt
        self.codprd = codprd


def prepare_product_lists(consulta):
    product_list_1p = []
    product_list_3p = []
    for product in consulta.codprd:
        item = {
            'CodigoMercadoria': product,
            'Quantidade': 1,
        }
        if product.startswith('intenal'):
            item['codGroupMerFrac'] = 0
            product_list_1p.append(item)
        else:
            item['seller'] = product.split('_')[0]
            product_list_3p.append(item)
    return product_list_1p, product_list_3p


def prepare_payload(consulta, product_list_1p, product_list_3p):
    params = {'idCliente': consulta.codCli, 'email': ''}
    try:
        response = requests.get(
            'http://AAAAAAAAAA/cliente/obterPreferencia', params=params
        )
        response.raise_for_status()
        data = response.json()
        fil_delivery = data['listCustomerRestriction'][0].get('fil_delivery', '').upper()
        uf_filial_faturamento = data['listCustomerRestriction'][0].get('ufFilialFaturamento', '').upper()
        filFat = data['listCustomerRestriction'][0].get('codeWarehouseBilling', '')
        filExp = data['listCustomerRestriction'][0].get('codeWarehouseDelivery', '')
        cndPgt = data['lstCondicoesPamento'][0].get('CODIGO_CONDICAO', '')
        desCndPgt = data['lstCondicoesPamento'][0].get('observacao', '')
    except requests.RequestException as e:
        # Log the error or handle it as needed
        raise RuntimeError(f'Failed to fetch client preferences: {e}')

    consulta.filEntrega = fil_delivery
    consulta.filFaturamento = uf_filial_faturamento
    consulta.cndPgt = cndPgt
    consulta.desCndPgt = desCndPgt

    return {
        'origemChamada': 'CRT',
        'uid': consulta.codCli,  # Agora pega os valores da instância
        'cupons_novos': [],
        'segment': 0,
        'tipoLimiteCred': '',
        'precoEspecial': '',
        'territorioRca': 0,
        'ie': '',
        'classEstadual': 0,
        'tipoSituacaoJuridica': '',
        'codSegNegCliTer': 0,
        'email': '@',
        'tipoConsulta': 1,
        'numberSpinPrice': '0',
        'codeDeliveryRegion': '0',
        'commercialActivity': 0,
        'group': 0,
        'codCidade': 0,
        'codCidadeEntrega': 0,
        'codRegiaoPreco': 0,
        'temVendor': '',
        'codigoCanal': 0,
        'ufTarget': uf_filial_faturamento,
        'ufFilialFaturamento': '',
        'bu': 0,
        'manual': 'N',
        'produtos': product_list_1p,
        'asm': 0,
        'produtosSeller': product_list_3p,
        'condicaoPagamento': consulta.cndPgt,
        'ProdutosExclusaoEan': [],
        'CODBRRUNDVNDCSM': None,
        'codeWarehouseDelivery': filExp,
        'codeWarehouseBilling': filFat,
        'codeDeliveryMode': 0,
        'codopdtrcetn': 0,
        'codVrsSis': 0,
        'listaProdutosEdl': [],
    }


def fetch_price_data(consulta, product_list_1p, product_list_3p):
    url_preco = 'https://AAAAAAAAAA/Mercadoria/obterPreco'
    payload = prepare_payload(consulta, product_list_1p, product_list_3p)
    response = requests.post(url_preco, json=payload)
    return response.json()


def process_price_data(consulta, response_data):
    prds_1p = response_data.get('resultado', None)
    prds_3p = response_data.get('lstPrecoSeller', None)
    dict_produto = {}
    # Processa produtos 3P (terceiros)
    if prds_3p:
        for produto in consulta.codprd:
            for product in prds_3p:
                if produto == product['codigoMercadoria']:
                    preco = float(product['preco'])
                    # Se o preço for 0, define como 'preço não encontrado'
                    if preco == 0:
                        dict_produto[produto] = {'preco': 'Preço não encontrado', 'preco_cx': None}
                    else:
                        dict_produto[produto] = {'preco': preco, 'preco_cx': None}
                    break

    # Processa produtos 1P (próprios)
    if prds_1p:
        for produto in consulta.codprd:
            for product in prds_1p:
                if produto == product['codigoMercadoria']:
                    preco = float(product['precos'][0]['precoNormal'])
                    preco_cx = float(
                        product['precos'][0].get('precoCaixa', 0)
                    )  # Garante que "precoCaixa" esteja disponível
                    # Se o preço for 0, define como 'preço não encontrado'
                    if preco == 0:
                        dict_produto[produto] = {
                            'preco': 'Preço não encontrado',
                            'preco_cx': preco_cx if preco_cx != 0 else None,
                        }
                    else:
                        dict_produto[produto] = {'preco': preco, 'preco_cx': preco_cx if preco_cx != 0 else None}
                    break

    return dict_produto


def get_precos_api(consulta):
    product_list_1p, product_list_3p = prepare_product_lists(consulta)
    response_data = fetch_price_data(consulta, product_list_1p, product_list_3p)
    return process_price_data(consulta, response_data)
