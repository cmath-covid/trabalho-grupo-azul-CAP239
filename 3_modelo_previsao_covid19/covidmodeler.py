#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Este teste busca seguir ao máximo os passos apresentados durante a aula de explicação do modelo e também do documento
pdf com a descrição das equações mestres do modelo. Neste teste, a geração dos valores é feita de maneira a consumir tudo
o que já foi predito, ou seja, a média, a cada dia de predição tem mais valores preditos anteriormente, ao passo que, depois
de 7 dias de previsão, são usados apenas valores preditos. Além disso, os valores considerados no dia (nkt) também são valores preditos
do dia anterior.
Neste modelo, as últimas alterações estão disponíveis.
"""


import pendulum
import numpy as np
import pandas as pd
from plotnine import *
import matplotlib.pyplot as plt

# Ferramentas para a configuração do plot
import locale
import matplotlib.dates as mdates
locale.setlocale(locale.LC_ALL, '')

# Definindo tipos de Espectros de peso
WeightSpectraCase1 = np.array([0.5, 0.45, 0.05])
WeightSpectraCase2 = np.array([0.7, 0.25, 0.05])

# Vetores de multiplicação base para a geração dos N_min, N_max
NMINVECCASE2 = np.array([1, 3, 5])
NMAXVECCASE2 = np.array([2, 4, 6])


##### Aquisição de dados de dados
def load_owd():
    OWURL = 'owid-covid-data.csv'
    _data = pd.read_csv(OWURL)
    _data["date"] = pd.to_datetime(_data["date"])
    _data.set_index("date", inplace = True)
    return _data


def get_nearest_date(data, actual_date, days = 7,  isMean = True):
    """Função para a busca em vizinhança de dados de países que não possuem os dados completos.
    
    Args:
        data (pd.DataFrame): Conjunto de dados OWD
        actual_date (str): String no formato "YYYY-MM-DD" para a busca
        days (int): Quantidade de dias consideradas para a busca
        is_mean (bool): Indica se o valore buscado deve sair já com a média
    Returns:
        pd.Series ou float
    """

    actual_date = pendulum.from_format(actual_date, "YYYY-MM-DD")
    elements = []
    
    while len(elements) < days:        
        actual_date = actual_date.subtract(days = 1)
        
        # Buscando
        _res = data[data.index == actual_date.to_date_string()]
        
        if not _res.empty:
            elements.append(_res)
    if isMean:
        return pd.concat(elements).new_cases.mean()
    else:
        return pd.concat(elements).new_cases.tolist()


def nkb_avg(data, actual_date = pendulum.now().to_date_string(), days = 7, isIncomplete = False, isMean = True):
    """Função para a geração da média dos dias anteriores ao dia inserido.
    Args:
        data (pd.DataFrame): Dados OWD
        actual_date (str): String com a data em que a busca começa, no formato "YYYY-MM-DD"
        days (int): Quantidade de dias anteriores a serem pesquisados
        is_incomplete (bool): Flag indicando se a busca precisa ser feita na vizinhança
        is_mean (bool): Flag inficando se o valor a ser devolvido é a média ou o conjunto de dados
    Returns:
        pd.Series ou int
    """

    if isIncomplete: # Caso particular
        return get_nearest_date(data, actual_date, days = days, isMean = isMean)
    
    day_before = pendulum.parse(actual_date).subtract(days = days)
    day_before_string = day_before.to_date_string()
    
    res = data[(data.index >= day_before_string) & (data.index < actual_date)]
    assert(res.shape[0] == days)
    if isMean:
        return int(data[(data.index >= day_before_string) & (data.index < actual_date)].new_cases.mean())
    return data[(data.index >= day_before_string) & (data.index < actual_date)].new_cases.tolist()

def nkt(data, actual_date = pendulum.now().to_date_string()):
    return data[data.index == actual_date].new_cases.values[0]
#######


# Equação 1
def generate_nmin_nmax(ns, gfactor, nminvec = NMINVECCASE2, nmaxvec = NMAXVECCASE2):
    """Função para a geração dos valores mínimos e máximos.
    Args:
        ns (np.ndarray): Valores de n
        gfactor (float): Valor de g
        nminvec (np.ndarray): Array com os valores que deve ser utilizados na multiplicação do valor mínimo
        nmaxvec (np.ndarray): Array com os valores que deve ser utilizados na multiplicação do valor mínimo
    Returns:
        tuple: Tupla com os valores (mínimo, máximo)
    """
    return (
        gfactor * ( (ns * nminvec).sum() ),
        gfactor * ( (ns * nmaxvec).sum() )
    )

# Equações 3, 4 e 5
def generate_nvec(nkt, weight_spectra):
    """Gera os vetores de N.
    Args:
        nkt (float): Número de casos atual
        weight_spectra (np.ndarray): Espectro de pesos
    See:
        A operação é considerada como uma convolução, onde os valores são multiplicados com todos os valores
    """
    return nkt * weight_spectra


def generate_gfactor(nkb_average, nkt):
    """Função para a geração do fator g
    Args:
        nkb_average (float): Valor médio dos casos anteriores
        nkt (float): Número de casos atual
    Returns:
        float
    """
    if nkt > nkb_average:
        return nkb_average / nkt
    else:
        return nkt / nkb_average


def wisdom_of_the_crowd(nmin_nmax):
    """Aplica a sabedoria das massas, gerando o valor predito para o dia
    Args:
        nmin_nmax (tuple): Tupla com os valores mínimos e máximos
    Returns:
        int: Valor médio estimado para o dia corrente
    """
    return int((nmin_nmax[0] + nmin_nmax[1]) / 2)


def delta_g(g, g0, qg, qg0):
    """Gera o delta g"""
    if g0 < g:
        return (g0 - g) - qg
    else:
        return (g0 - g) + qg0


def generate_qg(g):
    """Gera o qg"""
    return (1 - g) ** 2


def generate_qg0(g0):
    """Gera o qg0"""
    return (1 - g0) ** 2


def derivative_nk(nkb_average, nkt):
    """Gera d_nk"""
    return (nkb_average - nkt) / nkt


def suppression_factor(delta_g, derivative_nk):
    """Gera o fator de supressão"""
    return ((2*delta_g) + derivative_nk) / 3


def suppression_factor_updated_in_june19_2020(gfactor):
    return 1 - gfactor


def covidmodeler(data, dayzero, days_to_predict, weight_spectra, 
                        days_before = 7, 
                        isIncomplete = False, 
                        usePredict = True, use19JuneUpdate = False):
    """Função para a aplicação do modelo de predição de casos diários, baseado em cascatas multiplicativas e fluídos sociais
        desenvolvido pelo Prof. Dr. Reinaldo Rosa
    Args:
        data (pd.DataFrame): Conunto de dados considerados na predição
        dayzero (str): String no formato "YYYY-MM-DD" indicando o dia de início da predição
        days_to_predict (int): Quantidade de dias futuros a serem preditos
        weight_spectra (np.ndarray): Espectro de peso considerado pelo modelo
        days_before (int): Quantidade de dias anteriores utilizados na média movel
        is_incomplete (bool): Flag indicando se os dados estão completos. Caso não estejam, a busca por datas vizinhas
        é aplicada
        use_predict (bool): Flag indicando se os dados preditos deve ser usados na média móvel
        use19JuneUpdate (bool): Flag indicando se a atualização do modelo passada no dia 19 de Junho (Calculo do S) é para ser utilizada
    Returns:
        tuple: Tupla contendo ( Valores preditos, Parâmetros gerais , Parâmetros de supressão )
    """
    
    predictedvalues = {
		'date': [],
		'new_cases': []
	}
    generated_parameters = {
		'nmin': [],
		'nmax': [],
		'wisdom_of_the_crowd': [],
		'reference_date': [],
		'g': []
	}

    generated_supression_parameters = {
		'qg': [],
		'qg0': [],
		'dg': [],
		'dnk': [],
		's': [],
		'reference_date': []
    }
        
    def add_to_predictedvalues(date, newcase):
        predictedvalues['date'].append(date)
        predictedvalues['new_cases'].append(newcase)

    def add_to_generated_parameters(nmin_nmax, wisdom_of_the_crowd_value, reference_date, g):
        generated_parameters['nmin'].append(nmin_nmax[0])
        generated_parameters['nmax'].append(nmin_nmax[1])
        generated_parameters['wisdom_of_the_crowd'].append(wisdom_of_the_crowd_value)
        generated_parameters['reference_date'].append(reference_date)
        generated_parameters['g'].append(g)

    def add_to_supression(qg, qg0, dg, dnk, s, reference_date):
        generated_supression_parameters['qg'].append(qg)
        generated_supression_parameters['qg0'].append(qg0)
        generated_supression_parameters['dg'].append(dg)
        generated_supression_parameters['dnk'].append(dnk)
        generated_supression_parameters['s'].append(s)
        generated_supression_parameters['reference_date'].append(reference_date)

    def search_predicted_values(data, actual_date, days = 7):
        """Função auxiliar para acelerar a busca por dadas nos dias preditos"""
        _data = pd.DataFrame(data)
        _data.set_index('date', inplace=True)
        day_before = actual_date.subtract(days = days)
        res = _data[(_data.index >= day_before.to_date_string()) & (_data.index < actual_date.to_date_string())]
        assert(res.shape[0] == days)
        return res

    actual_date = pendulum.from_format(dayzero, 'YYYY-MM-DD')
	
    # Passos apenas para o primeiro dia
    nkb7 = nkb_avg(data, dayzero, days = days_before, isIncomplete = isIncomplete)
    nktvar = nkt(data, dayzero)
    ## Gerando os Ns	
    ns = generate_nvec(nktvar, weight_spectra)
    ## Gerando o fator de escala G
    gfactor = generate_gfactor(nkb7, nktvar)
    ## Gerando o minmax do primeiro dia		
    nmin_nmax = generate_nmin_nmax(ns, gfactor)
    	
    ## Aplicando a sabedoria das massas para gerar a predição
    nktvar = wisdom_of_the_crowd(nmin_nmax)
    
    ## Salvando as informações do primeiro dia
    actual_date = actual_date.add(days = 1)
    add_to_predictedvalues(actual_date, nktvar)
    add_to_generated_parameters(nmin_nmax, nktvar, actual_date, gfactor)
    
    # Começando a predição de todos os demais dias
    dd = days_before # temp

    for _ in range(0, days_to_predict - 1):
        if days_before < 0:
            # Busca nos valores preditos         
            nkb7 = search_predicted_values(predictedvalues, actual_date).new_cases.mean()
        elif usePredict:
            nkb7 = nkb_avg(data, actual_date.to_date_string(), days = dd, isIncomplete = isIncomplete, isMean = False)
            nkb7 = nkb7[: days_before]
            tmp = predictedvalues['new_cases']
        
            # Tratando os dados para que seja feito = Dados reais + dados preditos
            # até que isto alcance a quantidade de dias suficiente para a substituição
            # total por valores estimados
            idx = 0
            while len(nkb7) < dd:
                nkb7.append(tmp[idx])
                idx += 1
            assert(len(nkb7) == dd)
            nkb7 = np.mean(nkb7)
    
        ns = generate_nvec(nktvar, weight_spectra)

        g0 = gfactor # o g0 é sempre o do dia anterior
        gfactor = generate_gfactor(nkb7, nktvar)
        nmin_nmax = generate_nmin_nmax(ns, gfactor)
        nktvar = wisdom_of_the_crowd(nmin_nmax)
        
        ## Gerando informações que só são possíveis do segundo dia em diante
        qg = generate_qg(gfactor)
        qg0 = generate_qg0(g0)
        dg = delta_g(gfactor, g0, qg, qg0)
        dnk = derivative_nk(nkb7, nktvar)
        
        # Definindo a maneira com que o "s" é calculado
        ## Isto foi feito como requisito final do trabalho de Matemática Computacional
        ## Nesse, o fator de "s" é calculado com 1 - g
        if use19JuneUpdate:
            s = suppression_factor_updated_in_june19_2020(gfactor)
        else:
            s = suppression_factor(dg, dnk)

        ## Salvando os resultados
        add_to_predictedvalues(actual_date.add(days = 1), nktvar)
        add_to_generated_parameters(nmin_nmax, nktvar, actual_date.add(days = 1), gfactor)
        add_to_supression(qg, qg0, dg, dnk, s, actual_date.add(days = 1))
        
        if usePredict:
            days_before -= 1
        actual_date = actual_date.add(days = 1)
    return (
        pd.DataFrame(predictedvalues), pd.DataFrame(generated_parameters), pd.DataFrame(generated_supression_parameters)
    )


def generate_fiocruz_datamean(data: pd.DataFrame):
    """Função para gerar a estimativa de casos seguindo o estudo da Fiocruz (12 vezes a quantidade diária). Além disso
    a média entre os valores de caso diário observado e o estimado pela Fiocruz são calculados.
    Args:
        data (pd.DataFrame): Conjunto de dados com colunas "new_cases"
    Returns:
        pd.DataFrame: Tabela com as colunas "fiocruz_estimate" e "fiocruz_mean"
    """

    _data = data.copy()
    _data = _data.assign(fiocruz_estimate = lambda x: x['new_cases'] * 12)
    return _data.assign(fiocruz_mean = lambda x: (x['new_cases'] + x['fiocruz_estimate']) / 2)
     

def organize_data(predictedvalues, generated_parameters, data_owd):
    predictedvalues.set_index('date', inplace = True)
    predictedvalues.index = predictedvalues.index.tz_convert(tz = None)
    merged = pd.merge(data_owd, predictedvalues, left_index=True, right_index=True)

    generated_parameters.set_index('reference_date', inplace = True)
    generated_parameters.index = generated_parameters.index.tz_convert(tz = None)
    merged = generated_parameters.merge(merged, left_index=True, right_index=True)

    return merged


## Função auxiliar para a visualização das curvas G e S de maneira
### padronizada
def plot_g_and_s(generated_parameters, generated_supression_parameters, 
                    create_fig = True, use_s_mean = False, days_to_s_mean = 7):
    """Função auxiliar para gerar as curvas de G e S de uma execução do modelo

    Args:
        generated_parameters (pd.DataFrame): Tabela com todos os parâmetros gerados pelo modelo
        generated_supression_parameters (pd.DataFrame): Tabela com os parâmetros gerados de supressão
    Returns:
        None
    """

    generated_supression_parameters = generated_supression_parameters.copy()

    # Gerando os plots de G e S
    if create_fig:
        plt.figure(dpi = 300, figsize = (8, 3))
    
    ax = plt.subplot(1, 2, 1)
    plt.title("Parâmetro g")
    plt.plot(generated_parameters.iloc[1:, :].index, generated_parameters.iloc[1:, :].g, 'k--o')
    plt.xticks(rotation = 25)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    
    ax = plt.subplot(1, 2, 2)
    # Organizando os dados de parâmetros de supressão gerados
    generated_supression_parameters.set_index('reference_date', inplace = True)
    generated_supression_parameters.index = generated_supression_parameters.index.tz_convert(tz = None)

    if use_s_mean:
        generated_supression_parameters = (
            generated_supression_parameters
                .rolling(days_to_s_mean)
                .mean()
                .dropna()
        )

    plt.title("Parâmetro s")
    plt.plot(generated_supression_parameters.index, generated_supression_parameters.s, 'k--o')
    plt.xticks(rotation = 25)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.tight_layout()
