#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 19:02:48 2020

Neste teste a geração dos valores é feita de maneira a consumir tudo o que é predito. Validado
"""

import pendulum
import numpy as np
import pandas as pd
from plotnine import *
import matplotlib.pyplot as plt

OWURL = 'owid-covid-data.csv'

##### Aquisição de dados de dados
def load_owd():
    _data = pd.read_csv(OWURL)
    _data["date"] = pd.to_datetime(_data["date"])
    _data.set_index("date", inplace = True)
    return _data

def get_nearest_date(data, actual_date, days = 7,  isMean = True):
    """Função para busca em dados. Utilizada em casos como a Angola que
    não possui os dados completos
    
    Funcionamento: Partindo de uma data, encontra 7 data anteriores a esta
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

# Definindo tipos de Espectros de peso
WeightSpectraCase1 = np.array([0.5, 0.45, 0.05])
WeightSpectraCase2 = np.array([0.7, 0.25, 0.05])

# Vetores de multiplicação base para a geração dos N_min, N_max
NMINVECCASE1 = np.array([2, 4, 5])
NMAXVECCASE1 = np.array([4, 7, 10])
NMINVECCASE2 = np.array([1, 3, 5])
NMAXVECCASE2 = np.array([2, 4, 6])

# Equação 1
def generate_nmin_nmax(ns, gfactor, nminvec = NMINVECCASE2, nmaxvec = NMAXVECCASE2):
    return (
        gfactor * ( (ns * nminvec).sum() ),
        gfactor * ( (ns * nmaxvec).sum() )
    )

# Equações 3, 4 e 5
def generate_nvec(nkt, weight_spectra):
    return nkt * weight_spectra

def generate_gfactor(nkb_average, nkt):
    if nkt > nkb_average:
        return nkb_average / nkt
    else:
        return nkt / nkb_average

def wisdom_of_the_crowd(nmin_nmax):
    return int((nmin_nmax[0] + nmin_nmax[1]) / 2)

def delta_g(g, g0, qg, qg0):
    if g0 < g:
        return (g0 - g) - qg
    else:
        return (g0 - g) + qg0

def generate_qg(g):
    return (1 - g) ** 2

def generate_qg0(g0):
    return (1 - g0) ** 2

def derivative_nk(nkb_average, nkt):
    return (nkb_average - nkt) / nkt

def suppression_factor(delta_g, derivative_nk):
    return ((2*delta_g) + derivative_nk) / 3

def covidmodeler(data, dayzero, days_to_predict, weight_spectra, days_before = 7, isIncomplete = False, usePredict = True):
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
            # Busca nos valores preditos..            
            nkb7 = search_predicted_values(predictedvalues, actual_date).new_cases.mean()
        elif usePredict:
            nkb7 = nkb_avg(data, actual_date.to_date_string(), days = dd, isIncomplete = isIncomplete, isMean = False)
            nkb7 = nkb7[:days_before]
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
