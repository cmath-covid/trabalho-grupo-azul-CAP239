import scipy.stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def dfa1d(time_series, degree):
    # 1. A série temporal {Xk} com k = 1, ..., N é integrada na chamada função perfil Y(k)
    x = np.mean(time_series)
    time_series = time_series - x
    yk = np.cumsum(time_series)
    tam = len(time_series)

    # 2. A série (ou perfil) Y(k) é dividida em N intervalos não sobrepostos de tamanho S
    sf = np.ceil(tam / 4).astype(np.int)
    boxratio = np.power(2.0, 1.0 / 8.0)
    vetoutput = np.zeros(shape = (1,2))

    s = 4

    while s <= sf:
        serie = yk

        if np.mod(tam, s) != 0:
            l = s * int(np.trunc(tam/s))
            serie = yk[0:l]

        t = np.arange(s, len(serie), s)
        v = np.array(np.array_split(serie, t))
        l = len(v)
        x = np.arange(1, s + 1)

        # 3. Calcula-se a variância para cada segmento v = 1,…, n_s:
        p = np.polynomial.polynomial.polyfit(x, v.T, degree)
        yfit = np.polynomial.polynomial.polyval(x, p)
        vetvar = np.var(v - yfit)

        # 4. Calcula-se a função de flutuação DFA como a média das variâncias de cada intervalo
        fs = np.sqrt(np.mean(vetvar))
        vetoutput = np.vstack((vetoutput,[s, fs]))

        # A escala S cresce numa série geométrica
        s = np.ceil(s * boxratio).astype(np.int)

    # Array com o log da escala S e o log da função de flutuação   
    vetoutput = np.log10(vetoutput[1::1,:])

    # Separa as colunas do vetor 'vetoutput'
    x = vetoutput[:,0]
    y = vetoutput[:,1]

    # Regressão linear
    slope, intercept, _, _, _ = scipy.stats.linregress(x, y)

    # Calcula a reta de inclinação
    predict_y = intercept + slope * x

    # Calcula o erro
    pred_error = y - predict_y
    return slope, vetoutput, x, y, predict_y, pred_error


def plot_dfa(alfa, x, y, reta, country_title):
    """Função auxiliar para a visualização dos resultados da análise DFA
    """
    # plt.figure(dpi = 300)

    plt.plot(x, y, 's', color = 'red')

    plt.plot(x, reta, '-', color = 'blue',  linewidth=1.5)
    plt.title(f'{country_title} ($\\alpha$ = {round(alfa, 2)})')
    # plt.xlabel('$log_{10} (s)$')
    # plt.ylabel('$log_{10} F(s)$')
    # plt.show()]
