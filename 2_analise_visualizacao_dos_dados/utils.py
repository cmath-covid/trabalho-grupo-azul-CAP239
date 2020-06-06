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

# Parte do mf-dfa
def getMSSByUpscaling(dx, normType = np.inf, isDFA = 1, isNormalised = 1):
    ## Some initialiation
    aux_eps = np.finfo(float).eps
    
    # We prepare an array of values of the variable q-norm
    aux = [-16.0, -8.0, -4.0, -2.0, -1.0, -0.5, -0.0001, 0.0, 0.0001, 0.5, 0.9999, 1.0, 1.0001, 2.0, 4.0, 8.0, 16.0, 32.0]
    nq = len(aux)
    
    q = np.zeros((nq, 1))
    q[:, 1 - 1] = aux
    
    dx_len = len(dx)
    
    # We have to reserve the most major scale for shifts, so we divide the data
    # length by two. (As a result, the time measure starts from 2.0, not from
    # 1.0, see below.)
    dx_len = np.int(dx_len / 2)
    
    dx_shift = np.int(dx_len / 2)
    
    nScales = np.int(np.round(np.log2(dx_len)))    # Number of scales involved. P.S. We use 'round()' to prevent possible malcomputing of the logarithms
    j = 2 ** (np.arange(1, nScales + 1) - 1) - 1
    
    dataMeasure = np.zeros((nq, nScales))
    
    ## Computing the data measures in different q-norms
    for ji in range(1, nScales + 1):
        # At the scale 'j(ji)' we deal with '2 * (j(ji) + 1)' elements of the data 'dx'
        dx_k_len = 2 * (j[ji - 1] + 1)
        n = np.int(dx_len / dx_k_len)
        
        dx_leftShift = np.int(dx_k_len / 2)
        dx_rightShift = np.int(dx_k_len / 2)
        
        R = np.zeros(n)
        S = np.ones(n)
        for k in range(1, n + 1):
            # We get a portion of the data of the length '2 * (j(ji) + 1)' plus the data from the left and right boundaries
            dx_k_withShifts = dx[(k - 1) * dx_k_len + 1 + dx_shift - dx_leftShift - 1 : k * dx_k_len + dx_shift + dx_rightShift]
            
            # Then we perform free upscaling and, using the above-selected data (provided at the scale j = 0),
            # compute the velocities at the scale 'j(ji)'
            j_dx = np.convolve(dx_k_withShifts, np.ones(dx_rightShift), 'valid')
            
            # Then we compute the accelerations at the scale 'j(ji) + 1'
            r = (j_dx[1 + dx_rightShift - 1 : ] - j_dx[1 - 1 : -dx_rightShift]) / 2.0
            
            # Finally we compute the range ...
            if (normType == 0):
                R[k - 1] = np.max(r[2 - 1 : ]) - np.min(r[2 - 1 : ])
            elif (np.isinf(normType)):
                R[k - 1] = np.max(np.abs(r[2 - 1 : ]))
            else:
                R[k - 1] = (np.sum(r[2 - 1 : ] ** normType) / len(r[2 - 1 : ])) ** (1.0 / normType)
            # ... and the normalisation factor ("standard deviation")
            if (isDFA == 0):
                S[k - 1] = np.sqrt(np.sum(np.abs(np.diff(r)) ** 2.0) / (len(r) - 1))
    
        if (isNormalised == 1):      # Then we either normalise the R / S values, treating them as probabilities ...
            p = np.divide(R, S) / np.sum(np.divide(R, S))
        else:                        # ... or leave them unnormalised ...
            p = np.divide(R, S)
        # ... and compute the measures in the q-norms
        for k in range(1, n + 1):
            # This 'if' is needed to prevent measure blow-ups with negative values of 'q' when the probability is close to zero
            if (p[k - 1] < 1000.0 * aux_eps):
                continue
            
            dataMeasure[:, ji - 1] = dataMeasure[:, ji - 1] + np.power(p[k - 1], q[:, 1 - 1])

    # We pass from the scales ('j') to the time measure; the time measure at the scale j(nScales) (the most major one)
    # is assumed to be 2.0, while it is growing when the scale is tending to j(1) (the most minor one).
    # (The scale j(nScales)'s time measure is NOT equal to 1.0, because we reserved the highest scale for shifts
    # in the very beginning of the function.)
    timeMeasure = 2.0 * dx_len / (2 * (j + 1))
    
    scales = j + 1
    
    ## Determining the exponents 'tau' from 'dataMeasure(q, timeMeasure) ~ timeMeasure ^ tau(q)'
    tau = np.zeros((nq, 1))
    log10tm = np.log10(timeMeasure)
    log10dm = np.log10(dataMeasure)
    log10tm_mean = np.mean(log10tm)
    
    # For each value of the q-norm we compute the mean 'tau' over all the scales
    for qi in range(1, nq + 1):
        tau[qi - 1, 1 - 1] = np.sum(np.multiply(log10tm, (log10dm[qi - 1, :] - np.mean(log10dm[qi - 1, :])))) / np.sum(np.multiply(log10tm, (log10tm - log10tm_mean)))

    ## Finally, we only have to pass from 'tau(q)' to its conjugate function 'f(alpha)'
    # In doing so, first we find the Lipschitz-Holder exponents 'alpha' (represented by the variable 'LH') ...
    aux_top = (tau[2 - 1] - tau[1 - 1]) / (q[2 - 1] - q[1 - 1])
    aux_middle = np.divide(tau[3 - 1 : , 1 - 1] - tau[1 - 1 : -1 - 1, 1 - 1], q[3 - 1 : , 1 - 1] - q[1 - 1 : -1 - 1, 1 - 1])
    aux_bottom = (tau[-1] - tau[-1 - 1]) / (q[-1] - q[-1 - 1])
    LH = np.zeros((nq, 1))
    LH[:, 1 - 1] = -np.concatenate((aux_top, aux_middle, aux_bottom))
    # ... and then compute the conjugate function 'f(alpha)' itself
    f = np.multiply(LH, q) + tau

    ## The last preparations
    # We determine the minimum and maximum values of 'alpha' ...
    LH_min = LH[-1, 1 - 1]
    LH_max = LH[1 - 1, 1 - 1]
    # ... and find the minimum and maximum values of another multifractal characteristic, the so-called
    # generalised Hurst (or DFA) exponent 'h'. (These parameters are computed according to [2, p. 27].)
    h_min = -(1.0 + tau[-1, 1 - 1]) / q[-1, 1 - 1]
    h_max = -(1.0 + tau[1 - 1, 1 - 1]) / q[1 - 1, 1 - 1]
    
    stats = {'tau':       tau,
        'LH':        LH,
            'f':         f,
                'LH_min':    LH_min,
                    'LH_max':    LH_max,
                        'h_min':     h_min,
                            'h_max':     h_max}
    
    return [timeMeasure, dataMeasure, scales, stats, q]
