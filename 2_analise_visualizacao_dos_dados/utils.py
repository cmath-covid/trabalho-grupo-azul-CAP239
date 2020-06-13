import scipy.stats
import numpy as np
import pandas as pd
from math import ceil
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew


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


def cullen_frey(df, method='unbiased',discrete=False,boot=None): 
    """Código retirado do github da disciplina de Matemática Computacional 1.
    """

    if len(np.shape(df)) > 1: 
        raise TypeError('Samples must be a list with N x 1 dimensions')
        
    if not isinstance(df,list):
        df = list(df)

    if len(df) < 4:
        raise ValueError('The number of samples needs to be greater than 4')

    if boot is not None:    
        if not isinstance(boot,int):
            raise ValueError('boot must be integer')

    if method=='unbiased':        
        skewdata = skew(df, bias=False)
        kurtdata = kurtosis(df, bias=False)+3

    elif method=='sample':
        skewdata = skew(df, bias=True)
        kurtdata = kurtosis(df, bias=True)+3
 
    if boot is not None:
        if boot < 10:
            raise ValueError('boot must be greater than 10')

        n = len(df)

        nrow = n
        ncol = boot
        databoot = np.reshape(np.random.choice(df, size=n*boot, replace=True),(nrow,ncol)) 

        s2boot = (skew(pd.DataFrame(databoot)))**2
        kurtboot = kurtosis(pd.DataFrame(databoot))+3

        kurtmax = max(10,ceil(max(kurtboot)))
        xmax = max(4,ceil(max(s2boot)))

    else:
        kurtmax = max(10,ceil(kurtdata))
        xmax = max(4,ceil(skewdata**2))

    ymax = kurtmax-1

    res = [min(df),max(df),np.median(df),np.mean(df),np.std(df),skew(df),kurtosis(df)]

    # If discrete = False
    if not discrete:
        #Beta distribution
        p = np.exp(-100)
        lq = np.arange(-100,100.1,0.1)
        q = np.exp(lq)
        s2a = (4*(q-p)**2*(p+q+1))/((p+q+2)**2*p*q)
        ya = kurtmax-(3*(p+q+1)*(p*q*(p+q-6)+2*(p+q)**2)/(p*q*(p+q+2)*(p+q+3)))
        p = np.exp(100)
        lq = np.arange(-100,100.1,0.1)
        q = np.exp(lq)
        s2b = (4*(q-p)**2*(p+q+1))/((p+q+2)**2*p*q)
        yb = kurtmax-(3*(p+q+1)*(p*q*(p+q-6)+2*(p+q)**2)/(p*q*(p+q+2)*(p+q+3)))
        s2 = [*s2a,*s2b]
        y = [*ya,*yb]
        
        #Gama distribution
        lshape_gama = np.arange(-100,100,0.1)
        shape_gama = np.exp(lshape_gama)
        s2_gama = 4/shape_gama
        y_gama = kurtmax-(3+6/shape_gama) 
        
        #Lognormal distribution
        lshape_lnorm = np.arange(-100,100,0.1)
        shape_lnorm = np.exp(lshape_lnorm)
        es2_lnorm = np.exp(shape_lnorm**2, dtype=np.float64)
        s2_lnorm = (es2_lnorm+2)**2*(es2_lnorm-1)
        y_lnorm = kurtmax-(es2_lnorm**4+2*es2_lnorm**3+3*es2_lnorm**2-3)

        plt.figure(figsize=(12,9))

        #observations
        obs = plt.scatter(skewdata**2,kurtmax-kurtdata,s=200, c='blue', 
                      label='Observations',zorder=10)
        #beta
        beta = plt.fill(s2,y,color='lightgrey',alpha=0.6, label='beta', zorder=0)
        #gama
        gama = plt.plot(s2_gama,y_gama, '--', c='k', label='gama')
        #lognormal
        lnormal = plt.plot(s2_lnorm,y_lnorm, c='k', label='lognormal')
    
        if boot is not None:
            #bootstrap 
            bootstrap = plt.scatter(s2boot,kurtmax-kurtboot,marker='$\circ$',c='orange',s=50, 
                                    label='Bootstrap values', zorder=5)
           
        
        #markers
        normal = plt.scatter(0,kurtmax-3, marker=(8,2,0),s=400,c='k',label='normal',zorder=5)
        
        uniform = plt.scatter(0,kurtmax-9/5, marker='$\\bigtriangleup$',s=400,c='k',label='uniform',zorder=5)   
        
        exp_dist = plt.scatter(2**2,kurtmax-9, marker='$\\bigotimes$',s=400,c='k',label='exponential',zorder=5) 
        
        logistic = plt.scatter(0,kurtmax-4.2, marker='+',s=400,c='k',label='logistic',zorder=5)
        
        
        #Adjusting the axis
        yax = [str(kurtmax - i) for i in range(0,ymax+1)]
        plt.xlim(-0.08, xmax+0.4)
        plt.ylim(-1, ymax+0.08)
        plt.yticks(list(range(0,ymax+1)),labels=yax)
        
        #Adding the labels
        plt.xlabel('square of skewness', fontsize=13)
        plt.ylabel('kurtosis', fontsize=13)
        plt.title('Cullen and Frey graph', fontsize=15) 
        
        #Adding the legends
        legenda2 = plt.legend(handles=[obs,bootstrap],loc='upper center', labelspacing=1, frameon=False)
        plt.gca().add_artist(legenda2)

        plt.legend(handles=[normal,uniform,exp_dist,logistic,beta[0],lnormal[0],gama[0]], 
               title='Theoretical distributions',loc='upper right',labelspacing=1.4,frameon=False)
    
        plt.show()

    #If discrete = True
    else:
        # negbin distribution
        p = np.exp(-10)
        lr = np.arange(-100,100,0.1)
        r = np.exp(lr)
        s2a = (2-p)**2/(r*(1-p))
        ya = kurtmax-(3+6/r+p**2/(r*(1-p)))
        p = 1-np.exp(-10)
        lr = np.arange(100,-100,-0.1)
        r = np.exp(lr)
        s2b = (2-p)**2/(r*(1-p))
        yb = kurtmax-(3+6/r+p**2/(r*(1-p)))
        s2_negbin = [*s2a,*s2b]
        y_negbin = [*ya,*yb]

        # poisson distribution
        llambda = np.arange(-100,100,0.1)
        lambda_ = np.exp(llambda)
        s2_poisson = 1/lambda_
        y_poisson = kurtmax-(3+1/lambda_)
    
        plt.figure(figsize=(12,9))          
    
        #observations
        obs = plt.scatter(skewdata**2,kurtmax-kurtdata,s=200, c='blue', 
                      label='Observations',zorder=10)

        #negative binomial
        negbin = plt.fill(s2_negbin,y_negbin,color='lightgrey',alpha=0.6, label='negative binomial', zorder=0)

        #poisson
        poisson = plt.plot(s2_poisson,y_poisson, '--', c='k', label='poisson')

        if boot is not None:
            #bootstrap 
            bootstrap = plt.scatter(s2boot,kurtmax-kurtboot,marker='$\circ$',c='orange',s=50, 
                                    label='Bootstrap values', zorder=5)
  

        #markers
        normal = plt.scatter(0,kurtmax-3, marker=(8,2,0),s=400,c='k',label='normal',zorder=5)
    
        #adjusting the axis
        yax = [str(kurtmax - i) for i in range(0,ymax+1)]
        plt.xlim(-0.08, xmax+0.4)
        plt.ylim(-1, ymax+0.08)
        plt.yticks(list(range(0,ymax+1)),labels=yax)
    
        #adding the labels
        plt.xlabel('square of skewness', fontsize=13)
        plt.ylabel('kurtosis', fontsize=13)
        plt.title('Cullen and Frey graph', fontsize=15) 
    
        #adding the legends
        legenda1 = plt.legend(handles=[obs,bootstrap],loc='upper center', labelspacing=1, frameon=False)
        plt.gca().add_artist(legenda1)
    
        plt.legend(handles=[normal,negbin[0],poisson[0]],title='Theoretical distributions',loc='upper right',
                   labelspacing=1.4,frameon=False)
    
        plt.show()

    #print some statistical information
    print('=== summary statistics ===')
    print(f'min:{res[0]:.4f}\nmax:{res[1]:.4f}\nmean:{res[3]:.4f}\nstandard deviation:{res[4]:.4f}'\
	f'\nskewness:{res[5]:.4f}\nkurtosis:{res[6]:.4f} +3 for the plot')



def cullen_frey_subplot(df, method='unbiased',discrete=False,boot=None, legend = True, 
                            title = 'Cullen and Frey graph', 
                            xlabel = 'square of skewness', ylabel = 'kurtosis', 
                            legend_loc1 = 'upper center', legend_loc2 = 'upper right',
                            lim = True): 
    """Código retirado do github da disciplina de Matemática Computacional 1. Esta função
    foi adaptada para o funcionamento em subplots. As alterações foram feitas apenas na posição
    da legenda, na geração das figuras e definição dos limites. As lógicas não foram alteradas
    """

    if len(np.shape(df)) > 1: 
        raise TypeError('Samples must be a list with N x 1 dimensions')
        
    if not isinstance(df,list):
        df = list(df)

    if len(df) < 4:
        raise ValueError('The number of samples needs to be greater than 4')

    if boot is not None:    
        if not isinstance(boot,int):
            raise ValueError('boot must be integer')

    if method=='unbiased':        
        skewdata = skew(df, bias=False)
        kurtdata = kurtosis(df, bias=False)+3

    elif method=='sample':
        skewdata = skew(df, bias=True)
        kurtdata = kurtosis(df, bias=True)+3
 
    if boot is not None:
        if boot < 10:
            raise ValueError('boot must be greater than 10')

        n = len(df)

        nrow = n
        ncol = boot
        databoot = np.reshape(np.random.choice(df, size=n*boot, replace=True),(nrow,ncol)) 

        s2boot = (skew(pd.DataFrame(databoot)))**2
        kurtboot = kurtosis(pd.DataFrame(databoot))+3

        kurtmax = max(10,ceil(max(kurtboot)))
        xmax = max(4,ceil(max(s2boot)))

    else:
        kurtmax = max(10,ceil(kurtdata))
        xmax = max(4,ceil(skewdata**2))

    ymax = kurtmax-1

    res = [min(df),max(df),np.median(df),np.mean(df),np.std(df),skew(df),kurtosis(df)]

    # If discrete = False
    if not discrete:
        #Beta distribution
        p = np.exp(-100)
        lq = np.arange(-100,100.1,0.1)
        q = np.exp(lq)
        s2a = (4*(q-p)**2*(p+q+1))/((p+q+2)**2*p*q)
        ya = kurtmax-(3*(p+q+1)*(p*q*(p+q-6)+2*(p+q)**2)/(p*q*(p+q+2)*(p+q+3)))
        p = np.exp(100)
        lq = np.arange(-100,100.1,0.1)
        q = np.exp(lq)
        s2b = (4*(q-p)**2*(p+q+1))/((p+q+2)**2*p*q)
        yb = kurtmax-(3*(p+q+1)*(p*q*(p+q-6)+2*(p+q)**2)/(p*q*(p+q+2)*(p+q+3)))
        s2 = [*s2a,*s2b]
        y = [*ya,*yb]
        
        #Gama distribution
        lshape_gama = np.arange(-100,100,0.1)
        shape_gama = np.exp(lshape_gama)
        s2_gama = 4/shape_gama
        y_gama = kurtmax-(3+6/shape_gama) 
        
        #Lognormal distribution
        lshape_lnorm = np.arange(-100,100,0.1)
        shape_lnorm = np.exp(lshape_lnorm)
        es2_lnorm = np.exp(shape_lnorm**2, dtype=np.float64)
        s2_lnorm = (es2_lnorm+2)**2*(es2_lnorm-1)
        y_lnorm = kurtmax-(es2_lnorm**4+2*es2_lnorm**3+3*es2_lnorm**2-3)

        # plt.figure(figsize=(12,9))

        #observations
        obs = plt.scatter(skewdata**2,kurtmax-kurtdata,s=200, c='blue', 
                      label='Observations',zorder=10)
        #beta
        beta = plt.fill(s2,y,color='lightgrey',alpha=0.6, label='beta', zorder=0)
        #gama
        gama = plt.plot(s2_gama,y_gama, '--', c='k', label='gama')
        #lognormal
        lnormal = plt.plot(s2_lnorm,y_lnorm, c='k', label='lognormal')
    
        if boot is not None:
            #bootstrap 
            bootstrap = plt.scatter(s2boot,kurtmax-kurtboot,marker='$\circ$',c='orange',s=50, 
                                    label='Bootstrap values', zorder=5)
           
        
        #markers
        normal = plt.scatter(0,kurtmax-3, marker=(8,2,0),s=400,c='k',label='normal',zorder=5)
        
        uniform = plt.scatter(0,kurtmax-9/5, marker='$\\bigtriangleup$',s=400,c='k',label='uniform',zorder=5)   
        
        exp_dist = plt.scatter(2**2,kurtmax-9, marker='$\\bigotimes$',s=400,c='k',label='exponential',zorder=5) 
        
        logistic = plt.scatter(0,kurtmax-4.2, marker='+',s=400,c='k',label='logistic',zorder=5)
        
        
        #Adjusting the axis
        yax = [str(kurtmax - i) for i in range(0,ymax+1)]
        if lim:
            plt.xlim(-0.08, xmax+0.4)
            plt.ylim(-1, ymax+0.08)
        plt.yticks(list(range(0,ymax+1)),labels=yax)
        
        #Adding the labels
        plt.xlabel(xlabel, fontsize=13)
        plt.ylabel(ylabel, fontsize=13)
        plt.title(title, fontsize=15)
        
        #Adding the legends
        if legend:
            legenda2 = plt.legend(handles=[obs,bootstrap],loc=legend_loc1, labelspacing=1, frameon=False)
            plt.gca().add_artist(legenda2)

            plt.legend(handles=[normal,uniform,exp_dist,logistic,beta[0],lnormal[0],gama[0]], 
                   title='Theoretical distributions',loc=legend_loc2,labelspacing=1.4,frameon=False)
    
        # plt.show()

    #If discrete = True
    else:
        # negbin distribution
        p = np.exp(-10)
        lr = np.arange(-100,100,0.1)
        r = np.exp(lr)
        s2a = (2-p)**2/(r*(1-p))
        ya = kurtmax-(3+6/r+p**2/(r*(1-p)))
        p = 1-np.exp(-10)
        lr = np.arange(100,-100,-0.1)
        r = np.exp(lr)
        s2b = (2-p)**2/(r*(1-p))
        yb = kurtmax-(3+6/r+p**2/(r*(1-p)))
        s2_negbin = [*s2a,*s2b]
        y_negbin = [*ya,*yb]

        # poisson distribution
        llambda = np.arange(-100,100,0.1)
        lambda_ = np.exp(llambda)
        s2_poisson = 1/lambda_
        y_poisson = kurtmax-(3+1/lambda_)
    
        # plt.figure(figsize=(12,9))          
    
        #observations
        obs = plt.scatter(skewdata**2,kurtmax-kurtdata,s=200, c='blue', 
                      label='Observations',zorder=10)

        #negative binomial
        negbin = plt.fill(s2_negbin,y_negbin,color='lightgrey',alpha=0.6, label='negative binomial', zorder=0)

        #poisson
        poisson = plt.plot(s2_poisson,y_poisson, '--', c='k', label='poisson')

        if boot is not None:
            #bootstrap 
            bootstrap = plt.scatter(s2boot,kurtmax-kurtboot,marker='$\circ$',c='orange',s=50, 
                                    label='Bootstrap values', zorder=5)
  

        #markers
        normal = plt.scatter(0,kurtmax-3, marker=(8,2,0),s=400,c='k',label='normal',zorder=5)
    
        #adjusting the axis
        yax = [str(kurtmax - i) for i in range(0,ymax+1)]
        if lim:
            plt.xlim(-0.08, xmax+0.4)
            plt.ylim(-1, ymax+0.08)
        plt.yticks(list(range(0,ymax+1)),labels=yax)
    
        #adding the labels
        plt.xlabel('square of skewness', fontsize=13)
        plt.ylabel('kurtosis', fontsize=13)
        plt.title('Cullen and Frey graph', fontsize=15) 
    
        #adding the legends
        if legend:
            legenda1 = plt.legend(handles=[obs,bootstrap],loc=legend_loc1, labelspacing=1, frameon=False)
            plt.gca().add_artist(legenda1)

            plt.legend(handles=[normal,negbin[0],poisson[0]],title='Theoretical distributions',loc=legend_loc2,
                       labelspacing=1.4,frameon=False)
