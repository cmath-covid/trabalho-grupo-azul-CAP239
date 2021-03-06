## Trabalho em grupo - CAP-239-4

<center>
    <img src="images/logo_inpe.png"/>
</center>

Docentes:
- Dr. Reinaldo Rosa
- Dr. Leonardo B. L. Santos

Discentes:
 - Fernando Cossetin;
 - Felipe Menino Carlos;
 - Felipe Perin.

Este é um projeto desenvolvido na disciplina de Matemática Computacional do programa de pós-graduação do Instituto Nacional de Pesquisas Espaciais (INPE), no contexto de Análise Estatística 
e Espectral de Processos Estocásticos.
<hr>

Para acessar o documento do relatório final, [clique aqui](https://github.com/cmath-covid/trabalho-grupo-azul-CAP239/blob/master/CAP-239-FCOSSETIN_FMENINO_FPERIN.pdf). Abaixo são feitas as descrições de cada uma das atividades e seu relacionamento com os arquivos do repositório.

## Atividades

1. Dataset: A partir do *dataset* da OWD (UK) construa um *dataset* com os dados indicados abaixo, relativos aos países e regiões selecionados para o grupo.

| Time 	|   C1   	| C2     	| C3     	| C4   	| C5     	| R1 	| R2      	|
|:----:	|:------:	|--------	|--------	|------	|--------	|----	|---------	|
| Blue 	| Brasil 	| Canadá 	| México 	| Cuba 	| Rússia 	| MG 	| Niterói 	|

O desenvolvimento desta atividade foi feito através da criação de uma ferramenta automatizada para a aquisição de dados. As fontes de dados utilizadas são: (i) Para os dados de países utilizou-se o [Our World In Data](https://ourworldindata.org/coronavirus); (ii) Os dados das regiões foi feita a utilização dos dados da [Brasil.io](https://brasil.io/home/).

A ferramenta desenvolvida está neste [repositório](https://github.com/cmath-covid/ferramenta-aquisicao-de-dados)

2. Visualização, Cullen-Frey, Histogramas e PDFs: A partir do dataset do seu Team visualize, obtenha os respectivos Histogramas e PDFs para as seguintes variáveis: Número Total de Casos, Número Total de Mortes, Número Total De Testes, `Número Diário de Casos` (NDC), `Número Diário de Mortes` (NDM), `Número Diário de Testes` (NDT). Identifique semelhanças e discrepâncias entre os países. Este exercício é apenas para os dados da OWD. Implemente ainda uma análise de regressão linear em Python entre as variáveis: NDC e NDT para os países que apresentam PDF próximas

Para o desenvolvimento desta atividade, foram criados diversos scripts de análise de dados, todos eles armazenados no diretório [2_analise_visualizacao_dos_dados](2_analise_visualizacao_dos_dados). Cada um dos arquivos presentes dentro deste diretório são explicados abaixo

- [2_analise_visualizacao_dos_dados/1_analise_dos_dados_owd](2_analise_visualizacao_dos_dados/1_analise_dos_dados_owd.ipynb): Script para a análise exploratória de dados, contendo todas as visualizações das variáveis requisitadas pelo exercício, neste arquivo existem também análises de DFA, clusterização e multifractalidade (Espectro de singularidades);
- [2_analise_visualizacao_dos_dados/2_analise_cullen-frey](2_analise_visualizacao_dos_dados/2_analise_cullen-frey.ipynb): Como forma de tornar os resultados observados na análise exploratória de dados, este script foi criado para facilitar a visualização em conjunto da classificação estatística no Espaço de Cullen-Frey das variáveis de atualização diária (Que apresentam flutuação);
- [2_analise_visualizacao_dos_dados/3_ajustes_pdf_somente_cullen-frey](2_analise_visualizacao_dos_dados/3_ajustes_pdf_somente_cullen-frey.ipynb): Para o segundo exercício é requisitado o ajuste de PDFs aos dados, em um primeiro momento fez-se o ajuste considerando apenas as classes de PDFs apresentadas no espaço de Cullen-Frey gerado anteriormente. Este arquivo contém somente os ajustes de PDFs para as variáveis que apresentam flutuação;
- [2_analise_visualizacao_dos_dados/4_ajustes_pdf](2_analise_visualizacao_dos_dados/4_ajustes_pdf.ipynb): Como forma de investigar o comportamento do espaço de Cullen-Frey, foi criado um script que ajusta e busca a melhor PDF dentre as 86 disponíveis no [SciPy](https://www.scipy.org/). Isso é feito para mostrar que existem informações e características que podem ser considerados para a realização de uma PDF;
- [2_analise_visualizacao_dos_dados/5_analise_de_regressao](2_analise_visualizacao_dos_dados/5_analise_de_regressao.ipynb): Por fim, faz-se a criação de análises de regressão entre as variáveis de Número Diário de Casos X Número Diário de Testes.

Estes foram os passos utilizados para a realização deste exercício 2.

3. Previsão Diária: Adapte e Aplique o modelo IMCSF-COVID19.py para cálculo automático das curvas de g e s, para todos os países e regiões, até o dia 20/5/2020. Considere o primeiro dia com mais de 50 casos.

Nesta atividade, fez-se a implementação do modelo ICMSF-COVID19 já considerando todas as necessidades para a previsão de casos em múltiplos dias, a implementação foi criada para funcionar de maneira modular, podendo ser facilmente expandida e alterada. A forma de implementação privilegia a semântica dos dados, evitando que erros relacionados a manipulação de arrays venha a causar problemas em seu funcionamento.

A implementação está disponível no diretório [3_modelo_previsao_covid19](3_modelo_previsao_covid19).

**Atualização do dia 19/06/2020**: A pedido do professor, foi adicionado no modelo uma forma diferente de cálculo do *s*. Para isto, as funções foram adaptadas para trabalhar considerando, de forma opcional, este comportamento. Um exemplo completo de uso com essas modificações está disponível no arquivo [3_modelo_previsao_covid19/modelo_atualizacao_19_06_2020.ipynb](3_modelo_previsao_covid19/modelo_atualizacao_19_06_2020.ipynb)

4. Com base na teoria da “wisdom of the crowd ” adapte o modelo para prever o comportamento das curvas de g e s até uma data fixa escolhida como input. E com base na hipótese do máximo de 30±8 semanas identifique o pico e o final (mínimo e máximo) da epidemia em cada país.

Após a implementação do modelo no passo 3, foi percebido que o comportamento do mesmo tem tendências exponenciais, então, para entender este relação, os testes do diretório [4_final_da_pandemia](4_final_da_pandemia/1_verificando_fim_da_pandemia.ipynb) foram criados, estes são utilizados como base para os argumentos apresentados no relatório final, indicando que com este comportamento não é possível estimar este comportamento.

> Por conta dos resultados apresentados pelo modelo implementado no passo 3, foi assumido que este exercício não era passível de ser resolvido, uma vez que, os crescimentos exponenciais apresentados pelo modelo impedem extrapolações que poderiam ser usadas para a avaliação do fim da pandemia. De toda forma, a teoria de wisdow of the crowd é aplicada no modelo desenvolvido no exercício 3.

5. Desenvolva um modelo de interpolação aleatória de 23 pontos entre cada Medida diária, representando número de casos suspeitos por hora. Gere as séries e verifique se há assinatura de SOC a partir de todos os dados de flutuação. 

Nesta atividade, foram implementadas dois tipos diferentes de interpolação aleatória, a interpolação contínua e discreta, sendo cada uma dessas apresentadas no arquivo [5_lstm_soc/1_interpolacao_e_soc](5_lstm_soc/1_interpolacao_e_soc.ipynb). Com base na interpolação aleatória criada, é feita a análise de SOC, que está no mesmo arquivo em que os códigos de interpolação aleatória são definidos e explicados.

As implementações da LSTM são apresentadas no diretório [5_lstm_soc](5_lstm_soc)

## Desenvolvendo

Abaixo são listadas instruções que devem ser seguidas para a contribuição com este trabalho.

> Estes passos são definidos para a organização do grupo.

- 1̣° Faça o fork do projeto;
- 2° Crie um diretório com o número de sua atividade seguido de um título, separado por *underline* (e. g. 1_conjunto_de_dados);
- 3° Para desenvolver as atividades com Python, recomenda-se trabalhar com o [Jupyter Notebook](https://jupyter.org/). Ao trabalhar com o Jupyter considere a utilização do [template](template/template.ipynb);
- 4° Ao final as atividades, faça um Pull Request.

A criação destes passos foi feita apenas para manter tudo organizado durante o desenvolvimento do projeto.
