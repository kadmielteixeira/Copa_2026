```
# Simulador Copa do Mundo 2026 🏆

Modelo probabilístico para simular a Copa do Mundo 2026, desenvolvido por curiosidade e diversão.
A pergunta que motivou tudo foi simples: **qual a probabilidade do Brasil ser campeão?**

---

## Resultado

Das 5.000 simulações, a seleção com maior probabilidade de título é a **França com 43,1%**.
A probabilidade do Brasil ser campeão ficou **abaixo de 1%**.

Os quatro finalistas mais prováveis são França, Portugal, Espanha e Inglaterra,
que aparecem juntos na semifinal em **1.346 das 5.000 simulações (26,92%)**.

---

## Metodologia

O modelo combina as seguintes técnicas:

**Simulação de Monte Carlo**, com 5.000 execuções independentes do torneio completo,
cada uma com seed própria para garantir reprodutibilidade.

**Distribuição de Poisson** para modelagem de gols, onde cada partida gera
um λ (lambda) individual por seleção e o número de gols é amostrado de Poisson(λ),
refletindo a natureza discreta e rara dos eventos de gol no futebol.

**MLE (Maximum Likelihood Estimation)** via método de otimização Nelder,Mead,
calibrado sobre as 96 partidas da fase de grupos das Copas de 2018 e 2022,
atingindo ~57% de acurácia preditiva (baseline aleatório: 33%).

**Cross,Validation leave,one,out** sobre os dados históricos para validar os parâmetros calibrados.

**Força composta ponderada** com pesos empíricos sobre os atributos do FC26 via SoFIFA:
Geral (42%), Ataque (30%), Meio,campo (15%) e Defesa (13%).

**Fator de defesa adversária** como multiplicador no λ do atacante,
penalizando ataques contra defesas fortes e ampliando contra defesas fracas.

**Ruído gaussiano multiplicativo** aplicado a cada λ individual,
para representar sorte, garra, azar e imprevisibilidade jogo a jogo.

**Grid search** para calibração do limiar de empate,
minimizando o erro entre a taxa simulada e a taxa histórica de empates em Copas (19,8%).

---

## Parâmetros Calibrados

    LAMBDA_BASE   = 1.2024
    EXPO_FORCA    = 0.5873
    FATOR_DEF_AMP = 1.1861
    LIMIAR        = 32

---

## Dados de Entrada

O simulador utiliza os atributos das seleções exportados do **FC26 via SoFIFA**,
com as colunas: Selecao, Geral, Ataque, MeioCampo, Defesa e Idade.

Forneça um dos dois arquivos no mesmo diretório do script:

    sofifa_selecoes.xlsx   (preferencial)
    sofifa_selecoes.txt    (fallback em texto simples)

---

## Como Usar

Instalar dependências:

    pip install -r requirements.txt

Rodar uma simulação simples (seed 42):

    python simulador_calibrado_v2.0.py

Rodar com seed específica:

    python simulador_calibrado_v2.0.py --seed 2026

Rodar 5.000 simulações paralelas:

    python simulador_calibrado_v2.0.py --multi

Rodar com número de workers customizado:

    python simulador_calibrado_v2.0.py --multi --workers 4

---

## Arquivos Gerados

    graficos_calibrado/          , gráficos PNG com ranking, heatmap e bracket
    historico_calibrado.csv      , histórico de todas as partidas simuladas
    relatorio_calibrado_2026.pdf , relatório completo da simulação (modo single)
    ranking_5000_simulacoes.pdf  , ranking agregado das 5.000 simulações (modo multi)

---

## Limitações

Os dados de entrada são estáticos (FC26): lesões e suspensões não são capturados.
O ruído gaussiano não foi calibrado via MLE, sendo um hiperparâmetro heurístico.
A simulação de pênaltis usa probabilidades heurísticas, não calibradas por dados históricos.
O heatmap de fases usa estimativa por força relativa, não frequências simuladas.

---

## Dependências

    numpy, pandas, matplotlib, seaborn, fpdf2, openpyxl

---

## Contato

Dúvidas ou sugestões? Fique à vontade para abrir uma issue ou entrar em contato.
```
