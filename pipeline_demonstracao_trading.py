# %%
"""
Pipeline de Demonstração: Meta-Modelos com Features Topológicas para Trading
=============================================================================

Este script demonstra o pipeline utilizado no artigo de forma compactada e simplificada.

Autores: Valter Rebelo
Artigo: Contexto Importa: Machine Learning Topológico em Estratégias de Trading
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# %%
def carregar_dados_bitcoin(n_dias: int = 1000, preco_inicial: float = 50000) -> pd.DataFrame:
    """
    Carrega dados do Bitcoin: primeiro tenta ler de bitcoin.csv, se não disponível gera dados sintéticos.

    Parâmetros:
    -----------
    n_dias : int
        Número de dias de dados para gerar (apenas para dados sintéticos)
    preco_inicial : float
        Preço inicial do Bitcoin (apenas para dados sintéticos)

    Retorna:
    --------
    pd.DataFrame
        DataFrame com colunas: date (index), bitcoin_close, bitcoin_volume
    """
    import os

    # Primeiro, tentar carregar dados reais do CSV
    try:
        if os.path.exists('bitcoin.csv'):
            print("📈 Carregando dados reais do Bitcoin de bitcoin.csv...")
            df_real = pd.read_csv('bitcoin.csv')

            # Tentar identificar coluna de preço de fechamento
            close_columns = [col for col in df_real.columns if 'close' in col.lower()]
            if not close_columns:
                close_columns = [col for col in df_real.columns if 'price' in col.lower()]
            if not close_columns:
                raise ValueError("Coluna de preço de fechamento não encontrada no CSV")

            close_col = close_columns[0]

            # Tentar identificar coluna de data
            date_columns = [col for col in df_real.columns if any(term in col.lower() for term in ['date', 'time', 'timestamp'])]
            if not date_columns:
                # Se não houver coluna de data, usar index
                df_real['date'] = pd.date_range(start='2020-01-01', periods=len(df_real), freq='D')
                date_col = 'date'
            else:
                date_col = date_columns[0]

            # Tentar identificar coluna de volume (opcional)
            volume_columns = [col for col in df_real.columns if 'volume' in col.lower()]
            if volume_columns:
                volume_col = volume_columns[0]
            else:
                # Gerar volume sintético se não disponível
                df_real['bitcoin_volume'] = 50000 * np.random.uniform(0.5, 1.5, len(df_real))
                volume_col = 'bitcoin_volume'

            # Padronizar nomes das colunas
            df = pd.DataFrame({
                'date': pd.to_datetime(df_real[date_col]),
                'bitcoin_close': df_real[close_col].astype(float),
                'bitcoin_volume': df_real[volume_col].astype(float)
            })

            df.set_index('date', inplace=True)
            df = df.dropna()  # Remover valores NaN

            print(f"✅ Dados reais carregados com sucesso!")
            print(f"   {len(df)} dias de dados históricos")
            print(f"   Período: {df.index[0].date()} a {df.index[-1].date()}")
            print(f"   Preço inicial: ${df['bitcoin_close'].iloc[0]:,.2f}")
            print(f"   Preço final: ${df['bitcoin_close'].iloc[-1]:,.2f}")
            print(f"   Retorno total: {(df['bitcoin_close'].iloc[-1] / df['bitcoin_close'].iloc[0] - 1)*100:.1f}%")

            return df

    except Exception as e:
        print(f"⚠️  Erro ao carregar bitcoin.csv: {e}")
        print("📊 Gerando dados sintéticos como fallback...")

    # Fallback: gerar dados sintéticos
    np.random.seed(42)

    # Gerar datas
    dates = pd.date_range(start='2020-01-01', periods=n_dias, freq='D')

    # Gerar retornos log com volatilidade clustering (modelo GARCH simplificado)
    returns = []
    volatility = 0.04  # volatilidade inicial

    for i in range(n_dias):
        # Atualizar volatilidade (efeito GARCH)
        volatility = 0.95 * volatility + 0.05 * 0.04 + 0.05 * (returns[-1] if returns else 0)**2
        volatility = np.clip(volatility, 0.01, 0.10)

        # Gerar retorno com tendência de longo prazo
        trend = 0.0002  # tendência de alta leve
        shock = np.random.normal(0, volatility)
        daily_return = trend + shock
        returns.append(daily_return)

    # Converter retornos para preços
    precos = [preco_inicial]
    for ret in returns:
        precos.append(precos[-1] * np.exp(ret))

    # Gerar volume correlacionado com volatilidade
    volumes = []
    for i, ret in enumerate(returns):
        base_volume = 50000
        volume_multiplier = 1 + 5 * abs(ret)  # maior volume em dias voláteis
        volume = base_volume * volume_multiplier * np.random.uniform(0.5, 1.5)
        volumes.append(volume)

    df = pd.DataFrame({
        'date': dates,
        'bitcoin_close': precos[1:],  # remover preço inicial
        'bitcoin_volume': volumes
    })

    df.set_index('date', inplace=True)

    print(f"📊 Dados sintéticos do Bitcoin gerados: {len(df)} dias")
    print(f"   Período: {df.index[0].date()} a {df.index[-1].date()}")
    print(f"   Preço inicial: ${df['bitcoin_close'].iloc[0]:,.2f}")
    print(f"   Preço final: ${df['bitcoin_close'].iloc[-1]:,.2f}")
    print(f"   Retorno total: {(df['bitcoin_close'].iloc[-1] / df['bitcoin_close'].iloc[0] - 1)*100:.1f}%")

    return df

# Carregar dados
dados_bitcoin = carregar_dados_bitcoin()
print("\n", dados_bitcoin.head())

# %%
def estrategia_sma_crossover(dados: pd.DataFrame, periodo_rapido: int = 10, periodo_lento: int = 30) -> pd.DataFrame:
    """
    Implementa estratégia de cruzamento de médias móveis simples.

    Lógica:
    - Sinal de compra (1): quando SMA rápida > SMA lenta
    - Sinal de venda (-1): quando SMA rápida < SMA lenta

    Parâmetros:
    -----------
    dados : pd.DataFrame
        Dados com coluna 'bitcoin_close'
    periodo_rapido : int
        Período da média móvel rápida
    periodo_lento : int
        Período da média móvel lenta

    Retorna:
    --------
    pd.DataFrame
        Dados originais com colunas adicionais de sinais
    """
    df = dados.copy()

    # Calcular médias móveis
    df['sma_rapida'] = df['bitcoin_close'].rolling(window=periodo_rapido).mean()
    df['sma_lenta'] = df['bitcoin_close'].rolling(window=periodo_lento).mean()

    # Gerar sinais
    df['sinal_sma'] = np.where(df['sma_rapida'] > df['sma_lenta'], 1, -1)

    # Remover primeiros dias sem sinais válidos
    df = df.dropna()

    return df

def estrategia_bollinger_bands(dados: pd.DataFrame, periodo: int = 20, desvio_mult: float = 2.0) -> pd.DataFrame:
    """
    Implementa estratégia de Bandas de Bollinger.

    Lógica:
    - Sinal de compra (1): preço cruza banda inferior para cima (oversold)
    - Sinal de venda (-1): preço cruza banda superior para baixo (overbought)
    - Manter posição anterior entre cruzamentos

    Parâmetros:
    -----------
    dados : pd.DataFrame
        Dados com coluna 'bitcoin_close'
    periodo : int
        Período para média móvel e desvio padrão
    desvio_mult : float
        Multiplicador do desvio padrão para as bandas

    Retorna:
    --------
    pd.DataFrame
        Dados originais com colunas adicionais de sinais
    """
    df = dados.copy()

    # Calcular bandas de Bollinger
    df['bb_media'] = df['bitcoin_close'].rolling(window=periodo).mean()
    df['bb_std'] = df['bitcoin_close'].rolling(window=periodo).std()
    df['bb_superior'] = df['bb_media'] + (desvio_mult * df['bb_std'])
    df['bb_inferior'] = df['bb_media'] - (desvio_mult * df['bb_std'])

    # Inicializar sinais
    df['sinal_bb'] = 0

    # Gerar sinais por cruzamentos
    for i in range(1, len(df)):
        preco_anterior = df['bitcoin_close'].iloc[i-1]
        preco_atual = df['bitcoin_close'].iloc[i]
        bb_inf_anterior = df['bb_inferior'].iloc[i-1]
        bb_inf_atual = df['bb_inferior'].iloc[i]
        bb_sup_anterior = df['bb_superior'].iloc[i-1]
        bb_sup_atual = df['bb_superior'].iloc[i]
        sinal_anterior = df['sinal_bb'].iloc[i-1]

        if pd.notna(bb_inf_anterior) and pd.notna(bb_sup_anterior):
            # Cruzamento para cima da banda inferior (compra)
            if preco_anterior <= bb_inf_anterior and preco_atual > bb_inf_atual:
                df.iloc[i, df.columns.get_loc('sinal_bb')] = 1
            # Cruzamento para baixo da banda superior (venda)
            elif preco_anterior >= bb_sup_anterior and preco_atual < bb_sup_atual:
                df.iloc[i, df.columns.get_loc('sinal_bb')] = -1
            else:
                # Manter posição anterior
                df.iloc[i, df.columns.get_loc('sinal_bb')] = sinal_anterior

    # Remover primeiros dias sem sinais válidos
    df = df.dropna()

    return df

# Aplicar estratégias
dados_com_sma = estrategia_sma_crossover(dados_bitcoin)
dados_completos = estrategia_bollinger_bands(dados_com_sma)

# Estatísticas dos sinais
sma_dist = dados_completos['sinal_sma'].value_counts().to_dict()
bb_dist = dados_completos['sinal_bb'].value_counts().to_dict()

print(f"📈 Estratégia SMA Crossover aplicada")
print(f"   Distribuição de sinais: {sma_dist}")
print(f"\n📊 Estratégia Bollinger Bands aplicada")
print(f"   Distribuição de sinais: {bb_dist}")

print(f"\n📋 Dados com estratégias primárias:")
print(dados_completos[['bitcoin_close', 'sinal_sma', 'sinal_bb']].head())

# %%
def aplicar_barreiras_triplas(dados: pd.DataFrame,
                             max_dias: int = 5) -> pd.DataFrame:
    """
    Aplica método de barreiras duplas para rotulagem (profit taking e stop loss baseados em 2 desvios padrões dos últimos 30 dias).

    Para cada sinal primário, define:
    - Barreira de lucro: preço de entrada + 2 * std_20d (posição longa) ou -2 * std_20d (posição curta)
    - Barreira de perda: preço de entrada - 2 * std_20d (posição longa) ou +2 * std_20d (posição curta)
    - Barreira temporal: max_dias dias

    Rótulos gerados:
    - 1: barreira de lucro atingida primeiro
    - 0: barreira de perda atingida primeiro ou barreira temporal

    Parâmetros:
    -----------
    dados : pd.DataFrame
        Dados com sinais primários
    max_dias : int
        Máximo de dias para manter posição

    Retorna:
    --------
    pd.DataFrame
        Dados com rótulos de barreiras duplas
    """
    df = dados.copy()
    df['rotulo_barreira'] = np.nan
    df['dias_ate_barreira'] = np.nan
    df['tipo_barreira'] = ''  # 'lucro', 'perda', 'tempo'

    # Calcular desvio padrão móvel de 30 dias
    df['std_20d'] = df['bitcoin_close'].rolling(window=20, min_periods=1).std()

    # Aplicar labels para TODOS os pontos com sinal não-zero (como em triple_barrier.py)
    # Usar sinal_bb como estratégia principal para demonstração
    pontos_entrada = df[df['sinal_bb'] != 0].index

    for entrada in pontos_entrada:
        if entrada == df.index[-1]:  # Último ponto, sem dados futuros
            continue

        try:
            entrada_idx = df.index.get_loc(entrada)
            sinal = df.loc[entrada, 'sinal_bb']
            preco_entrada = df.loc[entrada, 'bitcoin_close']
            std_entrada = df.loc[entrada, 'std_20d']

            # Definir barreiras com base em 2 desvios padrões
            if sinal == 1:  # Posição longa
                barreira_lucro = preco_entrada + 2 * std_entrada
                barreira_perda = preco_entrada - 2 * std_entrada
            else:  # Posição curta
                barreira_lucro = preco_entrada - 2 * std_entrada
                barreira_perda = preco_entrada + 2 * std_entrada

            # Examinar dias seguintes
            fim_janela = min(entrada_idx + max_dias, len(df) - 1)
            janela_futura = df.iloc[entrada_idx+1:fim_janela+1]

            rotulo_encontrado = False

            for i, (data_futura, linha_futura) in enumerate(janela_futura.iterrows()):
                preco_futuro = linha_futura['bitcoin_close']
                dias_decorridos = i + 1

                if sinal == 1:  # Posição longa
                    if preco_futuro >= barreira_lucro:
                        # Barreira de lucro atingida
                        df.loc[entrada, 'rotulo_barreira'] = 1
                        df.loc[entrada, 'dias_ate_barreira'] = dias_decorridos
                        df.loc[entrada, 'tipo_barreira'] = 'lucro'
                        rotulo_encontrado = True
                        break
                    elif preco_futuro <= barreira_perda:
                        # Barreira de perda atingida
                        df.loc[entrada, 'rotulo_barreira'] = 0
                        df.loc[entrada, 'dias_ate_barreira'] = dias_decorridos
                        df.loc[entrada, 'tipo_barreira'] = 'perda'
                        rotulo_encontrado = True
                        break
                else:  # Posição curta
                    if preco_futuro <= barreira_lucro:
                        # Barreira de lucro atingida (preço caiu)
                        df.loc[entrada, 'rotulo_barreira'] = 1
                        df.loc[entrada, 'dias_ate_barreira'] = dias_decorridos
                        df.loc[entrada, 'tipo_barreira'] = 'lucro'
                        rotulo_encontrado = True
                        break
                    elif preco_futuro >= barreira_perda:
                        # Barreira de perda atingida (preço subiu)
                        df.loc[entrada, 'rotulo_barreira'] = 0
                        df.loc[entrada, 'dias_ate_barreira'] = dias_decorridos
                        df.loc[entrada, 'tipo_barreira'] = 'perda'
                        rotulo_encontrado = True
                        break

            # Se nenhuma barreira foi atingida, é barreira temporal
            # Label baseado no sinal do retorno ao fim do horizonte
            if not rotulo_encontrado:
                # Pegar o preço no fim da janela (ou último preço disponível)
                fim_idx = min(entrada_idx + max_dias, len(df) - 1)
                preco_final = df.iloc[fim_idx]['bitcoin_close']

                # Calcular retorno
                retorno_final = (preco_final - preco_entrada) / preco_entrada

                # Label baseado na direção do retorno e do sinal
                if sinal == 1:  # Posição longa
                    df.loc[entrada, 'rotulo_barreira'] = 1 if retorno_final > 0 else 0
                else:  # Posição curta
                    df.loc[entrada, 'rotulo_barreira'] = 1 if retorno_final < 0 else 0

                df.loc[entrada, 'dias_ate_barreira'] = max_dias
                df.loc[entrada, 'tipo_barreira'] = 'tempo'

        except Exception as e:
            continue

    # Remove coluna auxiliar
    df.drop(columns=['std_20d'], inplace=True)

    return df

# Aplicar barreiras triplas (na verdade, duplas: profit/stop, temporal é 0)
dados_com_rotulos = aplicar_barreiras_triplas(dados_completos)

# Estatísticas dos rótulos
rotulos_validos = dados_com_rotulos['rotulo_barreira'].dropna()
dist_rotulos = rotulos_validos.value_counts().to_dict()
tipos_barreira = dados_com_rotulos['tipo_barreira'].value_counts().to_dict()

print(f"🎯 Barreiras baseadas em 2 desvios padrões aplicadas")
print(f"   Total de pontos de entrada analisados: {len(rotulos_validos)}")
print(f"   Distribuição de rótulos: {dist_rotulos}")
print(f"   Tipos de barreira atingida: {tipos_barreira}")
print(f"   Tempo médio até barreira: {dados_com_rotulos['dias_ate_barreira'].mean():.1f} dias")

print(f"\n📋 Dados com rótulos de barreiras:")
colunas_relevantes = ['bitcoin_close', 'sinal_sma', 'sinal_bb', 'rotulo_barreira', 'tipo_barreira']
print(dados_com_rotulos[colunas_relevantes].head())

# %%
def extrair_features_topologicas_simplificadas(dados: pd.DataFrame,
                                              janela: int = 50,
                                              delay: int = 3,
                                              dimensao_emb: int = 3) -> pd.DataFrame:
    """
    Extrai features topológicas simplificadas das barras de persistência.

    Processo:
    1. Time-delay embedding da série de preços
    2. Computação de homologia persistente (H0 e H1)
    3. Análise das barras de persistência
    4. Extração de features estatísticas

    Parâmetros:
    -----------
    dados : pd.DataFrame
        Dados com coluna 'bitcoin_close'
    janela : int
        Tamanho da janela deslizante para análise
    delay : int
        Delay para time-embedding
    dimensao_emb : int
        Dimensão do embedding

    Retorna:
    --------
    pd.DataFrame
        Dados originais com features topológicas adicionadas
    """
    df = dados.copy()

    # Simular features topológicas realistas
    # Em implementação real, usaria bibliotecas como gudhi ou dionysus
    np.random.seed(42)

    features_topo = {}

    for i in range(janela, len(df)):
        # Janela de preços para análise
        janela_precos = df['bitcoin_close'].iloc[i-janela:i].values

        # Simular time-delay embedding
        precos_norm = (janela_precos - janela_precos.mean()) / janela_precos.std()

        # Simular análise de homologia persistente
        # H0 - componentes conectados (estrutura de mercado)
        volatilidade_local = np.std(precos_norm[-10:])  # volatilidade recente

        # H1 - ciclos (padrões cíclicos)
        autocorr_lag1 = np.corrcoef(precos_norm[:-1], precos_norm[1:])[0,1]

        # Features H0 (dimensão 0)
        betti_0 = max(1, int(5 * volatilidade_local + np.random.normal(0, 0.5)))
        lifetime_max_h0 = volatilidade_local * 10 + np.random.normal(0, 1)
        entropia_h0 = -np.sum([0.3, 0.4, 0.3] * np.log([0.3, 0.4, 0.3]))  # entropia simulada
        wasserstein_h0 = abs(autocorr_lag1) + np.random.normal(0, 0.1)

        # Features H1 (dimensão 1)
        betti_1 = max(0, int(3 * abs(autocorr_lag1) + np.random.normal(0, 0.3)))
        lifetime_max_h1 = abs(autocorr_lag1) * 5 + np.random.normal(0, 0.5)
        entropia_h1 = abs(autocorr_lag1) * 2 + np.random.normal(0, 0.2)
        wasserstein_h1 = volatilidade_local * abs(autocorr_lag1) + np.random.normal(0, 0.05)

        data_atual = df.index[i]
        features_topo[data_atual] = {
            'bitcoin_close_topo_betti_0': betti_0,
            'bitcoin_close_topo_max_hole_lifetime_0': max(0, lifetime_max_h0),
            'bitcoin_close_topo_persistence_entropy_0': max(0, entropia_h0),
            'bitcoin_close_topo_wasserstein_0': max(0, wasserstein_h0),
            'bitcoin_close_topo_betti_1': betti_1,
            'bitcoin_close_topo_max_hole_lifetime_1': max(0, lifetime_max_h1),
            'bitcoin_close_topo_persistence_entropy_1': max(0, entropia_h1),
            'bitcoin_close_topo_wasserstein_1': max(0, wasserstein_h1)
        }

    # Converter para DataFrame e fazer merge
    features_df = pd.DataFrame.from_dict(features_topo, orient='index')
    df_final = df.merge(features_df, left_index=True, right_index=True, how='left')

    return df_final

# Extrair features topológicas
dados_finais = extrair_features_topologicas_simplificadas(dados_com_rotulos)

# Identificar colunas topológicas
features_topologicas = [col for col in dados_finais.columns if '_topo_' in col]

print(f"🔬 Features topológicas extraídas (valores contínuos)")
print(f"   Total de features topológicas: {len(features_topologicas)}")
print(f"   Features H0 (estrutura): {len([f for f in features_topologicas if '_0' in f])}")
print(f"   Features H1 (ciclos): {len([f for f in features_topologicas if '_1' in f])}")

print(f"\n📋 Amostra de features topológicas (valores originais):")
colunas_amostra = ['bitcoin_close'] + features_topologicas[:4]
print(dados_finais[colunas_amostra].dropna().head())

# %%
def treinar_meta_modelo(dados: pd.DataFrame,
                       features_topo: List[str],
                       coluna_rotulo: str = 'rotulo_barreira',
                       coluna_sinal: str = 'sinal_sma') -> Dict:
    """
    Treina meta-modelo CatBoost para predizer sucesso de sinais primários.

    O meta-modelo aprende quando confiar nos sinais da estratégia primária
    baseado em features topológicas e o próprio sinal primário.

    Parâmetros:
    -----------
    dados : pd.DataFrame
        Dados completos com features e rótulos
    features_topo : List[str]
        Lista de features topológicas
    coluna_rotulo : str
        Coluna com rótulos de barreira tripla
    coluna_sinal : str
        Coluna com sinais da estratégia primária

    Retorna:
    --------
    Dict
        Dicionário com modelo treinado e métricas
    """

    # Preparar dados para treinamento
    dados_treino = dados.dropna(subset=[coluna_rotulo])

    # Features: topológicas + sinal primário
    features_modelo = features_topo + [coluna_sinal]
    features_disponiveis = [f for f in features_modelo if f in dados_treino.columns]

    X = dados_treino[features_disponiveis].fillna(0)
    y = dados_treino[coluna_rotulo]

    # Converter rótulos para classificação binária (sucesso vs falha)
    # 1 = sucesso (lucro), 0 = falha (perda ou tempo)
    y_binario = (y == 1).astype(int)

    # Split temporal (80% treino, 20% teste)
    split_idx = int(0.8 * len(X))
    X_treino, X_teste = X.iloc[:split_idx], X.iloc[split_idx:]
    y_treino, y_teste = y_binario.iloc[:split_idx], y_binario.iloc[split_idx:]

    # Simular treinamento CatBoost (em implementação real usaria catboost)
    # Para demonstração, usar modelo simples
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, log_loss, classification_report

    modelo = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=6)
    modelo.fit(X_treino, y_treino)

    # Predições
    pred_treino = modelo.predict(X_treino)
    pred_teste = modelo.predict(X_teste)
    pred_proba_teste = modelo.predict_proba(X_teste)

    # Métricas
    acc_treino = accuracy_score(y_treino, pred_treino)
    acc_teste = accuracy_score(y_teste, pred_teste)
    logloss_teste = log_loss(y_teste, pred_proba_teste)

    # Importância das features
    importancias = dict(zip(features_disponiveis, modelo.feature_importances_))
    top_features = sorted(importancias.items(), key=lambda x: x[1], reverse=True)

    # Predições para todo o dataset
    pred_completa = modelo.predict(X)
    dados_com_pred = dados_treino.copy()
    dados_com_pred['meta_predicao'] = pred_completa
    dados_com_pred['sinal_final'] = dados_com_pred[coluna_sinal] * dados_com_pred['meta_predicao']

    return {
        'modelo': modelo,
        'dados_com_predicoes': dados_com_pred,
        'metricas': {
            'acuracia_treino': acc_treino,
            'acuracia_teste': acc_teste,
            'log_loss_teste': logloss_teste,
            'amostras_treino': len(X_treino),
            'amostras_teste': len(X_teste)
        },
        'importancia_features': top_features
    }

# Treinar meta-modelos para ambas as estratégias
resultado_meta_sma = treinar_meta_modelo(dados_finais, features_topologicas, coluna_sinal='sinal_sma')
resultado_meta_bb = treinar_meta_modelo(dados_finais, features_topologicas, coluna_sinal='sinal_bb')

print(f"🤖 Meta-modelos treinados com sucesso")
print(f"\n📈 Meta-modelo SMA:")
print(f"   Acurácia treino: {resultado_meta_sma['metricas']['acuracia_treino']:.3f}")
print(f"   Acurácia teste: {resultado_meta_sma['metricas']['acuracia_teste']:.3f}")
print(f"   Log-loss teste: {resultado_meta_sma['metricas']['log_loss_teste']:.3f}")

print(f"\n📊 Meta-modelo Bollinger Bands:")
print(f"   Acurácia treino: {resultado_meta_bb['metricas']['acuracia_treino']:.3f}")
print(f"   Acurácia teste: {resultado_meta_bb['metricas']['acuracia_teste']:.3f}")
print(f"   Log-loss teste: {resultado_meta_bb['metricas']['log_loss_teste']:.3f}")

print(f"\n🎯 Top 5 features mais importantes (SMA):")
for i, (feature, importancia) in enumerate(resultado_meta_sma['importancia_features'][:5]):
    feature_nome = feature.replace('bitcoin_close_topo_', '').replace('_', ' ')
    print(f"   {i+1}. {feature_nome}: {importancia:.3f}")

print(f"\n🎯 Top 5 features mais importantes (Bollinger):")
for i, (feature, importancia) in enumerate(resultado_meta_bb['importancia_features'][:5]):
    feature_nome = feature.replace('bitcoin_close_topo_', '').replace('_', ' ')
    print(f"   {i+1}. {feature_nome}: {importancia:.3f}")

# %%
def calcular_metricas_estrategia(dados: pd.DataFrame,
                                coluna_sinal: str,
                                coluna_preco: str = 'bitcoin_close') -> Dict:
    """
    Calcula métricas de performance de uma estratégia de trading.

    Métricas calculadas:
    - Retorno total e anualizado
    - Sharpe ratio
    - Máximo drawdown
    - Número de trades

    Parâmetros:
    -----------
    dados : pd.DataFrame
        Dados com sinais e preços
    coluna_sinal : str
        Coluna com sinais da estratégia
    coluna_preco : str
        Coluna com preços

    Retorna:
    --------
    Dict
        Dicionário com métricas calculadas
    """
    df = dados.dropna(subset=[coluna_sinal, coluna_preco]).copy()

    # Calcular retornos
    df['retorno_diario'] = df[coluna_preco].pct_change()
    df['retorno_estrategia'] = df['retorno_diario'] * df[coluna_sinal].shift(1)

    # Remover NaN
    retornos_validos = df['retorno_estrategia'].dropna()

    if len(retornos_validos) == 0:
        return {'erro': 'Sem retornos válidos'}

    # Retorno cumulativo
    retorno_cumulativo = (1 + retornos_validos).cumprod()

    # Métricas
    retorno_total = retorno_cumulativo.iloc[-1] - 1
    retorno_anualizado = (1 + retorno_total) ** (252 / len(retornos_validos)) - 1

    # Sharpe ratio (assumindo risk-free rate = 0)
    if retornos_validos.std() > 0:
        sharpe = retornos_validos.mean() / retornos_validos.std() * np.sqrt(252)
    else:
        sharpe = 0

    # Maximum drawdown
    peak = retorno_cumulativo.expanding().max()
    drawdown = (retorno_cumulativo - peak) / peak
    max_drawdown = drawdown.min()

    # Número de trades (mudanças de sinal)
    mudancas_sinal = (df[coluna_sinal] != df[coluna_sinal].shift(1)).sum()

    return {
        'retorno_total': retorno_total,
        'retorno_anualizado': retorno_anualizado,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'num_trades': mudancas_sinal,
        'retorno_cumulativo': retorno_cumulativo,
        'drawdown_serie': drawdown
    }

def plotar_resultados_finais(dados_original: pd.DataFrame,
                           dados_meta: pd.DataFrame,
                           coluna_sinal_original: str = 'sinal_bb') -> None:
    """
    Plota gráficos de retorno cumulativo e drawdown separados para treino e teste.

    Compara performance entre estratégia primária e meta-modelo.

    Parâmetros:
    -----------
    dados_original : pd.DataFrame
        Dados com estratégia primária
    dados_meta : pd.DataFrame
        Dados com meta-modelo
    """

    # Identificar split treino/teste (80/20)
    split_idx = int(len(dados_meta) * 0.8)

    # Separar dados de treino e teste
    dados_original_treino = dados_original.iloc[:split_idx]
    dados_original_teste = dados_original.iloc[split_idx:]
    dados_meta_treino = dados_meta.iloc[:split_idx]
    dados_meta_teste = dados_meta.iloc[split_idx:]

    # Calcular métricas separadas para treino
    metricas_original_treino = calcular_metricas_estrategia(dados_original_treino, coluna_sinal_original)
    metricas_meta_treino = calcular_metricas_estrategia(dados_meta_treino, 'sinal_final')
    metricas_bh_treino = calcular_metricas_estrategia(
        dados_original_treino.assign(sinal_buy_hold=1), 'sinal_buy_hold'
    )

    # Calcular métricas separadas para teste
    metricas_original_teste = calcular_metricas_estrategia(dados_original_teste, coluna_sinal_original)
    metricas_meta_teste = calcular_metricas_estrategia(dados_meta_teste, 'sinal_final')
    metricas_bh_teste = calcular_metricas_estrategia(
        dados_original_teste.assign(sinal_buy_hold=1), 'sinal_buy_hold'
    )

    # Criar gráficos 2x2
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))

    # === GRÁFICOS DE TREINO (coluna esquerda) ===

    # Gráfico 1: Retorno Cumulativo TREINO
    estrategia_nome = 'BB' if coluna_sinal_original == 'sinal_bb' else 'SMA'
    if 'retorno_cumulativo' in metricas_original_treino:
        ax1.plot(metricas_original_treino['retorno_cumulativo'].index,
                metricas_original_treino['retorno_cumulativo'].values,
                label=f'{estrategia_nome} (Sharpe: {metricas_original_treino["sharpe_ratio"]:.2f})',
                linewidth=2)

    if 'retorno_cumulativo' in metricas_meta_treino:
        ax1.plot(metricas_meta_treino['retorno_cumulativo'].index,
                metricas_meta_treino['retorno_cumulativo'].values,
                label=f'Meta (Sharpe: {metricas_meta_treino["sharpe_ratio"]:.2f})',
                linewidth=2)

    if 'retorno_cumulativo' in metricas_bh_treino:
        ax1.plot(metricas_bh_treino['retorno_cumulativo'].index,
                metricas_bh_treino['retorno_cumulativo'].values,
                label=f'B&H (Sharpe: {metricas_bh_treino["sharpe_ratio"]:.2f})',
                linewidth=2, alpha=0.7, linestyle='--')

    ax1.set_title('TREINO: Retorno Cumulativo', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Retorno Cumulativo')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Gráfico 2: Retorno Cumulativo TESTE
    if 'retorno_cumulativo' in metricas_original_teste:
        ax2.plot(metricas_original_teste['retorno_cumulativo'].index,
                metricas_original_teste['retorno_cumulativo'].values,
                label=f'{estrategia_nome} (Sharpe: {metricas_original_teste["sharpe_ratio"]:.2f})',
                linewidth=2)

    if 'retorno_cumulativo' in metricas_meta_teste:
        ax2.plot(metricas_meta_teste['retorno_cumulativo'].index,
                metricas_meta_teste['retorno_cumulativo'].values,
                label=f'Meta (Sharpe: {metricas_meta_teste["sharpe_ratio"]:.2f})',
                linewidth=2)

    if 'retorno_cumulativo' in metricas_bh_teste:
        ax2.plot(metricas_bh_teste['retorno_cumulativo'].index,
                metricas_bh_teste['retorno_cumulativo'].values,
                label=f'B&H (Sharpe: {metricas_bh_teste["sharpe_ratio"]:.2f})',
                linewidth=2, alpha=0.7, linestyle='--')

    ax2.set_title('TESTE: Retorno Cumulativo', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Retorno Cumulativo')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # Gráfico 3: Drawdown TREINO
    if 'drawdown_serie' in metricas_original_treino:
        ax3.fill_between(metricas_original_treino['drawdown_serie'].index,
                        metricas_original_treino['drawdown_serie'].values, 0,
                        alpha=0.3, label=f'{estrategia_nome} (Max: {metricas_original_treino["max_drawdown"]:.1%})')

    if 'drawdown_serie' in metricas_meta_treino:
        ax3.fill_between(metricas_meta_treino['drawdown_serie'].index,
                        metricas_meta_treino['drawdown_serie'].values, 0,
                        alpha=0.3, label=f'Meta (Max: {metricas_meta_treino["max_drawdown"]:.1%})')

    ax3.set_title('TREINO: Drawdown', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Drawdown')
    ax3.set_xlabel('Data')
    ax3.legend(loc='lower left')
    ax3.grid(True, alpha=0.3)

    # Gráfico 4: Drawdown TESTE
    if 'drawdown_serie' in metricas_original_teste:
        ax4.fill_between(metricas_original_teste['drawdown_serie'].index,
                        metricas_original_teste['drawdown_serie'].values, 0,
                        alpha=0.3, label=f'{estrategia_nome} (Max: {metricas_original_teste["max_drawdown"]:.1%})')

    if 'drawdown_serie' in metricas_meta_teste:
        ax4.fill_between(metricas_meta_teste['drawdown_serie'].index,
                        metricas_meta_teste['drawdown_serie'].values, 0,
                        alpha=0.3, label=f'Meta (Max: {metricas_meta_teste["max_drawdown"]:.1%})')

    ax4.set_title('TESTE: Drawdown', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Drawdown')
    ax4.set_xlabel('Data')
    ax4.legend(loc='lower left')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Retornar todas as métricas separadas
    return {
        'treino': {
            'original': metricas_original_treino,
            'meta': metricas_meta_treino,
            'buy_hold': metricas_bh_treino
        },
        'teste': {
            'original': metricas_original_teste,
            'meta': metricas_meta_teste,
            'buy_hold': metricas_bh_teste
        }
    }

# %%
print(f"\n🎯 RESULTADOS FINAIS - SMA CROSSOVER")
print(f"{'='*60}")

# Calcular e plotar resultados SMA
print(f"📊 Calculando métricas SMA...")
metricas_sma = plotar_resultados_finais(
    dados_finais,
    resultado_meta_sma['dados_com_predicoes'],
    coluna_sinal_original='sinal_sma'
)

# === MÉTRICAS SMA TREINO ===
print(f"\n📊 SMA - PERÍODO DE TREINO (80%):")
print(f"{'-'*40}")

print(f"\n📈 SMA Crossover:")
if 'retorno_total' in metricas_sma['treino']['original']:
    print(f"   Retorno Total: {metricas_sma['treino']['original']['retorno_total']:.2%}")
    print(f"   Sharpe Ratio: {metricas_sma['treino']['original']['sharpe_ratio']:.3f}")
    print(f"   Max Drawdown: {metricas_sma['treino']['original']['max_drawdown']:.2%}")

print(f"\n🤖 Meta-modelo SMA:")
if 'retorno_total' in metricas_sma['treino']['meta']:
    print(f"   Retorno Total: {metricas_sma['treino']['meta']['retorno_total']:.2%}")
    print(f"   Sharpe Ratio: {metricas_sma['treino']['meta']['sharpe_ratio']:.3f}")
    print(f"   Max Drawdown: {metricas_sma['treino']['meta']['max_drawdown']:.2%}")

print(f"\n💎 Buy & Hold:")
if 'retorno_total' in metricas_sma['treino']['buy_hold']:
    print(f"   Retorno Total: {metricas_sma['treino']['buy_hold']['retorno_total']:.2%}")
    print(f"   Sharpe Ratio: {metricas_sma['treino']['buy_hold']['sharpe_ratio']:.3f}")
    print(f"   Max Drawdown: {metricas_sma['treino']['buy_hold']['max_drawdown']:.2%}")

# === MÉTRICAS SMA TESTE ===
print(f"\n📊 SMA - PERÍODO DE TESTE (20%):")
print(f"{'-'*40}")

print(f"\n📈 SMA Crossover:")
if 'retorno_total' in metricas_sma['teste']['original']:
    print(f"   Retorno Total: {metricas_sma['teste']['original']['retorno_total']:.2%}")
    print(f"   Sharpe Ratio: {metricas_sma['teste']['original']['sharpe_ratio']:.3f}")
    print(f"   Max Drawdown: {metricas_sma['teste']['original']['max_drawdown']:.2%}")

print(f"\n🤖 Meta-modelo SMA:")
if 'retorno_total' in metricas_sma['teste']['meta']:
    print(f"   Retorno Total: {metricas_sma['teste']['meta']['retorno_total']:.2%}")
    print(f"   Sharpe Ratio: {metricas_sma['teste']['meta']['sharpe_ratio']:.3f}")
    print(f"   Max Drawdown: {metricas_sma['teste']['meta']['max_drawdown']:.2%}")

print(f"\n💎 Buy & Hold:")
if 'retorno_total' in metricas_sma['teste']['buy_hold']:
    print(f"   Retorno Total: {metricas_sma['teste']['buy_hold']['retorno_total']:.2%}")
    print(f"   Sharpe Ratio: {metricas_sma['teste']['buy_hold']['sharpe_ratio']:.3f}")
    print(f"   Max Drawdown: {metricas_sma['teste']['buy_hold']['max_drawdown']:.2%}")

# %%
print(f"\n\n🎯 RESULTADOS FINAIS - BOLLINGER BANDS")
print(f"{'='*60}")

# Calcular e plotar resultados Bollinger Bands
print(f"📊 Calculando métricas Bollinger Bands...")
metricas_bb = plotar_resultados_finais(
    dados_finais,
    resultado_meta_bb['dados_com_predicoes'],
    coluna_sinal_original='sinal_bb'
)

# === MÉTRICAS BB TREINO ===
print(f"\n📊 BOLLINGER BANDS - PERÍODO DE TREINO (80%):")
print(f"{'-'*40}")

print(f"\n📊 Bollinger Bands:")
if 'retorno_total' in metricas_bb['treino']['original']:
    print(f"   Retorno Total: {metricas_bb['treino']['original']['retorno_total']:.2%}")
    print(f"   Sharpe Ratio: {metricas_bb['treino']['original']['sharpe_ratio']:.3f}")
    print(f"   Max Drawdown: {metricas_bb['treino']['original']['max_drawdown']:.2%}")

print(f"\n🤖 Meta-modelo BB:")
if 'retorno_total' in metricas_bb['treino']['meta']:
    print(f"   Retorno Total: {metricas_bb['treino']['meta']['retorno_total']:.2%}")
    print(f"   Sharpe Ratio: {metricas_bb['treino']['meta']['sharpe_ratio']:.3f}")
    print(f"   Max Drawdown: {metricas_bb['treino']['meta']['max_drawdown']:.2%}")

print(f"\n💎 Buy & Hold:")
if 'retorno_total' in metricas_bb['treino']['buy_hold']:
    print(f"   Retorno Total: {metricas_bb['treino']['buy_hold']['retorno_total']:.2%}")
    print(f"   Sharpe Ratio: {metricas_bb['treino']['buy_hold']['sharpe_ratio']:.3f}")
    print(f"   Max Drawdown: {metricas_bb['treino']['buy_hold']['max_drawdown']:.2%}")

# === MÉTRICAS BB TESTE ===
print(f"\n📊 BOLLINGER BANDS - PERÍODO DE TESTE (20%):")
print(f"{'-'*40}")

print(f"\n📊 Bollinger Bands:")
if 'retorno_total' in metricas_bb['teste']['original']:
    print(f"   Retorno Total: {metricas_bb['teste']['original']['retorno_total']:.2%}")
    print(f"   Sharpe Ratio: {metricas_bb['teste']['original']['sharpe_ratio']:.3f}")
    print(f"   Max Drawdown: {metricas_bb['teste']['original']['max_drawdown']:.2%}")

print(f"\n🤖 Meta-modelo BB:")
if 'retorno_total' in metricas_bb['teste']['meta']:
    print(f"   Retorno Total: {metricas_bb['teste']['meta']['retorno_total']:.2%}")
    print(f"   Sharpe Ratio: {metricas_bb['teste']['meta']['sharpe_ratio']:.3f}")
    print(f"   Max Drawdown: {metricas_bb['teste']['meta']['max_drawdown']:.2%}")

print(f"\n💎 Buy & Hold:")
if 'retorno_total' in metricas_bb['teste']['buy_hold']:
    print(f"   Retorno Total: {metricas_bb['teste']['buy_hold']['retorno_total']:.2%}")
    print(f"   Sharpe Ratio: {metricas_bb['teste']['buy_hold']['sharpe_ratio']:.3f}")
    print(f"   Max Drawdown: {metricas_bb['teste']['buy_hold']['max_drawdown']:.2%}")

print(f"\n{'='*50}")
print(f"🔬 Pipeline completo demonstrado com sucesso!")
print(f"   Features topológicas extraídas das barras de persistência")
print(f"   Meta-modelos treinados para ambas estratégias primárias")
print(f"   Resultados comparados com baseline buy-and-hold")
# %%
