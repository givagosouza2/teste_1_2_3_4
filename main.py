import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from scipy import signal
from sklearn.cluster import KMeans


# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================

st.set_page_config(
    page_title="Segmentação automática por K-means",
    layout="wide"
)

st.title("Segmentação automática do movimento por K-means")

st.caption(
    "Pré-processamento: limpeza → interpolação → filtro → "
    "distância vetorial em relação à baseline → K-means → segmentação."
)


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def detectar_separador(arquivo_bytes):
    """
    Tenta identificar automaticamente o separador do arquivo.
    """
    texto = arquivo_bytes.decode("utf-8", errors="ignore")

    primeira_linha = texto.splitlines()[0]

    candidatos = {
        ",": primeira_linha.count(","),
        ";": primeira_linha.count(";"),
        "\t": primeira_linha.count("\t")
    }

    return max(candidatos, key=candidatos.get)


def preparar_dados(
    df,
    time_col,
    x_col,
    y_col,
    z_col,
    fs=100.0,
    cutoff=10.0,
    filter_order=4,
    aplicar_detrend=False
):
    """
    Etapas:

    1. Seleciona tempo e eixos XYZ
    2. Converte para números
    3. Remove NaN / Inf
    4. Ordena por tempo
    5. Remove tempos duplicados
    6. Opcionalmente aplica detrend
    7. Interpola para frequência uniforme
    8. Aplica filtro passa-baixa Butterworth
    """

    dados = df[
        [time_col, x_col, y_col, z_col]
    ].copy()

    dados.columns = [
        "Tempo",
        "X",
        "Y",
        "Z"
    ]

    # --------------------------------------------------------
    # Conversão numérica
    # --------------------------------------------------------

    for coluna in dados.columns:
        dados[coluna] = pd.to_numeric(
            dados[coluna],
            errors="coerce"
        )

    dados = dados.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if len(dados) < 10:
        raise ValueError(
            "O arquivo possui poucos dados válidos."
        )

    # --------------------------------------------------------
    # Ordenação temporal
    # --------------------------------------------------------

    dados = dados.sort_values("Tempo")

    dados = dados.drop_duplicates(
        subset="Tempo",
        keep="first"
    )

    tempo_original = dados["Tempo"].to_numpy(
        dtype=float
    )

    xyz_original = dados[
        ["X", "Y", "Z"]
    ].to_numpy(dtype=float)

    # --------------------------------------------------------
    # Detecta se tempo está em ms ou segundos
    #
    # Critério simples:
    # se o intervalo mediano > 1, provavelmente está em ms
    # --------------------------------------------------------

    delta_mediano = np.median(
        np.diff(tempo_original)
    )

    if delta_mediano > 1:
        tempo_segundos = (
            tempo_original / 1000.0
        )
    else:
        tempo_segundos = tempo_original

    if np.any(
        np.diff(tempo_segundos) <= 0
    ):
        raise ValueError(
            "A coluna de tempo precisa ser crescente."
        )

    # --------------------------------------------------------
    # Frequência original
    # --------------------------------------------------------

    dt_mediano = np.median(
        np.diff(tempo_segundos)
    )

    fs_original = (
        1.0 / dt_mediano
    )

    # --------------------------------------------------------
    # Detrend opcional
    # --------------------------------------------------------

    if aplicar_detrend:

        xyz_pre = signal.detrend(
            xyz_original,
            axis=0,
            type="linear"
        )

    else:

        xyz_pre = xyz_original.copy()

    # --------------------------------------------------------
    # Interpolação
    # --------------------------------------------------------

    dt_novo = 1.0 / fs

    tempo_uniforme = np.arange(
        tempo_segundos[0],
        tempo_segundos[-1]
        + dt_novo / 2,
        dt_novo
    )

    xyz_interp = np.column_stack([
        np.interp(
            tempo_uniforme,
            tempo_segundos,
            xyz_pre[:, eixo]
        )
        for eixo in range(3)
    ])

    # --------------------------------------------------------
    # Filtro passa-baixa Butterworth
    # --------------------------------------------------------

    nyquist = fs / 2.0

    if cutoff <= 0 or cutoff >= nyquist:

        raise ValueError(
            f"A frequência de corte deve estar entre "
            f"0 e {nyquist:.1f} Hz."
        )

    sos = signal.butter(
        filter_order,
        cutoff / nyquist,
        btype="lowpass",
        output="sos"
    )

    xyz_filtrado = signal.sosfiltfilt(
        sos,
        xyz_interp,
        axis=0
    )

    proc = pd.DataFrame({
        "Tempo_s": tempo_uniforme,
        "Tempo_rel_s":
            tempo_uniforme
            - tempo_uniforme[0],

        "X_filtrado":
            xyz_filtrado[:, 0],

        "Y_filtrado":
            xyz_filtrado[:, 1],

        "Z_filtrado":
            xyz_filtrado[:, 2]
    })

    return proc, fs_original


# ============================================================
# CÁLCULO DA DISTÂNCIA À BASELINE
# ============================================================

def calcular_distancia_baseline(
    proc,
    baseline_segundos
):
    """
    Calcula o vetor médio da baseline:

        [Xb, Yb, Zb]

    e depois a distância Euclidiana entre
    cada amostra e esse vetor:

        D = sqrt(
            (X-Xb)^2 +
            (Y-Yb)^2 +
            (Z-Zb)^2
        )
    """

    tempo = proc[
        "Tempo_rel_s"
    ].to_numpy()

    mask = (
        tempo <= baseline_segundos
    )

    if mask.sum() < 5:

        raise ValueError(
            "Poucas amostras na baseline."
        )

    xyz = proc[
        [
            "X_filtrado",
            "Y_filtrado",
            "Z_filtrado"
        ]
    ].to_numpy()

    vetor_baseline = np.mean(
        xyz[mask],
        axis=0
    )

    diferenca = (
        xyz - vetor_baseline
    )

    distancia = np.sqrt(
        np.sum(
            diferenca ** 2,
            axis=1
        )
    )

    resultado = proc.copy()

    resultado[
        "Distancia_baseline"
    ] = distancia

    return (
        resultado,
        vetor_baseline
    )


# ============================================================
# K-MEANS
# ============================================================

def aplicar_kmeans(
    proc,
    n_clusters=5,
    random_state=42
):
    """
    Aplica K-means à distância em relação à baseline.

    Os clusters são reordenados pelos centróides:

        Estado 0 = menor distância da baseline
        Estado N = maior distância da baseline
    """

    sinal = proc[
        "Distancia_baseline"
    ].to_numpy().reshape(
        -1,
        1
    )

    km = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=20
    )

    labels_originais = (
        km.fit_predict(sinal)
    )

    centroides_originais = (
        km.cluster_centers_
        .ravel()
    )

    ordem = np.argsort(
        centroides_originais
    )

    mapa_estado = {
        cluster_original:
            estado_ordenado

        for estado_ordenado,
            cluster_original
        in enumerate(ordem)
    }

    estados = np.array([
        mapa_estado[label]
        for label
        in labels_originais
    ])

    centroides_ordenados = (
        centroides_originais[
            ordem
        ]
    )

    resultado = proc.copy()

    resultado[
        "Estado"
    ] = estados

    return (
        resultado,
        centroides_ordenados
    )


# ============================================================
# IDENTIFICAÇÃO DO ESTADO DA BASELINE
# ============================================================

def obter_estado_baseline(
    proc,
    baseline_segundos
):

    tempo = proc[
        "Tempo_rel_s"
    ].to_numpy()

    estados = proc[
        "Estado"
    ].to_numpy()

    mask = (
        tempo <= baseline_segundos
    )

    valores, contagens = (
        np.unique(
            estados[mask],
            return_counts=True
        )
    )

    estado_baseline = int(
        valores[
            np.argmax(contagens)
        ]
    )

    return estado_baseline


# ============================================================
# DETECÇÃO DE INÍCIO E FIM
# ============================================================

def detectar_segmento(
    proc,
    baseline_segundos=1.0,
    n_inicio=15,
    n_fim=75,
    tolerancia_baseline=0
):
    """
    INÍCIO
    ------

    Primeiro momento após a janela inicial de baseline
    em que aparecem n_inicio amostras consecutivas
    fora da faixa de baseline.

    FIM
    ---

    Primeiro retorno sustentado à faixa da baseline
    durante n_fim amostras consecutivas.

    tolerancia_baseline:

        0 → somente estado da baseline

        1 → aceita estado_baseline ± 1

    """

    tempo = proc[
        "Tempo_rel_s"
    ].to_numpy()

    estados = proc[
        "Estado"
    ].to_numpy()

    estado_baseline = (
        obter_estado_baseline(
            proc,
            baseline_segundos
        )
    )

    idx_busca = int(
        np.searchsorted(
            tempo,
            baseline_segundos,
            side="right"
        )
    )

    # --------------------------------------------------------
    # Define faixa de estados considerada baseline
    # --------------------------------------------------------

    estado_min = max(
        0,
        estado_baseline
        - tolerancia_baseline
    )

    estado_max = min(
        estados.max(),
        estado_baseline
        + tolerancia_baseline
    )

    dentro_baseline = (
        (estados >= estado_min)
        &
        (estados <= estado_max)
    )

    # --------------------------------------------------------
    # INÍCIO
    # --------------------------------------------------------

    idx_inicio = None

    for i in range(
        idx_busca,
        len(estados)
        - n_inicio + 1
    ):

        janela = (
            dentro_baseline[
                i:i+n_inicio
            ]
        )

        # Toda a janela está FORA da baseline
        if np.all(~janela):

            idx_inicio = i
            break

    # --------------------------------------------------------
    # FIM
    # --------------------------------------------------------

    idx_fim = None

    if idx_inicio is not None:

        primeiro_idx_fim = (
            idx_inicio
            + n_inicio
        )

        for i in range(
            primeiro_idx_fim,
            len(estados)
            - n_fim + 1
        ):

            janela = (
                dentro_baseline[
                    i:i+n_fim
                ]
            )

            # Toda a janela voltou à baseline
            if np.all(janela):

                idx_fim = i
                break

    return (
        estado_baseline,
        idx_inicio,
        idx_fim
    )


# ============================================================
# CSV
# ============================================================

def csv_bytes(df):

    return df.to_csv(
        index=False
    ).encode(
        "utf-8-sig"
    )


# ============================================================
# BARRA LATERAL
# ============================================================

st.sidebar.header(
    "Parâmetros"
)


fs = st.sidebar.number_input(
    "Frequência após interpolação (Hz)",
    min_value=20.0,
    max_value=500.0,
    value=100.0,
    step=10.0
)


cutoff = st.sidebar.number_input(
    "Filtro passa-baixa (Hz)",
    min_value=0.5,
    max_value=49.0,
    value=10.0,
    step=0.5
)


filter_order = st.sidebar.number_input(
    "Ordem do filtro",
    min_value=1,
    max_value=10,
    value=4,
    step=1
)


aplicar_detrend = st.sidebar.checkbox(
    "Aplicar detrend linear",
    value=False
)


st.sidebar.divider()


n_clusters = st.sidebar.number_input(
    "Número de estados do K-means",
    min_value=2,
    max_value=10,
    value=5,
    step=1
)


baseline_segundos = (
    st.sidebar.number_input(
        "Duração da baseline inicial (s)",
        min_value=0.2,
        max_value=5.0,
        value=1.0,
        step=0.1
    )
)


n_inicio = st.sidebar.number_input(
    "Amostras consecutivas para início",
    min_value=1,
    max_value=200,
    value=15,
    step=1
)


n_fim = st.sidebar.number_input(
    "Amostras consecutivas para fim",
    min_value=5,
    max_value=500,
    value=75,
    step=5
)


tolerancia_baseline = (
    st.sidebar.number_input(
        "Tolerância de estados na baseline",
        min_value=0,
        max_value=2,
        value=0,
        step=1
    )
)


st.sidebar.caption(
    f"Início mínimo = "
    f"{n_inicio/fs:.2f} s"
)

st.sidebar.caption(
    f"Retorno mínimo para fim = "
    f"{n_fim/fs:.2f} s"
)


# ============================================================
# UPLOAD
# ============================================================

arquivo = st.file_uploader(
    "Selecione o arquivo CSV",
    type=[
        "csv",
        "txt"
    ]
)


if arquivo is not None:

    try:

        # ----------------------------------------------------
        # Leitura do arquivo
        # ----------------------------------------------------

        conteudo = (
            arquivo.getvalue()
        )

        separador = (
            detectar_separador(
                conteudo
            )
        )

        df = pd.read_csv(
            io.BytesIO(conteudo),
            sep=separador
        )

        st.subheader(
            "Arquivo importado"
        )

        st.write(
            f"Separador detectado: "
            f"`{repr(separador)}`"
        )

        st.dataframe(
            df.head(),
            use_container_width=True
        )

        colunas = list(
            df.columns
        )

        # ----------------------------------------------------
        # Seleção das colunas
        # ----------------------------------------------------

        c1, c2, c3, c4 = (
            st.columns(4)
        )

        with c1:

            time_col = st.selectbox(
                "Tempo",
                colunas,
                index=(
                    colunas.index(
                        "TempoMs"
                    )
                    if "TempoMs"
                    in colunas
                    else 0
                )
            )

        with c2:

            x_col = st.selectbox(
                "Eixo X",
                colunas,
                index=(
                    colunas.index("X")
                    if "X" in colunas
                    else 0
                )
            )

        with c3:

            y_col = st.selectbox(
                "Eixo Y",
                colunas,
                index=(
                    colunas.index("Y")
                    if "Y" in colunas
                    else min(
                        1,
                        len(colunas)-1
                    )
                )
            )

        with c4:

            z_col = st.selectbox(
                "Eixo Z",
                colunas,
                index=(
                    colunas.index("Z")
                    if "Z" in colunas
                    else min(
                        2,
                        len(colunas)-1
                    )
                )
            )

        # ----------------------------------------------------
        # PROCESSAMENTO
        # ----------------------------------------------------

        proc, fs_original = (
            preparar_dados(
                df=df,
                time_col=time_col,
                x_col=x_col,
                y_col=y_col,
                z_col=z_col,
                fs=float(fs),
                cutoff=float(cutoff),
                filter_order=int(
                    filter_order
                ),
                aplicar_detrend=(
                    aplicar_detrend
                )
            )
        )

        # ----------------------------------------------------
        # Distância vetorial
        # ----------------------------------------------------

        proc, vetor_baseline = (
            calcular_distancia_baseline(
                proc,
                baseline_segundos=float(
                    baseline_segundos
                )
            )
        )

        # ----------------------------------------------------
        # K-means
        # ----------------------------------------------------

        proc, centroides = (
            aplicar_kmeans(
                proc,
                n_clusters=int(
                    n_clusters
                )
            )
        )

        # ----------------------------------------------------
        # Segmentação
        # ----------------------------------------------------

        (
            estado_baseline,
            idx_inicio,
            idx_fim
        ) = detectar_segmento(

            proc,

            baseline_segundos=float(
                baseline_segundos
            ),

            n_inicio=int(
                n_inicio
            ),

            n_fim=int(
                n_fim
            ),

            tolerancia_baseline=int(
                tolerancia_baseline
            )
        )

        # ====================================================
        # RESUMO
        # ====================================================

        st.subheader(
            "Resultado da segmentação"
        )

        m1, m2, m3, m4 = (
            st.columns(4)
        )

        m1.metric(
            "Frequência original estimada",
            f"{fs_original:.1f} Hz"
        )

        m2.metric(
            "Estado dominante da baseline",
            str(
                estado_baseline
            )
        )

        # ----------------------------------------------------
        # Início
        # ----------------------------------------------------

        if idx_inicio is not None:

            tempo_inicio = (
                proc.loc[
                    idx_inicio,
                    "Tempo_rel_s"
                ]
            )

            m3.metric(
                "Início detectado",
                f"{tempo_inicio:.3f} s"
            )

        else:

            tempo_inicio = None

            m3.metric(
                "Início detectado",
                "Não encontrado"
            )

        # ----------------------------------------------------
        # Fim
        # ----------------------------------------------------

        if idx_fim is not None:

            tempo_fim = (
                proc.loc[
                    idx_fim,
                    "Tempo_rel_s"
                ]
            )

            m4.metric(
                "Fim detectado",
                f"{tempo_fim:.3f} s"
            )

        else:

            tempo_fim = None

            m4.metric(
                "Fim detectado",
                "Não encontrado"
            )

        # ----------------------------------------------------
        # Duração
        # ----------------------------------------------------

        if (
            tempo_inicio is not None
            and
            tempo_fim is not None
        ):

            duracao = (
                tempo_fim
                - tempo_inicio
            )

            st.success(
                f"Segmento detectado: "
                f"{tempo_inicio:.3f} s → "
                f"{tempo_fim:.3f} s "
                f"| duração = "
                f"{duracao:.3f} s"
            )

        # ====================================================
        # VETOR DA BASELINE
        # ====================================================

        st.subheader(
            "Vetor médio da baseline"
        )

        base_df = pd.DataFrame({
            "Eixo": [
                "X",
                "Y",
                "Z"
            ],
            "Valor médio": (
                vetor_baseline
            )
        })

        st.dataframe(
            base_df,
            use_container_width=True,
            hide_index=True
        )

        # ====================================================
        # CENTRÓIDES
        # ====================================================

        st.subheader(
            "Estados do K-means"
        )

        cent_df = pd.DataFrame({
            "Estado":
                np.arange(
                    len(centroides)
                ),

            "Centroide_distancia":
                centroides
        })

        st.dataframe(
            cent_df,
            use_container_width=True,
            hide_index=True
        )

        # ====================================================
        # GRÁFICO 1
        # XYZ
        # ====================================================

        st.subheader(
            "Aceleração filtrada"
        )

        fig1, ax1 = plt.subplots(
            figsize=(13, 4)
        )

        ax1.plot(
            proc["Tempo_rel_s"],
            proc["X_filtrado"],
            label="X"
        )

        ax1.plot(
            proc["Tempo_rel_s"],
            proc["Y_filtrado"],
            label="Y"
        )

        ax1.plot(
            proc["Tempo_rel_s"],
            proc["Z_filtrado"],
            label="Z"
        )

        ax1.axvspan(
            0,
            baseline_segundos,
            alpha=0.15,
            label="Baseline"
        )

        if tempo_inicio is not None:

            ax1.axvline(
                tempo_inicio,
                linestyle="--",
                linewidth=2,
                label="Início"
            )

        if tempo_fim is not None:

            ax1.axvline(
                tempo_fim,
                linestyle="--",
                linewidth=2,
                label="Fim"
            )

        ax1.set_xlabel(
            "Tempo (s)"
        )

        ax1.set_ylabel(
            "Aceleração"
        )

        ax1.grid(
            alpha=0.2
        )

        ax1.legend()

        st.pyplot(
            fig1
        )

        plt.close(
            fig1
        )

        # ====================================================
        # GRÁFICO 2
        # DISTÂNCIA DA BASELINE
        # ====================================================

        st.subheader(
            "Distância vetorial em relação à baseline"
        )

        fig2, ax2 = plt.subplots(
            figsize=(13, 4)
        )

        ax2.plot(
            proc["Tempo_rel_s"],
            proc[
                "Distancia_baseline"
            ],
            linewidth=1.2
        )

        ax2.axvspan(
            0,
            baseline_segundos,
            alpha=0.15,
            label="Baseline"
        )

        if tempo_inicio is not None:

            ax2.axvline(
                tempo_inicio,
                linestyle="--",
                linewidth=2,
                label="Início"
            )

        if tempo_fim is not None:

            ax2.axvline(
                tempo_fim,
                linestyle="--",
                linewidth=2,
                label="Fim"
            )

        ax2.set_xlabel(
            "Tempo (s)"
        )

        ax2.set_ylabel(
            "Distância da baseline"
        )

        ax2.grid(
            alpha=0.2
        )

        ax2.legend()

        st.pyplot(
            fig2
        )

        plt.close(
            fig2
        )

        # ====================================================
        # GRÁFICO 3
        # ESTADOS
        # ====================================================

        st.subheader(
            "Sequência de estados do K-means"
        )

        fig3, ax3 = plt.subplots(
            figsize=(13, 3)
        )

        ax3.step(
            proc["Tempo_rel_s"],
            proc["Estado"],
            where="post",
            linewidth=1
        )

        ax3.axhline(
            estado_baseline,
            linestyle=":",
            linewidth=2,
            label=(
                f"Estado baseline = "
                f"{estado_baseline}"
            )
        )

        ax3.axvspan(
            0,
            baseline_segundos,
            alpha=0.15
        )

        if tempo_inicio is not None:

            ax3.axvline(
                tempo_inicio,
                linestyle="--",
                linewidth=2,
                label="Início"
            )

        if tempo_fim is not None:

            ax3.axvline(
                tempo_fim,
                linestyle="--",
                linewidth=2,
                label="Fim"
            )

        ax3.set_xlabel(
            "Tempo (s)"
        )

        ax3.set_ylabel(
            "Estado"
        )

        ax3.set_yticks(
            range(
                int(n_clusters)
            )
        )

        ax3.grid(
            alpha=0.2
        )

        ax3.legend()

        st.pyplot(
            fig3
        )

        plt.close(
            fig3
        )

        # ====================================================
        # SEGMENTO
        # ====================================================

        if (
            idx_inicio is not None
            and
            idx_fim is not None
        ):

            segmento = (
                proc.iloc[
                    idx_inicio:
                    idx_fim + 1
                ]
                .copy()
            )

            segmento[
                "Tempo_segmento_s"
            ] = (
                segmento[
                    "Tempo_rel_s"
                ]
                -
                segmento[
                    "Tempo_rel_s"
                ].iloc[0]
            )

            st.subheader(
                "Dados segmentados"
            )

            st.dataframe(
                segmento.head(200),
                use_container_width=True
            )

            st.download_button(
                "Baixar segmento CSV",
                data=csv_bytes(
                    segmento
                ),
                file_name=(
                    "segmento_detectado.csv"
                ),
                mime="text/csv"
            )

        # ====================================================
        # DOWNLOAD COMPLETO
        # ====================================================

        st.download_button(
            "Baixar sinal processado completo",
            data=csv_bytes(
                proc
            ),
            file_name=(
                "sinal_processado_com_estados.csv"
            ),
            mime="text/csv"
        )


    except Exception as e:

        st.error(
            f"Erro durante o processamento: {e}"
        )
