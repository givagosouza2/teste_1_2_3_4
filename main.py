import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from scipy import signal
from sklearn.cluster import KMeans


# ============================================================
# Configuração
# ============================================================
st.set_page_config(
    page_title="Segmentação por K-means",
    layout="wide"
)

st.title("Segmentação automática do teste por K-means")
st.caption(
    "Pré-processamento: detrend → interpolação para 100 Hz → filtro passa-baixa → "
    "magnitude da aceleração → K-means com estados ordenados por amplitude."
)


# ============================================================
# Funções
# ============================================================
def preparar_dados(df, time_col, x_col, y_col, z_col, fs=100.0, cutoff=10.0, filter_order=4):
    """
    1. Converte tempo de ms para s
    2. Remove NaN/Inf e tempos duplicados
    3. Detrend linear em X, Y e Z
    4. Interpola para frequência uniforme
    5. Aplica filtro Butterworth passa-baixa
    6. Calcula magnitude da aceleração filtrada
    """

    dados = df[[time_col, x_col, y_col, z_col]].copy()
    dados.columns = ["TempoMs", "X", "Y", "Z"]

    for c in dados.columns:
        dados[c] = pd.to_numeric(dados[c], errors="coerce")

    dados = dados.replace([np.inf, -np.inf], np.nan).dropna()

    if len(dados) < 10:
        raise ValueError("O arquivo possui poucos dados válidos.")

    # Ordena por tempo
    dados = dados.sort_values("TempoMs")

    # Remove tempos duplicados
    dados = dados.drop_duplicates(subset="TempoMs", keep="first")

    tempo = dados["TempoMs"].to_numpy(dtype=float) / 1000.0
    xyz = dados[["X", "Y", "Z"]].to_numpy(dtype=float)

    if len(tempo) < 10:
        raise ValueError("Não há amostras suficientes após a limpeza.")

    if np.any(np.diff(tempo) <= 0):
        raise ValueError("A coluna de tempo precisa ser estritamente crescente.")

    # Frequência original aproximada
    dt_mediano = np.median(np.diff(tempo))
    fs_original = 1.0 / dt_mediano

    # --------------------------------------------------------
    # 1) Detrend linear
    # --------------------------------------------------------
    xyz_detrend = signal.detrend(xyz, axis=0, type="linear")

    # --------------------------------------------------------
    # 2) Interpolação para frequência uniforme
    # --------------------------------------------------------
    dt_novo = 1.0 / fs
    tempo_uniforme = np.arange(
        tempo[0],
        tempo[-1] + dt_novo / 2,
        dt_novo
    )

    xyz_interp = np.column_stack([
        np.interp(tempo_uniforme, tempo, xyz_detrend[:, eixo])
        for eixo in range(3)
    ])

    # --------------------------------------------------------
    # 3) Filtro passa-baixa Butterworth
    # --------------------------------------------------------
    nyquist = fs / 2.0

    if cutoff <= 0 or cutoff >= nyquist:
        raise ValueError(
            f"A frequência de corte deve estar entre 0 e {nyquist:.1f} Hz."
        )

    sos = signal.butter(
        filter_order,
        cutoff / nyquist,
        btype="lowpass",
        output="sos"
    )

    xyz_filtrado = signal.sosfiltfilt(sos, xyz_interp, axis=0)

    # Magnitude da aceleração
    magnitude = np.sqrt(np.sum(xyz_filtrado ** 2, axis=1))

    proc = pd.DataFrame({
        "Tempo_s": tempo_uniforme,
        "Tempo_rel_s": tempo_uniforme - tempo_uniforme[0],
        "X_detrend_filtrado": xyz_filtrado[:, 0],
        "Y_detrend_filtrado": xyz_filtrado[:, 1],
        "Z_detrend_filtrado": xyz_filtrado[:, 2],
        "Magnitude": magnitude
    })

    return proc, fs_original


def aplicar_kmeans(proc, n_clusters=5, random_state=42):
    """
    Executa K-means sobre a magnitude.

    IMPORTANTE:
    Os rótulos do K-means são arbitrários.
    Por isso, os clusters são reordenados pelos centróides:

        Estado 0 = menor amplitude
        ...
        Estado 4 = maior amplitude

    Isso permite interpretar corretamente "estados superiores".
    """

    sinal = proc["Magnitude"].to_numpy().reshape(-1, 1)

    km = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=20
    )

    labels_originais = km.fit_predict(sinal)
    centroides_originais = km.cluster_centers_.ravel()

    ordem = np.argsort(centroides_originais)

    mapa_estado = {
        cluster_original: estado_ordenado
        for estado_ordenado, cluster_original in enumerate(ordem)
    }

    estados = np.array([
        mapa_estado[label]
        for label in labels_originais
    ], dtype=int)

    centroides_ordenados = centroides_originais[ordem]

    resultado = proc.copy()
    resultado["Estado"] = estados

    return resultado, centroides_ordenados


def detectar_segmento(
    proc,
    baseline_segundos=2.0,
    n_inicio=5,
    n_fim=10
):
    """
    Baseline:
        Estado dominante nos primeiros 'baseline_segundos'.

    Início:
        Primeiro instante após a baseline em que:
        - o estado deixa de ser o estado dominante da baseline
        - e existe uma sequência de n_inicio amostras consecutivas
          em estados SUPERIORES ao estado da baseline.

    Fim:
        Primeiro instante após o início em que aparecem
        n_fim amostras consecutivas exatamente no estado da baseline.

    O instante final retornado é o PRIMEIRO ponto da sequência de retorno.
    """

    tempo = proc["Tempo_rel_s"].to_numpy()
    estados = proc["Estado"].to_numpy()

    baseline_mask = tempo <= baseline_segundos

    if baseline_mask.sum() == 0:
        raise ValueError("Não foi possível definir a janela de baseline.")

    valores, contagens = np.unique(
        estados[baseline_mask],
        return_counts=True
    )

    estado_baseline = int(valores[np.argmax(contagens)])

    # Índice logo após o fim da baseline
    idx_busca = int(np.searchsorted(
        tempo,
        baseline_segundos,
        side="right"
    ))

    # --------------------------------------------------------
    # Detecta início
    # --------------------------------------------------------
    idx_inicio = None

    for i in range(idx_busca, len(estados) - n_inicio + 1):
        janela = estados[i:i + n_inicio]

        if (
            estados[i] != estado_baseline
            and np.all(janela > estado_baseline)
        ):
            idx_inicio = i
            break

    # --------------------------------------------------------
    # Detecta fim
    # --------------------------------------------------------
    idx_fim = None

    if idx_inicio is not None:
        primeiro_idx_fim = idx_inicio + n_inicio

        for i in range(
            primeiro_idx_fim,
            len(estados) - n_fim + 1
        ):
            janela = estados[i:i + n_fim]

            if np.all(janela == estado_baseline):
                idx_fim = i
                break

    return estado_baseline, idx_inicio, idx_fim


def csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8-sig")


# ============================================================
# Barra lateral
# ============================================================
st.sidebar.header("Parâmetros")

fs = st.sidebar.number_input(
    "Frequência após interpolação (Hz)",
    min_value=20.0,
    max_value=500.0,
    value=100.0,
    step=10.0
)

cutoff = st.sidebar.number_input(
    "Corte do filtro passa-baixa (Hz)",
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

n_clusters = st.sidebar.number_input(
    "Número de estados do K-means",
    min_value=2,
    max_value=10,
    value=5,
    step=1
)

baseline_segundos = st.sidebar.number_input(
    "Duração da baseline (s)",
    min_value=0.5,
    max_value=10.0,
    value=2.0,
    step=0.5
)

n_inicio = st.sidebar.number_input(
    "Amostras consecutivas para início",
    min_value=1,
    max_value=100,
    value=5,
    step=1
)

n_fim = st.sidebar.number_input(
    "Amostras consecutivas para fim",
    min_value=1,
    max_value=200,
    value=10,
    step=1
)


# ============================================================
# Upload
# ============================================================
arquivo = st.file_uploader(
    "Selecione o arquivo CSV",
    type=["csv"]
)

if arquivo is not None:

    try:
        df = pd.read_csv(arquivo)

        st.subheader("Arquivo importado")
        st.dataframe(df.head(), use_container_width=True)

        colunas = list(df.columns)

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            time_col = st.selectbox(
                "Tempo",
                colunas,
                index=colunas.index("TempoMs") if "TempoMs" in colunas else 0
            )

        with c2:
            x_col = st.selectbox(
                "Eixo X",
                colunas,
                index=colunas.index("X") if "X" in colunas else 0
            )

        with c3:
            y_col = st.selectbox(
                "Eixo Y",
                colunas,
                index=colunas.index("Y") if "Y" in colunas else 0
            )

        with c4:
            z_col = st.selectbox(
                "Eixo Z",
                colunas,
                index=colunas.index("Z") if "Z" in colunas else 0
            )

        # ----------------------------------------------------
        # Processamento
        # ----------------------------------------------------
        proc, fs_original = preparar_dados(
            df,
            time_col=time_col,
            x_col=x_col,
            y_col=y_col,
            z_col=z_col,
            fs=float(fs),
            cutoff=float(cutoff),
            filter_order=int(filter_order)
        )

        proc, centroides = aplicar_kmeans(
            proc,
            n_clusters=int(n_clusters)
        )

        estado_baseline, idx_inicio, idx_fim = detectar_segmento(
            proc,
            baseline_segundos=float(baseline_segundos),
            n_inicio=int(n_inicio),
            n_fim=int(n_fim)
        )

        # ----------------------------------------------------
        # Resumo
        # ----------------------------------------------------
        st.subheader("Resultado da segmentação")

        m1, m2, m3, m4 = st.columns(4)

        m1.metric(
            "Frequência original estimada",
            f"{fs_original:.1f} Hz"
        )

        m2.metric(
            "Estado dominante da baseline",
            str(estado_baseline)
        )

        if idx_inicio is not None:
            tempo_inicio = proc.loc[idx_inicio, "Tempo_rel_s"]
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

        if idx_fim is not None:
            tempo_fim = proc.loc[idx_fim, "Tempo_rel_s"]
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

        if idx_inicio is not None and idx_fim is not None:
            duracao = tempo_fim - tempo_inicio

            st.success(
                f"Segmento detectado: {tempo_inicio:.3f} s a "
                f"{tempo_fim:.3f} s | duração = {duracao:.3f} s"
            )

        # ----------------------------------------------------
        # Centróides
        # ----------------------------------------------------
        cent_df = pd.DataFrame({
            "Estado": np.arange(len(centroides)),
            "Centroide_magnitude": centroides
        })

        st.subheader("Estados do K-means ordenados por amplitude")
        st.dataframe(
            cent_df,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # Gráfico 1 - magnitude
        # ----------------------------------------------------
        st.subheader("Magnitude da aceleração")

        fig1, ax1 = plt.subplots(figsize=(13, 4))

        ax1.plot(
            proc["Tempo_rel_s"],
            proc["Magnitude"],
            linewidth=1
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

        ax1.set_xlabel("Tempo (s)")
        ax1.set_ylabel("Magnitude")
        ax1.legend()
        ax1.grid(alpha=0.2)

        st.pyplot(fig1)
        plt.close(fig1)

        # ----------------------------------------------------
        # Gráfico 2 - estados
        # ----------------------------------------------------
        st.subheader("Sequência de estados")

        fig2, ax2 = plt.subplots(figsize=(13, 3))

        ax2.step(
            proc["Tempo_rel_s"],
            proc["Estado"],
            where="post",
            linewidth=1
        )

        ax2.axhline(
            estado_baseline,
            linestyle=":",
            linewidth=2,
            label=f"Baseline = estado {estado_baseline}"
        )

        if tempo_inicio is not None:
            ax2.axvline(
                tempo_inicio,
                linestyle="--",
                linewidth=2
            )

        if tempo_fim is not None:
            ax2.axvline(
                tempo_fim,
                linestyle="--",
                linewidth=2
            )

        ax2.set_xlabel("Tempo (s)")
        ax2.set_ylabel("Estado")
        ax2.set_yticks(range(int(n_clusters)))
        ax2.grid(alpha=0.2)
        ax2.legend()

        st.pyplot(fig2)
        plt.close(fig2)

        # ----------------------------------------------------
        # Extrai segmento
        # ----------------------------------------------------
        if idx_inicio is not None and idx_fim is not None:

            segmento = proc.iloc[idx_inicio:idx_fim + 1].copy()
            segmento["Tempo_segmento_s"] = (
                segmento["Tempo_rel_s"]
                - segmento["Tempo_rel_s"].iloc[0]
            )

            st.subheader("Dados segmentados")
            st.dataframe(
                segmento.head(100),
                use_container_width=True
            )

            st.download_button(
                "Baixar segmento CSV",
                data=csv_bytes(segmento),
                file_name="segmento_detectado.csv",
                mime="text/csv"
            )

        st.download_button(
            "Baixar sinal processado completo",
            data=csv_bytes(proc),
            file_name="sinal_processado_com_estados.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Erro durante o processamento: {e}")
