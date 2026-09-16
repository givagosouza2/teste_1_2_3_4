def detectar_segmento(
    proc,
    baseline_segundos=1.0,
    n_inicio=10,
    n_fim=50
):

    tempo = proc["Tempo_rel_s"].to_numpy()
    estados = proc["Estado"].to_numpy()

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------
    baseline_mask = tempo <= baseline_segundos

    if baseline_mask.sum() == 0:
        raise ValueError("Não foi possível definir a janela de baseline.")

    valores, contagens = np.unique(
        estados[baseline_mask],
        return_counts=True
    )

    estado_baseline = int(
        valores[np.argmax(contagens)]
    )

    idx_busca = int(
        np.searchsorted(
            tempo,
            baseline_segundos,
            side="right"
        )
    )

    # --------------------------------------------------------
    # INÍCIO
    #
    # Procura uma sequência de estados DIFERENTES
    # do estado dominante da baseline.
    # --------------------------------------------------------
    idx_inicio = None

    for i in range(
        idx_busca,
        len(estados) - n_inicio + 1
    ):

        janela = estados[i:i+n_inicio]

        if np.all(janela != estado_baseline):
            idx_inicio = i
            break

    # --------------------------------------------------------
    # FIM
    #
    # Procura retorno sustentado ao estado da baseline.
    # --------------------------------------------------------
    idx_fim = None

    if idx_inicio is not None:

        primeiro_idx_fim = idx_inicio + n_inicio

        for i in range(
            primeiro_idx_fim,
            len(estados) - n_fim + 1
        ):

            janela = estados[i:i+n_fim]

            if np.all(janela == estado_baseline):
                idx_fim = i
                break

    return estado_baseline, idx_inicio, idx_fim
