import pandas as pd


def detectar_tipos_columnas(df: pd.DataFrame, umbral_fecha: float = 0.8):
    """
    Detecta columnas:
    - booleanas
    - numéricas
    - fechas
    - categóricas (NO se transforman)

    Retorna listas de columnas y el dataframe transformado
    """

    df = df.copy()

    columnas_booleanas = []
    columnas_numericas = []
    columnas_categoricas = []
    columnas_fecha = []

    mapa_booleanos = {
        'yes': True, 'y': True, 'si': True, 'sí': True, 'true': True, '1': True,
        'no': False, 'n': False, 'false': False, '0': False
    }

    for col in df.columns:
        serie = df[col]
        dtype = serie.dtype

        # ============================
        # BOOLEANOS NATIVOS
        # ============================
        if pd.api.types.is_bool_dtype(dtype):
            df[col] = serie.astype(bool)
            columnas_booleanas.append(col)
            continue

        # ============================
        # NUMÉRICOS
        # ============================
        if pd.api.types.is_numeric_dtype(dtype):
            valores_unicos = set(serie.dropna().unique())
            if valores_unicos.issubset({0, 1}):
                df[col] = serie.astype(bool)
                columnas_booleanas.append(col)
            else:
                df[col] = pd.to_numeric(serie, errors="coerce")
                columnas_numericas.append(col)
            continue

        # ============================
        # FECHAS
        # ============================
        fechas = pd.to_datetime(serie, errors="coerce", infer_datetime_format=True)
        if fechas.notna().mean() >= umbral_fecha:
            df[col] = fechas
            columnas_fecha.append(col)
            continue

        # ============================
        # CATEGÓRICAS (SOLO DETECTAR)
        # ============================
        columnas_categoricas.append(col)

    return {
        'numericas': columnas_numericas,
        'booleanas': columnas_booleanas,
        'fechas': columnas_fecha,
        'categoricas': columnas_categoricas,
        'dataframe_transformado': df
    }
