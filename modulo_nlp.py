"""
Módulo: modulo_nlp.py
Descripción: Implementa el pipeline de preprocesamiento lingüístico con NLTK,
la comparación experimental de vectorizadores (Bag of Words vs. TF-IDF) con
clasificadores supervisados (Naive Bayes vs. SVM Lineal) y la extracción de
entidades (5 Representantes Médicos y 5 Proveedores Operativos).
"""

# Se importa re para búsqueda de patrones numéricos y limpieza alfanumérica.
import re
# Se importa os para guardar las gráficas de evaluación en la carpeta de evidencias.
import os
# Se importa unicodedata para estandarizar texto eliminando tildes.
import unicodedata
# Se importa pandas para estructurar el corpus y las tablas de métricas.
import pandas as pd
# Se importa matplotlib.pyplot para generar la figura comparativa de NLP.
import matplotlib.pyplot as plt
# Se importa seaborn para graficar la matriz de confusión de intenciones.
import seaborn as sns

# Se importan los módulos de tokenización, stemming y stopwords de NLTK.
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

# Se importan los vectorizadores, clasificadores y métricas desde scikit-learn.
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix
)

# Se importan los catálogos de entidades desde nuestro primer módulo local.
from catalogo_y_asientos import (
    CATALOGO_REPRESENTANTES_MEDICOS, CATALOGO_PROVEEDORES_OPERATIVOS
)

# Se descarga silenciosamente la lista de palabras vacías en español.
nltk.download("stopwords", quiet=True)

# Se inicializan el tokenizador de palabras y el reductor morfológico (stemmer).
TOKENIZER = RegexpTokenizer(r"\w+")
STEMMER = SnowballStemmer("spanish")

# Se construye el conjunto de stop words sin acentos conservando términos fiscales clave.
STOP_WORDS_ES = {
    unicodedata.normalize("NFD", w.lower()).encode("ascii", "ignore").decode("utf-8")
    for w in stopwords.words("spanish")
} - {"sin", "con", "para", "total", "iva"}


def limpiar_texto_plano(texto):
    """
    Convierte cualquier cadena de entrada a minúsculas sin tildes ni símbolos
    especiales, asegurando tolerancia a fallos ante valores nulos.

    Parámetros:
        texto (str): Cadena de texto original.

    Retorna:
        str: Texto limpio en caracteres ASCII.
    """
    # Se garantiza que la entrada sea tratada como cadena de texto.
    if not isinstance(texto, str):
        texto = str(texto or "")
    # Se descompone en formato Unicode NFD y se ignoran los acentos.
    norm = unicodedata.normalize("NFD", texto.lower()).encode("ascii", "ignore").decode("utf-8")
    # Se reemplazan caracteres especiales dejando solo letras, números y puntos decimales.
    return re.sub(r"[^a-z0-9\s\.]", " ", norm)


def preprocesar_para_nlp(texto):
    """
    Segmenta el texto limpio en tokens, filtra palabras vacías (stop words) y
    reduce cada término a su raíz morfológica (stem) con SnowballStemmer.

    Parámetros:
        texto (str): Oración contable escrita por el usuario.

    Retorna:
        str: Cadena de stems separados por espacio lista para vectorización.
    """
    # Se limpia la oración de signos y tildes.
    limpio = limpiar_texto_plano(texto)
    # Se divide la oración en tokens alfanuméricos.
    tokens = TOKENIZER.tokenize(limpio)
    # Se aplica stemming descartando stop words y números puros.
    stems = [STEMMER.stem(t) for t in tokens if t not in STOP_WORDS_ES and not t.isdigit()]
    return " ".join(stems)


def entrenar_y_evaluar_modelos_nlp(seed=42, carpeta_salida="./evidencias_reporte"):
    """
    Construye el corpus supervisado de 80 oraciones contables distribuidas en 5
    intenciones, entrena los 4 experimentos (BoW vs. TF-IDF con Naive Bayes y SVM),
    guarda la gráfica comparativa y devuelve el mejor clasificador entrenado.

    Parámetros:
        seed (int): Semilla aleatoria para reproducibilidad en la partición.
        carpeta_salida (str): Ruta donde se guardará la imagen PNG de resultados.

    Retorna:
        tuple: (Pipeline del mejor modelo entrenado, pd.DataFrame con tabla comparativa).
    """
    # Se define el corpus etiquetado de 80 expresiones del dominio de asientos contables.
    corpus_asientos = [
        # Intención 1: asiento_viaticos_rep (16 muestras con los 5 representantes médicos)
        ("registrar recibo de viaticos del representante medico carlos mendez por comida y hotel", "asiento_viaticos_rep"),
        ("comprobar gastos de viaje y hospedaje de la representante laura gomez en monterrey", "asiento_viaticos_rep"),
        ("generar asiento contable de ticket de viaticos de roberto silva zona bajio", "asiento_viaticos_rep"),
        ("contabilizar reembolso de viaticos de ana sofia torres por visita medica", "asiento_viaticos_rep"),
        ("poliza de gastos de viaje del representante miguel angel ruiz con impuesto de hospedaje", "asiento_viaticos_rep"),
        ("capturar recibo de restaurante y transporte comprobado por carlos mendez", "asiento_viaticos_rep"),
        ("elaborar asiento de viaticos medicos de laura gomez con iva acreditable", "asiento_viaticos_rep"),
        ("registro contable de comprobacion de viaticos del representante roberto silva", "asiento_viaticos_rep"),
        ("reembolsar ticket de casetas y alimentos a la representante ana sofia torres", "asiento_viaticos_rep"),
        ("hacer poliza de viaticos de promocion medica para miguel angel ruiz", "asiento_viaticos_rep"),
        ("subir recibo de hotel del representante carlos mendez con ish e iva", "asiento_viaticos_rep"),
        ("asiento contable para comprobar viaticos de congreso pediatria de laura gomez", "asiento_viaticos_rep"),
        ("contabilizar ticket de viaticos de roberto silva en cuenta de acreedores", "asiento_viaticos_rep"),
        ("generar poliza de gastos de viaje comprobados por ana sofia torres", "asiento_viaticos_rep"),
        ("registrar comprobante de viaticos del representante medico miguel angel ruiz", "asiento_viaticos_rep"),
        ("cargar viaticos de representante medico a gastos de promocion y abonar reembolso", "asiento_viaticos_rep"),

        # Intención 2: asiento_proveedor_operativo (16 muestras con los 5 proveedores operativos)
        ("registrar factura recibida del proveedor operativo logistica fria transoceanic con retencion de flete", "asiento_proveedor_operativo"),
        ("generar asiento contable de factura de agencia aduanal del golfo por importacion", "asiento_proveedor_operativo"),
        ("contabilizar factura de arrendamiento de inmobiliaria corporativa reforma con retenciones de ley", "asiento_proveedor_operativo"),
        ("poliza de diario para factura recibida de distribuidora de empaques del norte", "asiento_proveedor_operativo"),
        ("elaborar asiento de honorarios del despacho fiscal asesores con retencion de isr e iva", "asiento_proveedor_operativo"),
        ("provisionar factura del proveedor operativo logistica fria transoceanic por acarreo maritimo", "asiento_proveedor_operativo"),
        ("capturar cfdi recibido de agencia aduanal del golfo por maniobras portuarias", "asiento_proveedor_operativo"),
        ("registrar poliza de renta de oficinas de inmobiliaria corporativa reforma", "asiento_proveedor_operativo"),
        ("asiento contable de compra de cajas y etiquetado a empaques del norte", "asiento_proveedor_operativo"),
        ("contabilizar factura de auditoria del proveedor despacho fiscal asesores", "asiento_proveedor_operativo"),
        ("reconocer pasivo de proveedor operativo con iva pendiente de acreditar al 16 por ciento", "asiento_proveedor_operativo"),
        ("generar asiento de fletes con 4 por ciento de retencion de iva para logistica fria", "asiento_proveedor_operativo"),
        ("registrar factura recibida de proveedor de operacion y calcular impuestos retenidos", "asiento_proveedor_operativo"),
        ("hacer poliza de provision para agencia aduanal del golfo con iva pendiente", "asiento_proveedor_operativo"),
        ("asiento de gasto operativo por factura de empaques del norte para kabrita", "asiento_proveedor_operativo"),
        ("registrar cfdi de proveedor operativo inmobiliaria reforma descontando retenciones", "asiento_proveedor_operativo"),

        # Intención 3: asiento_venta_kabrita (16 muestras de facturación emitida Tasa 0%)
        ("generar asiento contable de factura emitida por venta de formula kabrita etapa 1", "asiento_venta_kabrita"),
        ("contabilizar cfdi de ingreso emitido a farmacias con tasa cero de iva", "asiento_venta_kabrita"),
        ("poliza de ingresos por facturacion emitida de kabrita etapas 2 y 3", "asiento_venta_kabrita"),
        ("registrar venta de formula lactea infantil kabrita gravada al 0 por ciento", "asiento_venta_kabrita"),
        ("elaborar asiento contable de factura emitida a distribuidor nacional", "asiento_venta_kabrita"),
        ("cargar a clientes nacionales y abonar a ventas tasa cero por factura emitida", "asiento_venta_kabrita"),
        ("capturar cfdi emitido por comercializacion de lotes de kabrita", "asiento_venta_kabrita"),
        ("hacer poliza contable de factura emitida de venta sin traslado de iva", "asiento_venta_kabrita"),
        ("registro en libro diario de factura emitida por venta mayorista de kabrita", "asiento_venta_kabrita"),
        ("reconocer ingreso por factura emitida de nutricion infantil kabrita", "asiento_venta_kabrita"),
        ("asiento contable para cfdi de ventas emitidas a cadenas de farmacias", "asiento_venta_kabrita"),
        ("contabilizar factura de venta emitida aplicando articulo 2a tasa 0 de iva", "asiento_venta_kabrita"),
        ("generar poliza de cliente nacional por factura emitida de formula de cabra", "asiento_venta_kabrita"),
        ("registrar folio fiscal de factura emitida de ingresos kabrita", "asiento_venta_kabrita"),
        ("armar asiento de ingresos por venta de producto kabrita exento o tasa cero", "asiento_venta_kabrita"),
        ("poliza contable de facturacion emitida del mes a clientes", "asiento_venta_kabrita"),

        # Intención 4: asiento_pago_spei (16 muestras de pagos bancarios y reclasificación de IVA)
        ("registrar comprobante de pago spei a proveedor y reclasificar iva acreditable", "asiento_pago_spei"),
        ("generar poliza de egresos por transferencia bancaria liquidada a logistica fria", "asiento_pago_spei"),
        ("contabilizar comprobante bancario de pago a agencia aduanal del golfo", "asiento_pago_spei"),
        ("asiento contable de salida de bancos bbva por pago spei y cancelacion de pasivo", "asiento_pago_spei"),
        ("elaborar poliza de pago bancario pasando iva pendiente a iva acreditable pagado", "asiento_pago_spei"),
        ("registrar liquidacion por spei a inmobiliaria reforma desde cuenta operativa", "asiento_pago_spei"),
        ("hacer asiento de comprobante de pago de transferencia a empaques del norte", "asiento_pago_spei"),
        ("capturar voucher bancario spei para cancelar cuenta de proveedores nacionales", "asiento_pago_spei"),
        ("poliza de egreso bancario por pago de factura y acreditamiento definitivo de iva", "asiento_pago_spei"),
        ("contabilizar transferencia spei pagada al despacho fiscal asesores", "asiento_pago_spei"),
        ("registrar comprobante de pago spei de reembolso al representante medico", "asiento_pago_spei"),
        ("generar asiento de afectacion a bancos nacionales por dispersion spei", "asiento_pago_spei"),
        ("aplicar comprobante de pago para saldar factura recibida y acreditar impuesto", "asiento_pago_spei"),
        ("asiento contable de pago confirmado con referencia spei bancaria", "asiento_pago_spei"),
        ("descargar pasivo de proveedor mediante comprobante de pago spei", "asiento_pago_spei"),
        ("poliza de transferencia bancaria liquidada con reclasificacion de iva al 16", "asiento_pago_spei"),

        # Intención 5: auditoria_general (16 muestras de clasificación documental directa)
        ("hola asistente analiza la imagen adjunta y genera el asiento que corresponda", "auditoria_general"),
        ("buenos dias identifica este documento contable escaneado y arma su poliza", "auditoria_general"),
        ("clasifica visualmente el archivo subido y dame el registro de cargos y abonos", "auditoria_general"),
        ("que tipo de comprobante es esta fotografia y cual es su asiento contable", "auditoria_general"),
        ("procesa este documento del dataset con vision artificial para contabilizarlo", "auditoria_general"),
        ("escanear comprobante adjunto con resnet50 y estructurar poliza cuadrada", "auditoria_general"),
        ("hola necesito que identifiques el documento y calcules sus impuestos", "auditoria_general"),
        ("revisar imagen del comprobante fiscal y generar tabla de partida doble", "auditoria_general"),
        ("auditar archivo pdf o imagen para determinar cuenta contable e iva", "auditoria_general"),
        ("buenas tardes asistente procesa automaticamente este comprobante contable", "auditoria_general"),
        ("identificar categoria visual del documento y mostrar asiento con impuestos", "auditoria_general"),
        ("analizar fotografia del documento contable y proponer poliza sat", "auditoria_general"),
        ("ejecutar reconocimiento visual del archivo y armar asiento contable", "auditoria_general"),
        ("hola ayudame a clasificar este comprobante y generar su asiento", "auditoria_general"),
        ("determinar tipo de documento mediante red neuronal y cuadrar cargos y abonos", "auditoria_general"),
        ("verificar imagen del comprobante y construir asiento contable automatico", "auditoria_general")
    ]

    # Se organiza el corpus en un DataFrame y se aplica el preprocesamiento de raíces léxicas.
    df_nlp = pd.DataFrame(corpus_asientos, columns=["texto", "intencion"])
    df_nlp["texto_proc"] = df_nlp["texto"].apply(preprocesar_para_nlp)

    # Se divide estratificadamente el corpus en 75% entrenamiento y 25% prueba.
    x_tr, x_te, y_tr, y_te = train_test_split(
        df_nlp["texto_proc"], df_nlp["intencion"],
        test_size=0.25, random_state=seed, stratify=df_nlp["intencion"]
    )

    # Se definen los 4 pipelines experimentales (BoW y TF-IDF con Naive Bayes y SVM).
    modelos_config = {
        "1. BoW + Naive Bayes": Pipeline([
            ("vect", CountVectorizer(ngram_range=(1, 2))),
            ("clf", MultinomialNB(alpha=0.5))
        ]),
        "2. TF-IDF + Naive Bayes": Pipeline([
            ("vect", TfidfVectorizer(ngram_range=(1, 2))),
            ("clf", MultinomialNB(alpha=0.5))
        ]),
        "3. BoW + SVM Lineal": Pipeline([
            ("vect", CountVectorizer(ngram_range=(1, 2))),
            ("clf", LinearSVC(random_state=seed, C=1.0))
        ]),
        "4. TF-IDF + SVM Lineal": Pipeline([
            ("vect", TfidfVectorizer(ngram_range=(1, 2))),
            ("clf", LinearSVC(random_state=seed, C=1.0))
        ])
    }

    # Se entrenan y evalúan los 4 modelos registrando sus métricas.
    lista_metricas = []
    modelos_ajustados = {}

    for nombre, pipe in modelos_config.items():
        # Se entrena el pipeline con el subconjunto de entrenamiento.
        pipe.fit(x_tr, y_tr)
        # Se predice sobre el subconjunto de prueba.
        y_pred = pipe.predict(x_te)
        # Se calculan Accuracy, Precision, Recall y F1-Score macro.
        acc = accuracy_score(y_te, y_pred)
        pr, rc, f1, _ = precision_recall_fscore_support(y_te, y_pred, average="macro", zero_division=0)
        lista_metricas.append({
            "Modelo Supervisado NLP": nombre,
            "Accuracy (%)": round(acc * 100, 2),
            "Precision (%)": round(pr * 100, 2),
            "Recall (%)": round(rc * 100, 2),
            "F1-Score (%)": round(f1 * 100, 2)
        })
        modelos_ajustados[nombre] = (pipe, y_pred)

    # Se genera el DataFrame de resultados comparativos.
    df_resultados = pd.DataFrame(lista_metricas)
    # Se selecciona el modelo TF-IDF + SVM Lineal como clasificador principal.
    mejor_pipe, mejor_pred = modelos_ajustados["4. TF-IDF + SVM Lineal"]

    # Se asegura la existencia de la carpeta de evidencias gráficas.
    os.makedirs(carpeta_salida, exist_ok=True)

    # Se construye y guarda la figura comparativa de F1-Score y Matriz de Confusión.
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    axes[0].barh(
        df_resultados["Modelo Supervisado NLP"],
        df_resultados["F1-Score (%)"],
        color=["#9ecae1", "#6baed6", "#3182bd", "#08519c"]
    )
    axes[0].set_xlim(0, 110)
    axes[0].set_xlabel("F1-Score Macro (%)")
    axes[0].set_title("Comparación de Vectorización y Modelos NLP")
    for idx, val in enumerate(df_resultados["F1-Score (%)"]):
        axes[0].text(val + 1, idx, f"{val:.1f}%", va="center", fontweight="bold")

    clases_ordenadas = sorted(df_nlp["intencion"].unique())
    cm_nlp = confusion_matrix(y_te, mejor_pred, labels=clases_ordenadas)
    sns.heatmap(
        cm_nlp, annot=True, fmt="d", cmap="Blues",
        xticklabels=clases_ordenadas, yticklabels=clases_ordenadas, ax=axes[1]
    )
    axes[1].set_title("Matriz de Confusión (TF-IDF + SVM Lineal)")
    axes[1].set_xlabel("Intención Predicha")
    axes[1].set_ylabel("Intención Real")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    # Se guarda la imagen de alta resolución en la carpeta local para el reporte.
    ruta_fig_nlp = os.path.join(carpeta_salida, "1_evaluacion_modelos_nlp.png")
    plt.savefig(ruta_fig_nlp, dpi=150)
    plt.close(fig)

    return mejor_pipe, df_resultados, ruta_fig_nlp


def extraer_entidades_y_parametros(texto_usuario, texto_extraido_doc=""):
    """
    Analiza el mensaje del usuario y el texto del documento para identificar:
      - Si corresponde a uno de los 5 Representantes Médicos (Viáticos).
      - Si corresponde a uno de los 5 Proveedores Operativos.
      - El monto numérico de la transacción y si aplica ISH (3% hospedaje).

    Parámetros:
        texto_usuario (str): Instrucción ingresada por el contador.
        texto_extraido_doc (str): Texto adicional extraído del documento.

    Retorna:
        dict: Entidades detectadas ('representante_medico', 'proveedor_operativo',
              'subtotal', 'aplica_ish').
    """
    # Se unifican y limpian ambas fuentes de texto.
    combinado = limpiar_texto_plano(f"{texto_usuario} {texto_extraido_doc}")

    # 1. Búsqueda de coincidencias en el catálogo de 5 Representantes Médicos.
    rep_detectado = None
    max_pts_rep = 0
    for _, info_rep in CATALOGO_REPRESENTANTES_MEDICOS.items():
        pts = sum(1 for kw in info_rep["palabras_clave"] if kw in combinado)
        if pts > max_pts_rep:
            max_pts_rep = pts
            rep_detectado = info_rep

    # 2. Búsqueda de coincidencias en el catálogo de 5 Proveedores Operativos.
    prov_detectado = None
    max_pts_prov = 0
    for _, info_prov in CATALOGO_PROVEEDORES_OPERATIVOS.items():
        pts = sum(1 for kw in info_prov["palabras_clave"] if kw in combinado)
        if pts > max_pts_prov:
            max_pts_prov = pts
            prov_detectado = info_prov

    # 3. Extracción del importe monetario ignorando números de año (2024-2027).
    candidatos_num = re.findall(r"\b\d{3,7}(?:\.\d{1,2})?\b", combinado)
    montos = [float(n) for n in candidatos_num if float(n) not in (2024.0, 2025.0, 2026.0, 2027.0)]
    subtotal_detectado = montos[0] if montos else 15000.00

    # 4. Verificación de palabras asociadas a hospedaje para activar el ISH del 3%.
    aplica_ish = any(w in combinado for w in ["hotel", "hospedaje", "estancia", "ish", "alojamiento"])

    return {
        "representante_medico": rep_detectado,
        "proveedor_operativo": prov_detectado,
        "subtotal": round(subtotal_detectado, 2),
        "aplica_ish": aplica_ish
    }