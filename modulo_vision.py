"""
Módulo: modulo_vision.py
Descripción: Gestiona la lectura segura de documentos contables locales (.pdf, .png,
.jpg) desde ./dataset_contable/, calcula pesos de clase para compensar la escasez
de recibos de viáticos, aplica Data Augmentation y entrena el clasificador visual
mediante Transfer Learning y Fine-Tuning con ResNet50.
"""

# Se importa os para el manejo de rutas de directorios locales.
import os
# Se importa glob para buscar archivos de imagen y PDF en las subcarpetas.
import glob
# Se importa random para transformaciones controladas en memoria.
import random
# Se importa numpy para construir los tensores de entrenamiento y prueba.
import numpy as np
# Se importa pandas para estructurar las métricas de desempeño visual.
import pandas as pd
# Se importa matplotlib.pyplot para guardar las gráficas de visión y activaciones.
import matplotlib.pyplot as plt
# Se importa seaborn para renderizar la matriz de confusión de documentos.
import seaborn as sns
# Se importan Image e ImageDraw de Pillow para procesar imágenes locales.
from PIL import Image, ImageDraw

# Se verifica la disponibilidad de PyMuPDF (fitz) para convertir páginas PDF a imagen.
try:
    import fitz
    PYMUPDF_ACTIVO = True
except ImportError:
    PYMUPDF_ACTIVO = False

# Se importan herramientas de partición, pesos de clase y matriz de confusión.
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix

# Se importa TensorFlow y los componentes de Keras para ResNet50.
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# Nombres exactos de las 4 subcarpetas en ./dataset_contable/
CLASES_DOCUMENTOS = [
    "comprobante_pago",
    "factura_emitida",
    "factura_recibida",
    "recibo_viaticos"
]

# Resolución espacial de entrada para ResNet50 (96x96 píxeles en 3 canales RGB).
IMG_SIZE = 96
# Cantidad mínima en memoria por clase para garantizar particiones estratificadas sin error.
MINIMO_MUESTRAS_RAM = 60


def leer_documento_local_seguro(ruta_archivo):
    """
    Abre de manera segura un archivo local (.pdf, .png, .jpg, .jpeg) y lo convierte
    en un arreglo numérico RGB de (96, 96, 3). Si el archivo está corrupto, devuelve
    None sin interrumpir el programa.

    Parámetros:
        ruta_archivo (str): Ruta física del documento en el disco local.

    Retorna:
        np.ndarray | None: Tensor de imagen (96, 96, 3) o None en caso de fallo.
    """
    try:
        extension = os.path.splitext(ruta_archivo)[1].lower()
        # Si el documento es un archivo PDF, se rasteriza su primera página con PyMuPDF.
        if extension == ".pdf":
            if not PYMUPDF_ACTIVO:
                return None
            doc_pdf = fitz.open(ruta_archivo)
            pix = doc_pdf.load_page(0).get_pixmap(dpi=120)
            img_pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            doc_pdf.close()
        else:
            # Si es PNG, JPG o JPEG, se abre directamente con Pillow en modo RGB.
            img_pil = Image.open(ruta_archivo).convert("RGB")

        # Se redimensiona la imagen al tamaño estándar de la red neuronal.
        img_redim = img_pil.resize((IMG_SIZE, IMG_SIZE))
        return np.array(img_redim, dtype=np.float32)
    except Exception:
        return None


def _generar_plantilla_memoria(clase_idx, semilla):
    """
    Función interna de respaldo que sintetiza una plantilla visual en memoria RAM
    (sin escribir archivos en las carpetas del usuario) únicamente cuando una
    carpeta tiene menos muestras del umbral estadístico.

    Parámetros:
        clase_idx (int): Índice numérico de la clase documental (0 a 3).
        semilla (int): Semilla de variación geométrica.

    Retorna:
        np.ndarray: Imagen RGB sintética de (96, 96, 3).
    """
    rng = random.Random(semilla)
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (235, 235, 232))
    draw = ImageDraw.Draw(img)
    dx, dy = rng.randint(-3, 3), rng.randint(-3, 3)
    nombre_c = CLASES_DOCUMENTOS[clase_idx]

    if nombre_c == "comprobante_pago":
        draw.rectangle([6+dx, 16+dy, 90+dx, 80+dy], fill=(255, 252, 246), outline=(170, 145, 120))
        draw.rectangle([6+dx, 16+dy, 90+dx, 28+dy], fill=(155, 35, 50))
        draw.ellipse([58+dx, 54+dy, 82+dx, 74+dy], outline=(190, 50, 50), width=2)
    elif nombre_c == "factura_emitida":
        draw.rectangle([8+dx, 6+dy, 88+dx, 90+dy], fill=(252, 252, 250), outline=(175, 175, 175))
        draw.rectangle([8+dx, 6+dy, 88+dx, 20+dy], fill=(28, 142, 72))
        draw.rectangle([14+dx, 68+dy, 30+dx, 84+dy], fill=(25, 25, 25))
    elif nombre_c == "factura_recibida":
        draw.rectangle([8+dx, 6+dy, 88+dx, 90+dy], fill=(246, 249, 255), outline=(160, 170, 190))
        draw.rectangle([8+dx, 6+dy, 88+dx, 20+dy], fill=(25, 55, 155))
        draw.rectangle([66+dx, 68+dy, 82+dx, 84+dy], fill=(20, 30, 55))
    else:
        draw.rectangle([0, 0, IMG_SIZE, IMG_SIZE], fill=(140, 130, 120))
        draw.rectangle([26+dx, 4+dy, 70+dx, 92+dy], fill=(250, 248, 240), outline=(130, 130, 120))
        for x_b in range(31, 66, 3):
            draw.line([x_b+dx, 73+dy, x_b+dx, 86+dy], fill=(15, 15, 15), width=2)

    return np.array(img, dtype=np.float32)


def cargar_dataset_y_entrenar_resnet50(ruta_dataset="./dataset_contable", seed=42, carpeta_salida="./evidencias_reporte"):
    """
    Lee las 4 subcarpetas de ./dataset_contable/, aplica compensación para clases
    con pocos archivos (como recibo_viaticos), ejecuta Transfer Learning y Fine-Tuning
    con ResNet50 y guarda las gráficas de evaluación y mapas de activación.

    Parámetros:
        ruta_dataset (str): Ruta de la carpeta local con los documentos contables.
        seed (int): Semilla de reproducibilidad.
        carpeta_salida (str): Carpeta donde se exportarán las figuras PNG.

    Retorna:
        tuple: (Modelo ResNet50 entrenado, X_test_img, y_test_img, df_comparacion_visual,
                conteo_archivos_reales, diccionario_pesos_clases).
    """
    x_lista, y_lista = [], []
    conteo_reales = {}

    # 1. Escaneo de las 4 carpetas locales dentro de ./dataset_contable/
    for idx_c, nombre_clase in enumerate(CLASES_DOCUMENTOS):
        carpeta_c = os.path.join(ruta_dataset, nombre_clase)
        os.makedirs(carpeta_c, exist_ok=True)

        # Se recolectan archivos PDF, PNG, JPG y JPEG.
        archivos = []
        for pat in ("*.png", "*.jpg", "*.jpeg", "*.pdf", "*.PNG", "*.JPG", "*.PDF"):
            archivos.extend(glob.glob(os.path.join(carpeta_c, pat)))
        archivos = sorted(list(set(archivos)))

        imgs_clase = []
        for ruta_f in archivos:
            tensor = leer_documento_local_seguro(ruta_f)
            if tensor is not None:
                imgs_clase.append(tensor)

        conteo_reales[nombre_clase] = len(imgs_clase)

        # Se completa en memoria RAM hasta MINIMO_MUESTRAS_RAM si hay pocos recibos físicos.
        while len(imgs_clase) < MINIMO_MUESTRAS_RAM:
            sem = seed * 500 + idx_c * 100 + len(imgs_clase)
            if conteo_reales[nombre_clase] > 0:
                img_base = Image.fromarray(imgs_clase[len(imgs_clase) % conteo_reales[nombre_clase]].astype(np.uint8))
                ang = random.Random(sem).uniform(-5, 5)
                rotada = img_base.rotate(ang, fillcolor=(240, 240, 240))
                imgs_clase.append(np.array(rotada, dtype=np.float32))
            else:
                imgs_clase.append(_generar_plantilla_memoria(idx_c, sem))

        for im in imgs_clase:
            x_lista.append(im)
            y_lista.append(idx_c)

    x_total = np.array(x_lista, dtype=np.float32)
    y_total = np.array(y_lista, dtype=np.int32)

    # 2. División estratificada en Entrenamiento (64%), Validación (16%) y Prueba (20%).
    x_tmp, x_test, y_tmp, y_test = train_test_split(
        x_total, y_total, test_size=0.20, random_state=seed, stratify=y_total
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_tmp, y_tmp, test_size=0.20, random_state=seed, stratify=y_tmp
    )

    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=4)
    y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=4)
    y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=4)

    # 3. Cálculo de Pesos de Clase (Class Weights).
    pesos_arr = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
    pesos_dict = dict(enumerate(pesos_arr))

    # 4. Definición de la capa de Aumento de Datos (Data Augmentation).
    capa_aug = tf.keras.Sequential([
        layers.RandomRotation(0.05, seed=seed),
        layers.RandomZoom(0.07, seed=seed),
        layers.RandomTranslation(0.04, 0.04, seed=seed)
    ], name="data_augmentation_contable")

    os.makedirs(carpeta_salida, exist_ok=True)

    # Se guarda la evidencia visual de las 4 clases y su Data Augmentation.
    fig_aug, axes_aug = plt.subplots(2, 4, figsize=(13, 5.8))
    for i in range(4):
        idx_m = np.where(y_train == i)[0][0]
        axes_aug[0, i].imshow(x_train[idx_m].astype(np.uint8))
        axes_aug[0, i].set_title(f"Original: {CLASES_DOCUMENTOS[i]}", fontsize=9, fontweight="bold")
        axes_aug[0, i].axis("off")

        im_aug = capa_aug(tf.expand_dims(x_train[idx_m], 0), training=True)[0].numpy()
        axes_aug[1, i].imshow(np.clip(im_aug, 0, 255).astype(np.uint8))
        axes_aug[1, i].set_title("Con Data Augmentation", fontsize=9)
        axes_aug[1, i].axis("off")

    plt.suptitle("Muestras de ./dataset_contable/ y Efecto de Data Augmentation", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(carpeta_salida, "2_muestras_data_augmentation.png"), dpi=150)
    plt.close(fig_aug)

    # 5. Construcción y entrenamiento Fase 1: Transfer Learning con ResNet50.
    callbacks_vis = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)
    ]

    base_resnet = ResNet50(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_resnet.trainable = False

    entradas = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = capa_aug(entradas)
    x = preprocess_input(x)
    x = base_resnet(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.35)(x)
    salidas = layers.Dense(4, activation="softmax")(x)

    modelo_resnet = models.Model(entradas, salidas, name="ResNet50_Contable")
    modelo_resnet.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0008),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    modelo_resnet.fit(
        x_train, y_train_cat,
        validation_data=(x_val, y_val_cat),
        epochs=12, batch_size=16,
        class_weight=pesos_dict,
        callbacks=callbacks_vis,
        verbose=0
    )
    loss_tl, acc_tl = modelo_resnet.evaluate(x_test, y_test_cat, verbose=0)

    # 6. Fase 2: Fine-Tuning descongelando las últimas 20 capas de ResNet50.
    base_resnet.trainable = True
    for capa in base_resnet.layers[:-20]:
        capa.trainable = False

    modelo_resnet.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    modelo_resnet.fit(
        x_train, y_train_cat,
        validation_data=(x_val, y_val_cat),
        epochs=8, batch_size=16,
        class_weight=pesos_dict,
        callbacks=callbacks_vis,
        verbose=0
    )
    loss_ft, acc_ft = modelo_resnet.evaluate(x_test, y_test_cat, verbose=0)

    df_comp_vis = pd.DataFrame([
        {"Etapa del Modelo Visual": "1. ResNet50 Transfer Learning (Base Congelada)", "Exactitud Prueba (%)": round(acc_tl * 100, 2), "Pérdida (Loss)": round(loss_tl, 4)},
        {"Etapa del Modelo Visual": "2. ResNet50 Fine-Tuning (Últimas 20 Capas)", "Exactitud Prueba (%)": round(acc_ft * 100, 2), "Pérdida (Loss)": round(loss_ft, 4)}
    ])

    # 7. Gráfica de comparación y Matriz de Confusión Visual.
    pred_vis = np.argmax(modelo_resnet.predict(x_test, verbose=0), axis=1)
    cm_vis = confusion_matrix(y_test, pred_vis)

    fig_ev, axes_ev = plt.subplots(1, 2, figsize=(13, 4.8))
    axes_ev[0].bar(df_comp_vis["Etapa del Modelo Visual"], df_comp_vis["Exactitud Prueba (%)"], color=["#41ab5d", "#005a32"], width=0.45)
    axes_ev[0].set_ylim(0, 112)
    axes_ev[0].set_ylabel("Exactitud en Prueba (%)")
    axes_ev[0].set_title("Rendimiento Transfer Learning vs. Fine-Tuning")
    for i, v in enumerate(df_comp_vis["Exactitud Prueba (%)"]):
        axes_ev[0].text(i, v + 2, f"{v:.2f}%", ha="center", fontweight="bold")

    sns.heatmap(cm_vis, annot=True, fmt="d", cmap="Greens", xticklabels=CLASES_DOCUMENTOS, yticklabels=CLASES_DOCUMENTOS, ax=axes_ev[1])
    axes_ev[1].set_title("Matriz de Confusión en Documentos Contables")
    axes_ev[1].set_xlabel("Documento Predicho")
    axes_ev[1].set_ylabel("Documento Real")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(carpeta_salida, "3_evaluacion_resnet50_confusion.png"), dpi=150)
    plt.close(fig_ev)

    # 8. Extracción y guardado de Mapas de Activación de ResNet50.
    capa_conv = base_resnet.get_layer("conv1_relu").output
    extractor_act = models.Model(inputs=base_resnet.input, outputs=capa_conv)
    muestra = x_test[0:1]
    activaciones = extractor_act.predict(preprocess_input(muestra.copy()), verbose=0)

    fig_act, axes_act = plt.subplots(1, 5, figsize=(14, 3))
    axes_act[0].imshow(muestra[0].astype(np.uint8))
    axes_act[0].set_title(f"Doc: {CLASES_DOCUMENTOS[y_test[0]]}", fontweight="bold", fontsize=9)
    axes_act[0].axis("off")
    for f_i in range(4):
        axes_act[f_i + 1].imshow(activaciones[0, :, :, f_i * 4], cmap="viridis")
        axes_act[f_i + 1].set_title(f"Mapa Activación {f_i + 1}", fontsize=9)
        axes_act[f_i + 1].axis("off")

    plt.suptitle("Mapas de Activación Convolucionales en ResNet50", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(carpeta_salida, "4_mapas_activacion_resnet50.png"), dpi=150)
    plt.close(fig_act)

    return modelo_resnet, x_test, y_test, df_comp_vis, conteo_reales, pesos_dict