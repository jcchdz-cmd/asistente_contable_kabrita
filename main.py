"""
Módulo Principal: main.py
Descripción: Orquesta la ejecución completa del Avance de Proyecto 2 en Visual
Studio Code. Carga el catálogo de 5 Representantes Médicos y 5 Proveedores Operativos,
evalúa los modelos NLP, entrena ResNet50 sobre ./dataset_contable/ y genera los
asientos contables con impuestos.
"""

# Se importa sys para mostrar la versión de Python en la consola de VS Code.
import sys
# Se importa warnings para mantener limpia la salida de la terminal.
import warnings
warnings.filterwarnings("ignore")

# Se importa numpy para localizar muestras de cada clase en las pruebas integradas.
import numpy as np
# Se importa tensorflow para reportar la versión instalada.
import tensorflow as tf

# Se importan las funciones de nuestros tres módulos locales del proyecto.
from catalogo_y_asientos import (
    obtener_tabla_catalogo_entidades, construir_asiento_contable_completo
)
from modulo_nlp import (
    preprocesar_para_nlp, entrenar_y_evaluar_modelos_nlp, extraer_entidades_y_parametros
)
from modulo_vision import (
    CLASES_DOCUMENTOS, cargar_dataset_y_entrenar_resnet50
)


def ejecutar_caso_integrado(instruccion, idx_test, folio, clf_nlp, mod_vis, x_test):
    """
    Ejecuta un escenario completo del asistente: clasifica la intención con TF-IDF+SVM,
    identifica visualmente el comprobante con ResNet50, extrae el proveedor operativo
    o el nombre del representante médico y genera la tabla del asiento contable con impuestos.

    Parámetros:
        instruccion (str): Solicitud en lenguaje natural del contador.
        idx_test (int): Índice del documento dentro del conjunto de prueba.
        folio (str): Folio fiscal del comprobante.
        clf_nlp (Pipeline): Clasificador TF-IDF + SVM entrenado.
        mod_vis (Model): Red neuronal ResNet50 entrenada.
        x_test (np.ndarray): Conjunto de imágenes de prueba.
    """
    # 1. Clasificación de la intención textual.
    intencion = clf_nlp.predict([preprocesar_para_nlp(instruccion)])[0]

    # 2. Clasificación visual del comprobante con ResNet50.
    tensor_doc = x_test[idx_test:idx_test+1]
    probs = mod_vis.predict(tensor_doc, verbose=0)[0]
    idx_clase = int(np.argmax(probs))
    clase_visual = CLASES_DOCUMENTOS[idx_clase]
    certeza = float(probs[idx_clase]) * 100

    # 3. Extracción de entidades (Representante Médico / Proveedor Operativo / Monto / ISH).
    entidades = extraer_entidades_y_parametros(instruccion)

    # 4. Construcción del asiento contable con impuestos cuadrados.
    df_asiento, resumen_poliza = construir_asiento_contable_completo(
        clase_documento=clase_visual,
        intencion_nlp=intencion,
        entidades=entidades,
        folio_doc=folio
    )

    # 5. Impresión formateada en la consola de Visual Studio Code.
    print("=" * 100)
    print(f"INSTRUCCIÓN DEL USUARIO:           \"{instruccion}\"")
    print(f"DOCUMENTO IDENTIFICADO (ResNet50): [{clase_visual}] (Certeza: {certeza:.2f}%)")
    print(f"INTENCIÓN DETECTADA (TF-IDF+SVM):  [{intencion}]")
    print(f"ENCABEZADO DE PÓLIZA E IMPUESTOS:  {resumen_poliza}")
    print("-" * 100)
    print(df_asiento.to_string(index=False))
    print("=" * 100 + "\n")


def main():
    """
    Función principal que ejecuta secuencialmente todos los bloques requeridos
    por la rúbrica del Avance de Proyecto 2.
    """
    print("\n" + "=" * 100)
    print("ASISTENTE CONTABLE MULTIMODAL KABRITA - EJECUCIÓN LOCAL EN VISUAL STUDIO CODE")
    print(f"Versión de Python: {sys.version.split()[0]} | Versión de TensorFlow: {tf.__version__}")
    print("=" * 100 + "\n")

    # 1. Mostrar catálogo de 5 Representantes Médicos y 5 Proveedores Operativos.
    print("[PASO 1] CATÁLOGO DE ENTIDADES CONTABLES (5 REP. MÉDICOS Y 5 PROVEEDORES OPERATIVOS):")
    df_cat = obtener_tabla_catalogo_entidades()
    print(df_cat.to_string(index=False))
    print("\n")

    # 2. Entrenar y evaluar modelos supervisados de NLP (BoW vs. TF-IDF con NB y SVM).
    print("[PASO 2] ENTRENANDO Y COMPARANDO MODELOS SUPERVISADOS DE NLP...")
    clf_nlp, df_res_nlp, ruta_graf_nlp = entrenar_y_evaluar_modelos_nlp()
    print(df_res_nlp.to_string(index=False))
    print(f"-> Gráfica NLP guardada en: {ruta_graf_nlp}\n")

    # 3. Cargar ./dataset_contable/ y entrenar ResNet50 (Transfer Learning y Fine-Tuning).
    print("[PASO 3] ESCANEANDO ./dataset_contable/ Y ENTRENANDO RESNET50 (TRANSFER LEARNING + FINE-TUNING)...")
    mod_vis, x_test, y_test, df_res_vis, conteo_reales, pesos_dict = cargar_dataset_y_entrenar_resnet50()
    print("Archivos físicos leídos por carpeta:", conteo_reales)
    print("Pesos de clase aplicados:", {CLASES_DOCUMENTOS[k]: round(v, 3) for k, v in pesos_dict.items()})
    print("\nRESULTADOS DE VISIÓN POR COMPUTADORA (RESNET50):")
    print(df_res_vis.to_string(index=False))
    print("-> Gráficas visuales guardadas en la carpeta: ./evidencias_reporte/\n")

    # 4. Ejecución de 4 casos de prueba integrados con desglose de impuestos.
    print("[PASO 4] DEMOSTRACIÓN DE GENERACIÓN DE ASIENTOS CONTABLES E IMPUESTOS:\n")

    # Caso 1: Viáticos comprobados por Representante Médico (IVA 16% + ISH 3% de hotel).
    idx_viat = int(np.where(y_test == CLASES_DOCUMENTOS.index("recibo_viaticos"))[0][0])
    ejecutar_caso_integrado(
        instruccion="Registrar recibo de viaticos de hotel y alimentos comprobado por la representante medica Laura Gomez por 6500.00",
        idx_test=idx_viat,
        folio="VIAT-LG-019",
        clf_nlp=clf_nlp, mod_vis=mod_vis, x_test=x_test
    )

    # Caso 2: Proveedor Operativo de Fletes (IVA 16% y Retención del 4% de IVA).
    idx_frec = int(np.where(y_test == CLASES_DOCUMENTOS.index("factura_recibida"))[0][0])
    ejecutar_caso_integrado(
        instruccion="Generar asiento contable de factura recibida del proveedor operativo Logistica Fria Transoceanic por flete maritimo de 85000.00",
        idx_test=idx_frec,
        folio="FR-LFT-402",
        clf_nlp=clf_nlp, mod_vis=mod_vis, x_test=x_test
    )

    # Caso 3: Proveedor Operativo de Arrendamiento (IVA 16%, Ret. ISR 10% y Ret. IVA 10.67%).
    ejecutar_caso_integrado(
        instruccion="Contabilizar factura de renta de oficinas de Inmobiliaria Corporativa Reforma por 45000.00 aplicando retenciones de ley",
        idx_test=idx_frec,
        folio="FR-ICR-889",
        clf_nlp=clf_nlp, mod_vis=mod_vis, x_test=x_test
    )

    # Caso 4: Factura Emitida por Venta de Fórmula Infantil Kabrita (Tasa 0% IVA).
    idx_femi = int(np.where(y_test == CLASES_DOCUMENTOS.index("factura_emitida"))[0][0])
    ejecutar_caso_integrado(
        instruccion="Elaborar asiento contable de factura emitida por venta de formula infantil Kabrita Etapa 1 y 2 por 210000.00",
        idx_test=idx_femi,
        folio="FE-KAB-915",
        clf_nlp=clf_nlp, mod_vis=mod_vis, x_test=x_test
    )


if __name__ == "__main__":
    main()