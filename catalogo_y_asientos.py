"""
Módulo: catalogo_y_asientos.py
Descripción: Gestiona el catálogo contable de Ausnutria Nutrition México (Kabrita),
incluyendo los 5 Representantes Médicos (proveedores de viáticos), los 5 Proveedores
Operativos y el generador de asientos contables cuadrados con cálculo de impuestos.
"""

# Se importa pandas para estructurar los catálogos y las pólizas en tablas contables.
import pandas as pd

# ==============================================================================
# 1. CATÁLOGO DE 5 REPRESENTANTES MÉDICOS (COMPROBACIÓN DE VIÁTICOS)
# ==============================================================================
# Diccionario que vincula a cada representante médico con su cuenta de gasto y pasivo.
CATALOGO_REPRESENTANTES_MEDICOS = {
    "carlos mendez": {
        "nombre_oficial": "Carlos Méndez (Rep. Médico Zona Centro)",
        "palabras_clave": ["carlos", "mendez", "zona centro"],
        "cuenta_gasto": "601.32.01",
        "nombre_cuenta_gasto": "Gastos de Viaje y Viáticos - Promoción Médica Centro",
        "cuenta_pasivo": "205.02.01",
        "nombre_cuenta_pasivo": "Acreedores Diversos Viáticos - Carlos Méndez"
    },
    "laura gomez": {
        "nombre_oficial": "Laura Gómez (Rep. Médico Zona Norte)",
        "palabras_clave": ["laura", "gomez", "zona norte"],
        "cuenta_gasto": "601.32.02",
        "nombre_cuenta_gasto": "Gastos de Viaje y Viáticos - Promoción Médica Norte",
        "cuenta_pasivo": "205.02.02",
        "nombre_cuenta_pasivo": "Acreedores Diversos Viáticos - Laura Gómez"
    },
    "roberto silva": {
        "nombre_oficial": "Roberto Silva (Rep. Médico Zona Bajío)",
        "palabras_clave": ["roberto", "silva", "bajio"],
        "cuenta_gasto": "601.32.03",
        "nombre_cuenta_gasto": "Gastos de Viaje y Viáticos - Promoción Médica Bajío",
        "cuenta_pasivo": "205.02.03",
        "nombre_cuenta_pasivo": "Acreedores Diversos Viáticos - Roberto Silva"
    },
    "ana sofia torres": {
        "nombre_oficial": "Ana Sofía Torres (Rep. Médico Zona Occidente)",
        "palabras_clave": ["ana", "sofia", "torres", "occidente"],
        "cuenta_gasto": "601.32.04",
        "nombre_cuenta_gasto": "Gastos de Viaje y Viáticos - Promoción Médica Occidente",
        "cuenta_pasivo": "205.02.04",
        "nombre_cuenta_pasivo": "Acreedores Diversos Viáticos - Ana Sofía Torres"
    },
    "miguel angel ruiz": {
        "nombre_oficial": "Miguel Ángel Ruiz (Rep. Médico Zona Sureste)",
        "palabras_clave": ["miguel", "angel", "ruiz", "sureste"],
        "cuenta_gasto": "601.32.05",
        "nombre_cuenta_gasto": "Gastos de Viaje y Viáticos - Promoción Médica Sureste",
        "cuenta_pasivo": "205.02.05",
        "nombre_cuenta_pasivo": "Acreedores Diversos Viáticos - Miguel Ángel Ruiz"
    }
}

# ==============================================================================
# 2. CATÁLOGO DE 5 PROVEEDORES OPERATIVOS Y SUS REGLAS DE IMPUESTOS
# ==============================================================================
# Diccionario que define cuentas de costo/gasto, pasivo y tasas de retención fiscal.
CATALOGO_PROVEEDORES_OPERATIVOS = {
    "logistica fria transoceanic": {
        "nombre_oficial": "Logística Fría Transoceanic S.A. de C.V.",
        "palabras_clave": ["logistica", "fria", "transoceanic", "flete", "maritimo", "acarreo"],
        "cuenta_gasto": "601.48.01",
        "nombre_cuenta_gasto": "Fletes y Acarreos de Importación Kabrita",
        "cuenta_pasivo": "201.01.01",
        "nombre_cuenta_pasivo": "Proveedores Nacionales - Logística Fría Transoceanic",
        "tasa_iva": 0.16,       # Tasa general de IVA del 16%
        "ret_iva": 0.04,        # Retención del 4% de IVA aplicable a fletes
        "ret_isr": 0.00,        # Sin retención de ISR por tratarse de persona moral de transporte
        "cuenta_ret_isr": None
    },
    "agencia aduanal del golfo": {
        "nombre_oficial": "Agencia Aduanal del Golfo S.C.",
        "palabras_clave": ["agencia", "aduanal", "golfo", "aduana", "pedimento", "maniobras"],
        "cuenta_gasto": "601.54.01",
        "nombre_cuenta_gasto": "Gastos y Derechos Aduanales de Importación",
        "cuenta_pasivo": "201.01.02",
        "nombre_cuenta_pasivo": "Proveedores Nacionales - Agencia Aduanal del Golfo",
        "tasa_iva": 0.16,
        "ret_iva": 0.00,
        "ret_isr": 0.00,
        "cuenta_ret_isr": None
    },
    "inmobiliaria corporativa reforma": {
        "nombre_oficial": "Inmobiliaria Corporativa Reforma S.A.",
        "palabras_clave": ["inmobiliaria", "reforma", "renta", "arrendamiento", "oficinas", "bodega"],
        "cuenta_gasto": "601.45.01",
        "nombre_cuenta_gasto": "Arrendamiento de Oficinas y Almacén Sanitario",
        "cuenta_pasivo": "201.01.03",
        "nombre_cuenta_pasivo": "Proveedores Nacionales - Inmobiliaria Reforma",
        "tasa_iva": 0.16,
        "ret_iva": 0.106667,    # Retención de 2/3 partes del IVA (10.6667%)
        "ret_isr": 0.10,        # Retención del 10% de ISR sobre arrendamiento
        "cuenta_ret_isr": ("216.04.01", "ISR Retenido por Arrendamiento (10%)")
    },
    "empaques del norte": {
        "nombre_oficial": "Distribuidora de Empaques del Norte S.A. de C.V.",
        "palabras_clave": ["empaques", "etiquetado", "cajas", "envases", "insumos", "nom051"],
        "cuenta_gasto": "501.03.01",
        "nombre_cuenta_gasto": "Costo de Material de Empaque y Etiquetado NOM-051",
        "cuenta_pasivo": "201.01.04",
        "nombre_cuenta_pasivo": "Proveedores Nacionales - Empaques del Norte",
        "tasa_iva": 0.16,
        "ret_iva": 0.00,
        "ret_isr": 0.00,
        "cuenta_ret_isr": None
    },
    "despacho fiscal asesores": {
        "nombre_oficial": "Despacho Contable y Fiscal Asesores S.C.",
        "palabras_clave": ["despacho", "asesores", "honorarios", "auditoria", "contable", "consultoria"],
        "cuenta_gasto": "601.34.01",
        "nombre_cuenta_gasto": "Honorarios Profesionales y Auditoría Fiscal",
        "cuenta_pasivo": "201.01.05",
        "nombre_cuenta_pasivo": "Proveedores Nacionales - Despacho Fiscal Asesores",
        "tasa_iva": 0.16,
        "ret_iva": 0.106667,    # Retención de 2/3 partes del IVA (10.6667%)
        "ret_isr": 0.10,        # Retención del 10% de ISR por servicios profesionales
        "cuenta_ret_isr": ("216.03.01", "ISR Retenido por Honorarios Profesionales (10%)")
    }
}


def obtener_tabla_catalogo_entidades():
    """
    Construye y devuelve un DataFrame con el resumen completo de las 10 entidades
    configuradas (5 Representantes Médicos y 5 Proveedores Operativos), mostrando
    sus cuentas de cargo, abono y reglas de impuestos.

    Retorna:
        pd.DataFrame: Tabla estructurada con el catálogo de entidades contables.
    """
    # Se inicializa la lista que almacenará los registros del catálogo.
    filas = []
    # Se recorren los 5 representantes médicos para agregarlos a la tabla.
    for _, v in CATALOGO_REPRESENTANTES_MEDICOS.items():
        filas.append({
            "Tipo_Entidad": "Representante Médico (Viáticos)",
            "Entidad": v["nombre_oficial"],
            "Cuenta_Cargo": f"{v['cuenta_gasto']} ({v['nombre_cuenta_gasto']})",
            "Cuenta_Abono": f"{v['cuenta_pasivo']} ({v['nombre_cuenta_pasivo']})",
            "Regla_Impuestos": "IVA Acreditable 16% (118.01) + ISH 3% (si incluye hotel)"
        })
    # Se recorren los 5 proveedores operativos para agregarlos a la tabla.
    for _, v in CATALOGO_PROVEEDORES_OPERATIVOS.items():
        imp_str = f"IVA 16% (119.01) | Ret. IVA: {v['ret_iva']*100:.2f}% | Ret. ISR: {v['ret_isr']*100:.0f}%"
        filas.append({
            "Tipo_Entidad": "Proveedor Operativo",
            "Entidad": v["nombre_oficial"],
            "Cuenta_Cargo": f"{v['cuenta_gasto']} ({v['nombre_cuenta_gasto']})",
            "Cuenta_Abono": f"{v['cuenta_pasivo']} ({v['nombre_cuenta_pasivo']})",
            "Regla_Impuestos": imp_str
        })
    # Se convierte la lista de diccionarios en un DataFrame de pandas.
    return pd.DataFrame(filas)


def construir_asiento_contable_completo(clase_documento, intencion_nlp, entidades, folio_doc="CFDI-2026-MX"):
    """
    Genera la póliza del asiento contable respetando la ley de partida doble
    (Suma de Cargos == Suma de Abonos) a partir del tipo de documento visual,
    la intención textual, la entidad detectada y los impuestos correspondientes.

    Parámetros:
        clase_documento (str): Categoría visual ('comprobante_pago', 'factura_emitida',
                               'factura_recibida', 'recibo_viaticos').
        intencion_nlp (str): Intención clasificada por el modelo supervisado NLP.
        entidades (dict): Diccionario con representante_medico, proveedor_operativo,
                          subtotal y aplica_ish.
        folio_doc (str): Folio o referencia fiscal del comprobante auditado.

    Retorna:
        tuple: (pd.DataFrame con el detalle del asiento cuadrado, str con el título fiscal).
    """
    # Se extrae el importe subtotal base desde el diccionario de entidades.
    subtotal = entidades["subtotal"]
    # Se extrae el representante médico si fue detectado en la instrucción.
    rep_medico = entidades["representante_medico"]
    # Se extrae el proveedor operativo si fue detectado en la instrucción.
    prov_operativo = entidades["proveedor_operativo"]
    # Se verifica si la operación de viáticos involucra hospedaje para calcular ISH 3%.
    aplica_ish = entidades["aplica_ish"]
    # Se inicializa la lista de renglones que conformarán la póliza contable.
    movimientos = []

    # --------------------------------------------------------------------------
    # CASO 1: RECIBO DE VIÁTICOS COMPROBADO POR REPRESENTANTE MÉDICO
    # --------------------------------------------------------------------------
    if (clase_documento == "recibo_viaticos" or
            intencion_nlp == "asiento_viaticos_rep" or
            (rep_medico is not None and prov_operativo is None)):
        # Si no se especificó nombre, se asigna el primer representante por defecto para evitar errores.
        rep = rep_medico if rep_medico else CATALOGO_REPRESENTANTES_MEDICOS["carlos mendez"]
        # Se calcula el IVA acreditable al 16% sobre el subtotal del gasto.
        iva_16 = round(subtotal * 0.16, 2)
        # Se calcula el Impuesto Sobre Hospedaje (3%) si el gasto incluye hotel.
        ish_3 = round(subtotal * 0.03, 2) if aplica_ish else 0.00
        # Se calcula el monto total a reembolsar al representante médico.
        total_reembolso = round(subtotal + iva_16 + ish_3, 2)

        # Se define el encabezado descriptivo de la póliza.
        resumen = (f"Póliza de Viáticos Médicos | Representante: {rep['nombre_oficial']} | "
                   f"Impuestos: IVA 16% + ISH {'3%' if aplica_ish else '0%'}")
        # Cargo 1: Cuenta de gasto de viaje asignada a la zona del representante médico.
        movimientos.append({
            "Cuenta_SAT": rep["cuenta_gasto"],
            "Nombre_Cuenta": rep["nombre_cuenta_gasto"],
            "Concepto": f"Viáticos comprobados s/g {folio_doc}",
            "Cargo_Debe": subtotal,
            "Abono_Haber": 0.00
        })
        # Cargo 2: IVA Acreditable Pagado al 16%.
        movimientos.append({
            "Cuenta_SAT": "118.01.01",
            "Nombre_Cuenta": "IVA Acreditable Pagado (16%)",
            "Concepto": f"IVA CFDI viáticos {folio_doc}",
            "Cargo_Debe": iva_16,
            "Abono_Haber": 0.00
        })
        # Cargo 3 (Condicional): Impuesto Sobre Hospedaje (ISH 3%).
        if ish_3 > 0:
            movimientos.append({
                "Cuenta_SAT": "601.84.01",
                "Nombre_Cuenta": "Impuesto Sobre Hospedaje Local (ISH 3%)",
                "Concepto": f"ISH estancia hotel {folio_doc}",
                "Cargo_Debe": ish_3,
                "Abono_Haber": 0.00
            })
        # Abono: Cuenta de pasivo a favor del representante médico en Acreedores Diversos.
        movimientos.append({
            "Cuenta_SAT": rep["cuenta_pasivo"],
            "Nombre_Cuenta": rep["nombre_cuenta_pasivo"],
            "Concepto": f"Reembolso por pagar a representante {folio_doc}",
            "Cargo_Debe": 0.00,
            "Abono_Haber": total_reembolso
        })

    # --------------------------------------------------------------------------
    # CASO 2: FACTURA EMITIDA (VENTA DE FÓRMULA INFANTIL KABRITA - TASA 0% IVA)
    # --------------------------------------------------------------------------
    elif clase_documento == "factura_emitida" or intencion_nlp == "asiento_venta_kabrita":
        # Las fórmulas infantiles Kabrita están gravadas a Tasa 0% según Art. 2-A LIVA.
        resumen = "Póliza de Ingresos | Venta Fórmula Infantil Kabrita | Impuesto: IVA Tasa 0% (Art. 2-A LIVA)"
        # Cargo: Clientes Nacionales por el total de la factura emitida.
        movimientos.append({
            "Cuenta_SAT": "105.01.01",
            "Nombre_Cuenta": "Clientes Nacionales (Distribuidores Kabrita)",
            "Concepto": f"Venta Kabrita s/g factura {folio_doc}",
            "Cargo_Debe": subtotal,
            "Abono_Haber": 0.00
        })
        # Abono: Ingresos por ventas gravadas a tasa 0%.
        movimientos.append({
            "Cuenta_SAT": "401.04.01",
            "Nombre_Cuenta": "Ventas Gravadas a Tasa 0% (Línea Kabrita)",
            "Concepto": f"Ingreso Tasa 0% IVA s/g {folio_doc}",
            "Cargo_Debe": 0.00,
            "Abono_Haber": subtotal
        })

    # --------------------------------------------------------------------------
    # CASO 3: COMPROBANTE DE PAGO SPEI (CANCELACIÓN DE PASIVO Y RECLASIFICACIÓN IVA)
    # --------------------------------------------------------------------------
    elif clase_documento == "comprobante_pago" or intencion_nlp == "asiento_pago_spei":
        # Se selecciona el proveedor operativo detectado o el proveedor logístico por defecto.
        prov = prov_operativo if prov_operativo else CATALOGO_PROVEEDORES_OPERATIVOS["logistica fria transoceanic"]
        # Se calculan los impuestos que fueron provisionados en la factura original.
        iva_16 = round(subtotal * prov["tasa_iva"], 2)
        ret_iva = round(subtotal * prov["ret_iva"], 2)
        ret_isr = round(subtotal * prov["ret_isr"], 2)
        # Se determina el flujo neto de efectivo transferido por SPEI.
        neto_pagado = round(subtotal + iva_16 - ret_iva - ret_isr, 2)

        resumen = f"Póliza de Egreso SPEI | Beneficiario: {prov['nombre_oficial']} | Reclasificación IVA 16%"
        # Cargo 1: Cancelación del pasivo en la subcuenta del proveedor operativo.
        movimientos.append({
            "Cuenta_SAT": prov["cuenta_pasivo"],
            "Nombre_Cuenta": prov["nombre_cuenta_pasivo"],
            "Concepto": f"Liquidación de factura vía SPEI {folio_doc}",
            "Cargo_Debe": neto_pagado,
            "Abono_Haber": 0.00
        })
        # Abono 1: Salida de dinero de la cuenta bancaria operativa BBVA.
        movimientos.append({
            "Cuenta_SAT": "102.01.01",
            "Nombre_Cuenta": "Bancos Nacionales BBVA Cuenta Operativa",
            "Concepto": f"Transferencia SPEI {folio_doc}",
            "Cargo_Debe": 0.00,
            "Abono_Haber": neto_pagado
        })
        # Cargo 2: Reclasificación fiscal a IVA Acreditable efectivamente pagado.
        movimientos.append({
            "Cuenta_SAT": "118.01.01",
            "Nombre_Cuenta": "IVA Acreditable Pagado (16%)",
            "Concepto": f"Acreditamiento definitivo IVA {folio_doc}",
            "Cargo_Debe": iva_16,
            "Abono_Haber": 0.00
        })
        # Abono 2: Cancelación de la cuenta puente de IVA Pendiente de Pago.
        movimientos.append({
            "Cuenta_SAT": "119.01.01",
            "Nombre_Cuenta": "IVA Pendiente de Pago (16%)",
            "Concepto": f"Cancelación IVA pendiente {folio_doc}",
            "Cargo_Debe": 0.00,
            "Abono_Haber": iva_16
        })

    # --------------------------------------------------------------------------
    # CASO 4: FACTURA RECIBIDA DE PROVEEDOR OPERATIVO (CON IVA Y RETENCIONES)
    # --------------------------------------------------------------------------
    else:
        # Se asigna el proveedor operativo identificado o Logística Fría por defecto.
        prov = prov_operativo if prov_operativo else CATALOGO_PROVEEDORES_OPERATIVOS["logistica fria transoceanic"]
        # Se calcula el IVA trasladado al 16% y las retenciones de ley según el proveedor.
        iva_16 = round(subtotal * prov["tasa_iva"], 2)
        ret_iva = round(subtotal * prov["ret_iva"], 2)
        ret_isr = round(subtotal * prov["ret_isr"], 2)
        # El abono neto al proveedor descuenta las retenciones de ISR e IVA.
        neto_proveedor = round(subtotal + iva_16 - ret_iva - ret_isr, 2)

        resumen = f"Póliza de Provisión Operativa | Proveedor: {prov['nombre_oficial']} | IVA 16% y Retenciones"
        # Cargo 1: Cuenta de gasto o costo operativo específica del proveedor.
        movimientos.append({
            "Cuenta_SAT": prov["cuenta_gasto"],
            "Nombre_Cuenta": prov["nombre_cuenta_gasto"],
            "Concepto": f"Factura recibida s/g {folio_doc}",
            "Cargo_Debe": subtotal,
            "Abono_Haber": 0.00
        })
        # Cargo 2: IVA Pendiente de Pago al 16%.
        movimientos.append({
            "Cuenta_SAT": "119.01.01",
            "Nombre_Cuenta": "IVA Pendiente de Pago (16%)",
            "Concepto": f"IVA 16% por acreditar {folio_doc}",
            "Cargo_Debe": iva_16,
            "Abono_Haber": 0.00
        })
        # Abono Condicional 1: Retención de ISR (10% en arrendamiento u honorarios).
        if ret_isr > 0 and prov["cuenta_ret_isr"] is not None:
            movimientos.append({
                "Cuenta_SAT": prov["cuenta_ret_isr"][0],
                "Nombre_Cuenta": prov["cuenta_ret_isr"][1],
                "Concepto": f"Retención ISR 10% s/g {folio_doc}",
                "Cargo_Debe": 0.00,
                "Abono_Haber": ret_isr
            })
        # Abono Condicional 2: Retención de IVA (4% en fletes o 10.67% en honorarios/rentas).
        if ret_iva > 0:
            movimientos.append({
                "Cuenta_SAT": "216.10.01",
                "Nombre_Cuenta": "Impuestos Retenidos de IVA (Fletes/Servicios)",
                "Concepto": f"Retención IVA de ley s/g {folio_doc}",
                "Cargo_Debe": 0.00,
                "Abono_Haber": ret_iva
            })
        # Abono Principal: Pasivo neto a pagar en la subcuenta del proveedor operativo.
        movimientos.append({
            "Cuenta_SAT": prov["cuenta_pasivo"],
            "Nombre_Cuenta": prov["nombre_cuenta_pasivo"],
            "Concepto": f"Pasivo neto a favor proveedor {folio_doc}",
            "Cargo_Debe": 0.00,
            "Abono_Haber": neto_proveedor
        })

    # Se convierte la lista de movimientos en un DataFrame contable.
    df_asiento = pd.DataFrame(movimientos)
    # Se calculan las sumas totales de Cargos (Debe) y Abonos (Haber).
    suma_debe = round(df_asiento["Cargo_Debe"].sum(), 2)
    suma_haber = round(df_asiento["Abono_Haber"].sum(), 2)

    # Se agrega el renglón verificador de Sumas Iguales al pie de la tabla.
    fila_cuadre = pd.DataFrame([{
        "Cuenta_SAT": "SUMAS IGUALES",
        "Nombre_Cuenta": "VERIFICACIÓN DE PARTIDA DOBLE (DEBE == HABER)",
        "Concepto": "Póliza cuadrada sin diferencias",
        "Cargo_Debe": suma_debe,
        "Abono_Haber": suma_haber
    }])
    df_asiento = pd.concat([df_asiento, fila_cuadre], ignore_index=True)

    return df_asiento, resumen