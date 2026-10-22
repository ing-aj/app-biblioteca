import streamlit as st
import fitz  # PyMuPDF para leer PDFs gigantes rápido
import pandas as pd
import re
import json
import unicodedata
from io import BytesIO
from rapidfuzz import fuzz
from google import genai

# Configuración de la página web
st.set_page_config(page_title="Sistema Bibliotecario", page_icon="📚", layout="wide")

st.title("📚 Extractor de Bibliografías y Comparador de Inventario")
st.markdown("Extrae los libros únicos de los Planes de Estudio (PDF) y compáralos con los Exceles de la Biblioteca.")

# --- BARRA LATERAL PARA LA LLAVE DE IA ---
with st.sidebar:
    st.header("🔑 Configuración")
    api_key = st.text_input("Pega aquí tu API Key de Gemini:", type="password")
    st.caption("Consíguela gratis en: aistudio.google.com")
    st.divider()
    umbral_similitud = st.slider(
        "Nivel de coincidencia (Título y Autor)", 
        min_value=60, max_value=95, value=78,
        help="78% es ideal para encontrar libros aunque el título o autor tengan pequeñas variaciones."
    )

# --- FUNCIONES DE LIMPIEZA Y EXTRACCIÓN ---
def limpiar_texto(texto):
    """Quita acentos, diagonales, signos y pasa todo a minúsculas para comparar mejor."""
    if pd.isna(texto) or not str(texto).strip():
        return ""
    texto = str(texto).lower().strip()
    texto = texto.replace("/", " ")  # Quita diagonales de los títulos del Excel
    texto = ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )
    texto = re.sub(r'[^a-z0-9\s]', ' ', texto)
    return re.sub(r'\s+', ' ', texto).strip()

def extraer_bloques_bibliografia(archivo_pdf):
    """Recorre PDFs de hasta +1000 páginas y recorta solo las secciones de Bibliografía."""
    doc = fitz.open(stream=archivo_pdf.read(), filetype="pdf")
    bloques = []
    total_paginas = len(doc)
    
    palabras_fin = [
        "DURACION:", "DURACIÓN:", "CREDITOS:", "CRÉDITOS:", 
        "OBJETIVO:", "TEMARIO", "MODALIDAD DE ENSEÑANZA", 
        "METODOLOGIA DE ENSEÑANZA", "METODOLOGÍA DE ENSEÑANZA",
        "PROCEDIMIENTOS DE EVALUACION", "PROCEDIMIENTO DE EVALUACION"
    ]
    
     i = 0
    while i < total_paginas:
        texto_pag = doc[i].get_text("text")
        texto_sin_acentos = limpiar_texto(texto_pag)
        
        if "bibliografia" in texto_sin_acentos:
            # Unimos la página actual y la siguiente por si la bibliografía se corta entre hojas
            texto_combinado = texto_pag
            if i + 1 < total_paginas:
                texto_combinado += "\n" + doc[i + 1].get_text("text")
            
            # Cortar desde donde aparece la palabra BIBLIOGRAFÍA
            partes = re.split(r'BIBLIOGRAF[ÍI]A\s*:?', texto_combinado, flags=re.IGNORECASE)
            for parte in partes[1:]:
                lineas_validas = []
                for linea in parte.split('\n'):
                    linea_limpia = linea.strip()
                    if not linea_limpia or linea_limpia.isdigit():
                        continue
                    # Si empieza la siguiente materia, dejamos de leer este bloque
                    if any(fin in linea_limpia.upper() for fin in palabras_fin):
                        break
                    lineas_validas.append(linea_limpia)
                
                if lineas_validas:
                    bloques.append("\n".join(lineas_validas))
        i += 1
        
    return bloques

def analizar_con_ia(bloques, clave_api):
    """Envía los bloques de bibliografía a Gemini para separar Título, Autor, Editorial, Año y Edición."""
    client = genai.Client(api_key=clave_api)
    libros_extraidos = []
    
    # Agrupar bloques en paquetes para procesar rápido sin saturar
    lotes = []
    lote_actual = ""
    for b in bloques:
        if len(lote_actual) + len(b) < 10000:
            lote_actual += "\n---\n" + b
        else:
            lotes.append(lote_actual)
            lote_actual = b
    if lote_actual:
        lotes.append(lote_actual)
        
    barra = st.progress(0, text="Analizando bibliografías con Inteligencia Artificial...")
    
    for idx, lote in enumerate(lotes):
        prompt = f"""
        Eres un bibliotecario experto. Extrae todos los libros individuales del siguiente texto de planes de estudio.
        Si en un mismo párrafo hay dos libros pegados, sepáralos en dos registros distintos.
        Ignora frases generales como "Artículos actualizados proporcionados por el profesor".
        
        Devuelve ÚNICAMENTE un arreglo JSON válido con objetos que tengan exactamente estas claves:
        "Titulo", "Autor", "Editorial", "Anio", "Edicion"
        (Si algún dato no aparece en el texto, pon "").
        
        Texto a analizar:
        {lote}
        """
        try:
            respuesta = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={"response_mime_type": "application/json"}
            )
            datos = json.loads(respuesta.text)
            if isinstance(datos, list):
                libros_extraidos.extend(datos)
        except Exception as e:
            st.warning(f"Aviso en bloque {idx+1}: {e}")
            
        barra.progress((idx + 1) / len(lotes), text=f"Procesando bloque {idx+1} de {len(lotes)}...")
        
    barra.empty()
    return libros_extraidos

def eliminar_duplicados_pdf(lista_libros):
    """Crea una lista única sin duplicados comparando Título y Autor."""
    unicos = []
    for libro in lista_libros:
        titulo = str(libro.get("Titulo", "")).strip()
        autor = str(libro.get("Autor", "")).strip()
        if not titulo and not autor:
            continue
            
        t_norm = limpiar_texto(titulo)
        a_norm = limpiar_texto(autor)
        
        es_duplicado = False
        for u in unicos:
            sim_titulo = fuzz.token_set_ratio(t_norm, u["_t_norm"])
            sim_autor = fuzz.token_set_ratio(a_norm, u["_a_norm"])
            if (sim_titulo >= 88 and sim_autor >= 70) or sim_titulo >= 95:
                es_duplicado = True
                # Conservar datos si al primero le faltaba editorial o año
                if not u["Editorial"] and libro.get("Editorial"):
                    u["Editorial"] = libro.get("Editorial")
                if not u["Año"] and libro.get("Anio"):
                    u["Año"] = libro.get("Anio")
                if not u["Edición"] and libro.get("Edicion"):
                    u["Edición"] = libro.get("Edicion")
                break
                
        if not es_duplicado:
            unicos.append({
                "Título": titulo,
                "Autor": autor,
                "Editorial": str(libro.get("Editorial", "")).strip(),
                "Año": str(libro.get("Anio", "")).strip(),
                "Edición": str(libro.get("Edicion", "")).strip(),
                "_t_norm": t_norm,
                "_a_norm": a_norm
            })
            
    df = pd.DataFrame(unicos)
    if not df.empty:
        df = df.drop(columns=["_t_norm", "_a_norm"])
    return df

def leer_excel_inventario(archivo_excel):
    """Encuentra automáticamente la fila donde empiezan los encabezados (fila 1 o fila 11) y limpia los datos."""
    df_raw = pd.read_excel(archivo_excel, header=None)
    fila_encabezado = 0
    
    # Buscar en las primeras 25 filas dónde están las palabras TITULO y AUTOR
    for idx, fila in df_raw.head(25).iterrows():
        valores = [limpiar_texto(val) for val in fila.values]
        if any("titulo" in v for v in valores) and any("autor" in v for v in valores):
            fila_encabezado = idx
            break
            
    df = pd.read_excel(archivo_excel, header=fila_encabezado)
    
    # Identificar las columnas importantes sin importar mayúsculas o acentos
    col_map = {}
    for col in df.columns:
        c_limpia = limpiar_texto(col)
        if "titulo" in c_limpia and "Título" not in col_map:
            col_map["Título"] = col
        elif "autor" in c_limpia and "Autor" not in col_map:
            col_map["Autor"] = col
        elif "clasificacion" in c_limpia and "Clasificación" not in col_map:
            col_map["Clasificación"] = col
            
    if "Título" not in col_map:
        return pd.DataFrame()
        
    df_limpio = pd.DataFrame()
    df_limpio["Título_Inv"] = df[col_map["Título"]].astype(str).str.replace(r'/\s*$', '', regex=True).str.strip()
    df_limpio["Autor_Inv"] = df[col_map["Autor"]].fillna("").astype(str).str.strip() if "Autor" in col_map else ""
    df_limpio["Clasificación_Inv"] = df[col_map["Clasificación"]].fillna("").astype(str).str.strip() if "Clasificación" in col_map else ""
    
    # Quitar filas vacías
    df_limpio = df_limpio[df_limpio["Título_Inv"].notna() & (df_limpio["Título_Inv"] != "nan") & (df_limpio["Título_Inv"] != "")]
    df_limpio["_t_norm"] = df_limpio["Título_Inv"].apply(limpiar_texto)
    df_limpio["_a_norm"] = df_limpio["Autor_Inv"].apply(limpiar_texto)
    return df_limpio

def convertir_a_excel(df):
    """Convierte una tabla de datos en un archivo Excel descargable."""
    salida = BytesIO()
    with pd.ExcelWriter(salida, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Datos")
    return salida.getvalue()

# --- INTERFAZ DE USUARIO PASO A PASO ---
st.header("1️⃣ Paso 1: Subir Plan de Estudios (PDF)")
pdf_subido = st.file_uploader("Arrastra aquí el archivo PDF del Plan de Estudios", type=["pdf"])

if pdf_subido and st.button("🔍 Extraer Bibliografía Única del PDF", type="primary"):
    if not api_key:
        st.error("⚠️ Por favor, pega primero tu API Key de Gemini en el menú de la izquierda.")
    else:
        with st.spinner("Localizando apartados de BIBLIOGRAFÍA en el PDF..."):
            bloques = extraer_bloques_bibliografia(pdf_subido)
        st.info(f"✅ Se encontraron {len(bloques)} secciones de bibliografía en el PDF. Separando datos y eliminando duplicados...")
        
        libros_brutos = analizar_con_ia(bloques, api_key)
        df_unicos = eliminar_duplicados_pdf(libros_brutos)
        st.session_state["df_bibliografia"] = df_unicos

# Mostrar Tabla 1 si ya se extrajo
if "df_bibliografia" in st.session_state and not st.session_state["df_bibliografia"].empty:
    df_bib = st.session_state["df_bibliografia"]
    st.subheader(f"📋 Bibliografía Única Extraída ({len(df_bib)} libros en total)")
    st.caption("💡 Tip: Puedes hacer clic en cualquier celda, presionar Ctrl+A y luego Ctrl+C para copiar toda la tabla, o usar el botón de descarga.")
    
    st.dataframe(df_bib, use_container_width=True, hide_index=True)
    
    st.download_button(
        label="📥 Descargar Bibliografía en Excel (.xlsx)",
        data=convertir_a_excel(df_bib),
        file_name="Bibliografia_Plan_Estudios.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
    st.divider()
    st.header("2️⃣ Paso 2: Subir los Exceles de Inventario y Comparar")
    exceles_subidos = st.file_uploader(
        "Sube aquí los 2 archivos Excel del inventario de la biblioteca", 
        type=["xlsx", "xls"], 
        accept_multiple_files=True
    )
    
    if exceles_subidos and st.button("⚖️ Comparar con Inventario de Biblioteca", type="primary"):
        with st.spinner("Leyendo inventarios y buscando coincidencias por Título y Autor..."):
            listas_inv = [leer_excel_inventario(archivo) for archivo in exceles_subidos]
            df_inventario = pd.concat(listas_inv, ignore_index=True)
            
            resultados = []
            for _, libro in df_bib.iterrows():
                t_pdf = limpiar_texto(libro["Título"])
                a_pdf = limpiar_texto(libro["Autor"])
                
                ejemplares_encontrados = 0
                clasificaciones = set()
                titulos_coincidentes = set()
                
                for _, fila_inv in df_inventario.iterrows():
                    sim_t = fuzz.token_set_ratio(t_pdf, fila_inv["_t_norm"])
                    sim_a = fuzz.token_set_ratio(a_pdf, fila_inv["_a_norm"]) if a_pdf and fila_inv["_a_norm"] else 0
                    
                    # Puntaje combinado dándole peso principal al Título y apoyo al Autor
                    puntaje = (sim_t * 0.75) + (sim_a * 0.25) if (a_pdf and fila_inv["_a_norm"]) else sim_t
                    
                    if puntaje >= umbral_similitud or (sim_t >= 90 and sim_a >= 50):
                        ejemplares_encontrados += 1
                        if fila_inv["Clasificación_Inv"]:
                            clasificaciones.add(fila_inv["Clasificación_Inv"])
                        titulos_coincidentes.add(fila_inv["Título_Inv"])
                
                resultados.append({
                    "Título (Plan de Estudios)": libro["Título"],
                    "Autor": libro["Autor"],
                    "Editorial": libro["Editorial"],
                    "Año": libro["Año"],
                    "Edición": libro["Edición"],
                    "¿Existe en Biblioteca?": "✅ SÍ" if ejemplares_encontrados > 0 else "❌ NO",
                    "Total de Ejemplares": ejemplares_encontrados,
                    "Clasificación en Biblioteca": " | ".join(list(clasificaciones)[:5]),
                    "Título Registrado en Excel": " / ".join(list(titulos_coincidentes)[:2])
                })
                
            st.session_state["df_comparativa"] = pd.DataFrame(resultados)

# Mostrar Tabla 2 (Comparativa Final)
if "df_comparativa" in st.session_state and not st.session_state["df_comparativa"].empty:
    df_comp = st.session_state["df_comparativa"]
    st.subheader("📊 Resultado de la Comparativa con Inventario")
    
    si_hay = len(df_comp[df_comp["Total de Ejemplares"] > 0])
    no_hay = len(df_comp[df_comp["Total de Ejemplares"] == 0])
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Libros Únicos Requeridos", len(df_comp))
    col2.metric("Libros Disponibles en Biblioteca", si_hay)
    col3.metric("Libros Faltantes", no_hay)
    
    st.dataframe(df_comp, use_container_width=True, hide_index=True)
    
    st.download_button(
        label="📥 Descargar Reporte Comparativo en Excel (.xlsx)",
        data=convertir_a_excel(df_comp),
        file_name="Comparativa_Inventario_Biblioteca.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
