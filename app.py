# app.py

import streamlit as st
import pandas as pd
import re
import os
from sqlalchemy import text
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.chains import create_sql_query_chain
from langchain_google_vertexai import ChatVertexAI

# ============================================
# 0) Configuración de la Página y Título
# ============================================
st.set_page_config(page_title="IANA para OML", page_icon="logo.png", layout="wide")

# Creamos columnas para alinear el logo y el título
col1, col2 = st.columns([1, 4]) 

with col1:
    # Simplemente usa el nombre del archivo local
    st.image("logo.png", width=120)

with col2:
    st.title("IANA: Tu Asistente IA para Análisis de Datos")
    st.markdown("Soy la red de agentes IA de **OML**. Hazme una pregunta sobre los datos de **Farmacapsulas**.")

# ============================================
# 1) Conexión a la Base de Datos y LLMs (con caché para eficiencia)
# ============================================

@st.cache_resource
def get_database_connection():
    """Establece y cachea la conexión a la base de datos."""
    with st.spinner("🔌 Conectando a la base de datos de Farmacapsulas..."):
        try:
            db_user = st.secrets["db_credentials"]["user"]
            db_pass = st.secrets["db_credentials"]["password"]
            db_host = st.secrets["db_credentials"]["host"]
            db_name = st.secrets["db_credentials"]["database"]
            uri = f"mysql+pymysql://{db_user}:{db_pass}@{db_host}/{db_name}"
            # Nota: La tabla se especifica aquí para la conexión, pero no se mencionará al usuario.
            db = SQLDatabase.from_uri(uri, include_tables=["data_farma"])
            st.success("✅ Conexión a la base de datos establecida.")
            return db
        except Exception as e:
            st.error(f"Error al conectar a la base de datos: {e}")
            return None

@st.cache_resource
def get_llms():
    """Inicializa y cachea los modelos de lenguaje."""
    with st.spinner("🧠 Inicializando la red de agentes IANA..."):
        try:
            api_key = st.secrets["google_api_key"]
            project_id = st.secrets["google_project_id"]
            os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
    
            llm_sql = ChatGoogleGenerativeAI(
                model="gemini-pro",
                temperature=0.1,
                google_api_key=api_key,
                location="us-central1"  # <-- AÑADE ESTO
            )
            llm_analista = ChatGoogleGenerativeAI(
                model="gemini-pro",
                temperature=0.1,
                google_api_key=api_key,
                location="us-central1"  # <-- AÑADE ESTO
            )
            llm_orq = ChatGoogleGenerativeAI(
                model="gemini-pro",
                temperature=0.0,
                google_api_key=api_key,
                location="us-central1"  # <-- AÑADE ESTO
            )
            
            st.success("✅ Agentes de IANA listos.")
            return llm_sql, llm_analista, llm_orq
        
        except Exception as e:
            st.error(f"Error al inicializar los LLMs. Asegúrate de que tu API key es correcta. Error: {e}")
            return None, None, None

db = get_database_connection()
llm_sql, llm_analista, llm_orq = get_llms()

@st.cache_resource
def get_sql_agent(_llm, _db):
    """Crea y cachea el agente SQL."""
    if not _llm or not _db:
        return None
    with st.spinner("🛠️ Configurando agente SQL de IANA..."):
        toolkit = SQLDatabaseToolkit(db=_db, llm=_llm)
        agent = create_sql_agent(
            llm=_llm, 
            toolkit=toolkit, 
            verbose=False,
            top_k=1000)
        st.success("✅ Agente SQL configurado.")
        return agent

agente_sql = get_sql_agent(llm_sql, db)

# ============================================
# 2) Funciones de Agentes (Lógica Principal)
# ============================================

def markdown_table_to_df(texto: str) -> pd.DataFrame:
    """Convierte una tabla en formato Markdown a un DataFrame de pandas."""
    lineas = [l.strip() for l in texto.splitlines() if l.strip().startswith('|')]
    if not lineas: return pd.DataFrame()
    lineas = [l for l in lineas if not re.match(r'^\|\s*-', l)]
    filas = [[c.strip() for c in l.strip('|').split('|')] for l in lineas]
    if len(filas) < 2: return pd.DataFrame()
    header, data = filas[0], filas[1:]
    df = pd.DataFrame(data, columns=header)
    for c in df.columns:
        s = df[c].astype(str).str.replace(',', '', regex=False).str.replace(' ', '', regex=False)
        try: df[c] = pd.to_numeric(s)
        except Exception: df[c] = s
    return df

def _df_preview(df: pd.DataFrame, n: int = 20) -> str:
    """Crea un preview en texto de un DataFrame."""
    if df is None or df.empty: return ""
    try: return df.head(n).to_markdown(index=False)
    except Exception: return df.head(n).to_string(index=False)


def ejecutar_sql_real(pregunta_usuario: str):
    st.info("🤖 Entendido. El agente de datos de IANA está traduciendo tu pregunta a SQL...")
    
    # >> CAMBIO: Instrucciones más claras sobre cómo agrupar los datos.
    prompt_con_instrucciones = f"""
    Tu tarea es generar una consulta SQL para los datos de Farmacapsulas.
    
    Presta mucha atención a palabras como 'diariamente', 'mensual', 'por tipo', etc. La cláusula GROUP BY debe coincidir exactamente con lo que pide el usuario. 
    Si el usuario pide un total 'diario', solo debes agrupar por la fecha. Si pide 'por tipo', solo agrupa por el tipo de servicio.

    Columnas importantes:
    - `FECHA_SOLICITUD`: Fecha del servicio.
    - `CATEGORIA_SERVICIO`, `TIPO`: Categorías para agrupar.
    - `CANTIDAD_SERVICIOS`, `TOTAL_HORAS`: Valores numéricos para sumar o promediar.

    REGLA CLAVE: Nunca agregues un 'LIMIT'.
    Pregunta original: "{pregunta_usuario}"
    """
    try:
        query_chain = create_sql_query_chain(llm_sql, db)
        sql_query = query_chain.invoke({"question": prompt_con_instrucciones})
        
        # >> CAMBIO: Limpiamos la consulta para evitar errores de sintaxis
        sql_query = re.sub(r"^\s*```sql\s*|\s*```\s*$", "", sql_query, flags=re.IGNORECASE).strip()

        st.code(sql_query, language='sql')
        with st.spinner("⏳ Ejecutando la consulta en la base de datos..."):
            with db._engine.connect() as conn:
                df = pd.read_sql(text(sql_query), conn)
        st.success("✅ ¡Consulta ejecutada!")
        return {"sql": sql_query, "df": df}
    except Exception as e:
        st.warning(f"❌ Error en la consulta directa. Intentando un método alternativo... Error: {e}")
        return {"sql": None, "df": None, "error": str(e)}


def ejecutar_sql_en_lenguaje_natural(pregunta_usuario: str):
    st.info("🤔 La consulta directa falló. Activando el agente SQL experto de IANA como plan B.")
    
    # >> CAMBIO CLAVE: El prompt ahora es mucho más estricto y prohíbe resúmenes.
    prompt_sql = (
        "Tu tarea es responder la pregunta del usuario consultando la base de datos. "
        "Debes devolver ÚNICAMENTE una tabla de datos en formato Markdown. "
        "REGLA CRÍTICA: Devuelve SIEMPRE TODAS las filas de datos que encuentres. NUNCA resumas, trunques ni expliques los resultados. No agregues texto como 'Se muestran las 10 primeras filas' o 'Aquí está la tabla'. "
        "Responde siempre en español. "
        "Pregunta del usuario: "
        f"{pregunta_usuario}"
    )
    try:
        with st.spinner("💬 Pidiendo al agente SQL que responda en lenguaje natural..."):
            res = agente_sql.invoke(prompt_sql)
            texto = res["output"] if isinstance(res, dict) and "output" in res else str(res)
        st.info("📝 Recibí una respuesta en texto. Intentando convertirla en una tabla de datos...")
        df_md = markdown_table_to_df(texto)
        return {"texto": texto, "df": df_md}
    except Exception as e:
        st.error(f"❌ El agente SQL experto también encontró un problema: {e}")
        return {"texto": f"[SQL_ERROR] {e}", "df": pd.DataFrame()}


def analizar_con_datos(pregunta_usuario: str, datos_texto: str, df: pd.DataFrame | None):
    st.info("\n🧠 Ahora, el analista experto de IANA está examinando los datos...")
    
    # >> CAMBIO CLAVE: Se añaden reglas de formato estrictas al prompt del analista.
    prompt_analisis = f"""
    Tu nombre es IANA. Eres un analista de datos senior de OML para su cliente Farmacapsulas.
    Tu tarea es generar un análisis ejecutivo, breve y fácil de leer para un gerente.
    Responde siempre en español.

    REGLAS DE FORMATO MUY IMPORTANTES:
    1.  Inicia con el título: "Análisis Ejecutivo de Datos para Farmacapsulas".
    2.  Debajo del título, presenta tus conclusiones como una lista de ítems (viñetas con markdown `-`).
    3.  Cada ítem debe ser una oración corta, clara y directa al punto.
    4.  Limita el análisis a un máximo de 5 ítems clave; si el cliente especifica una cantidad de ítems, genera el número exacto que pidió.
    5.  No escribas párrafos largos.

    Pregunta del usuario: {pregunta_usuario}
    Datos disponibles para tu análisis:
    {_df_preview(df, 20)}

    Ahora, genera el análisis siguiendo estrictamente las reglas de formato.
    """
    with st.spinner("💡 Generando análisis y recomendaciones..."):
        analisis = llm_analista.invoke(prompt_analisis).content
    st.success("💡 ¡Análisis completado!")
    return analisis

def responder_conversacion(pregunta_usuario: str):
    """Activa el modo conversacional de IANA."""
    st.info("💬 Activando modo de conversación...")
    # >> CAMBIO: Eliminada la mención a 'data_farma'.
    prompt_personalidad = f"""
    Tu nombre es IANA, una asistente de IA de OML para su cliente Farmacapsulas.
    Tu personalidad es amable, servicial y profesional.
    Tu objetivo principal es ayudar a analizar los datos de Farmacapsulas.
    Ejemplos de lo que puedes hacer es: "Puedo contar cuántos servicios se hicieron por mes", "puedo analizar las horas totales por tipo de servicio", etc.
    NO intentes generar código SQL. Solo responde de forma conversacional.
    Responde siempre en español.

    Pregunta del usuario: "{pregunta_usuario}"
    """
    respuesta = llm_analista.invoke(prompt_personalidad).content
    # Usamos la clave "texto" para la respuesta principal y "analisis" como nulo.
    return {"texto": respuesta, "df": None, "analisis": None}

# --- Orquestador Principal ---

def clasificar_intencion(pregunta: str) -> str:
    # >> CAMBIO: Prompt mejorado para clasificar mejor las preguntas generales.
    prompt_orq = f"""
    Devuelve UNA sola palabra exacta según la intención del usuario:
    - `consulta`: si pide extraer, filtrar o contar datos específicos. (Ej: 'cuántos servicios en abril?')
    - `analista`: si pide interpretar, resumir o recomendar acciones sobre datos. (Ej: 'analiza las tendencias')
    - `conversacional`: si es un saludo, una pregunta general sobre tus capacidades (Ej: '¿qué puedes hacer?' o '¿cómo me puedes ayudar?'), o no está relacionada con datos específicos.
    Mensaje: {pregunta}
    """
    clasificacion = llm_orq.invoke(prompt_orq).content.strip().lower().replace('"', '').replace("'", "")
    return clasificacion

def obtener_datos_sql(pregunta_usuario: str) -> dict:
    res_real = ejecutar_sql_real(pregunta_usuario)
    if res_real.get("df") is not None and not res_real["df"].empty:
        return {"sql": res_real["sql"], "df": res_real["df"], "texto": None}
    
    res_nat = ejecutar_sql_en_lenguaje_natural(pregunta_usuario)
    return {"sql": None, "df": res_nat["df"], "texto": res_nat["texto"]}


def orquestador(pregunta_usuario: str, chat_history: list):
    with st.expander("⚙️ Ver Proceso de IANA", expanded=False):
        st.info(f"🚀 Recibido: '{pregunta_usuario}'")
        with st.spinner("🔍 IANA está analizando tu pregunta..."):
            clasificacion = clasificar_intencion(pregunta_usuario)
        st.success(f"✅ ¡Intención detectada! Tarea: {clasificacion.upper()}.")

        if clasificacion == "conversacional":
            return responder_conversacion(pregunta_usuario)

        # --- INICIO DE LA LÓGICA DE MEMORIA CORREGIDA ---
        if clasificacion == "analista":
            palabras_clave_contexto = [
                "esto", "esos", "esa", "información", 
                "datos", "tabla", "anterior", "acabas de dar"
            ]
            # Usamos una expresión más simple para verificar si la pregunta es sobre el contexto
            es_pregunta_de_contexto = any(palabra in pregunta_usuario.lower() for palabra in palabras_clave_contexto)

            # >> CORRECCIÓN CLAVE: Ahora miramos el penúltimo mensaje (la respuesta anterior de IANA)
            # y nos aseguramos de que el historial tenga al menos 2 mensajes.
            if es_pregunta_de_contexto and len(chat_history) > 1:
                mensaje_anterior = chat_history[-2] # El penúltimo mensaje
                
                if mensaje_anterior["role"] == "assistant" and "df" in mensaje_anterior["content"]:
                    df_contexto = mensaje_anterior["content"]["df"]
                    
                    if df_contexto is not None and not df_contexto.empty:
                        st.info("💡 Usando datos de la conversación anterior para el análisis...")
                        analisis = analizar_con_datos(pregunta_usuario, "Datos de la tabla anterior.", df_contexto)
                        # Devolvemos la tabla anterior junto con el nuevo análisis para mantenerla visible
                        return {"tipo": "analista", "df": df_contexto, "texto": None, "analisis": analisis}
        # --- FIN DE LA LÓGICA DE MEMORIA CORREGIDA ---

        # Si no es una pregunta de contexto, sigue el flujo normal
        res_datos = obtener_datos_sql(pregunta_usuario)
        resultado = {"tipo": clasificacion, **res_datos, "analisis": None}
        
        if clasificacion == "analista":
            if res_datos.get("df") is not None and not res_datos["df"].empty:
                analisis = analizar_con_datos(pregunta_usuario, res_datos.get("texto", ""), res_datos["df"])
                resultado["analisis"] = analisis
            else:
                resultado["texto"] = "Para poder realizar un análisis, primero necesito datos. Por favor, haz una pregunta más específica para obtener la información que quieres analizar."
                resultado["df"] = None
    return resultado

# ============================================
# 3) Interfaz de Chat de Streamlit
# ============================================

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": {"texto": "¡Hola! Soy IANA, tu asistente de IA de OML. Estoy lista para analizar los datos de Farmacapsulas. ¿Qué te gustaría saber?"}}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "texto" in message["content"] and message["content"]["texto"]: st.markdown(message["content"]["texto"])
        if "df" in message["content"] and message["content"]["df"] is not None: st.dataframe(message["content"]["df"])
        if "analisis" in message["content"] and message["content"]["analisis"]: st.markdown(message["content"]["analisis"])

if prompt := st.chat_input("Pregúntale a IANA sobre los datos de Farmacapsulas..."):
    if not all([db, llm_sql, llm_analista, llm_orq, agente_sql]):
        st.error("La aplicación no está completamente inicializada. Revisa los errores de conexión o de API key.")
    else:
        st.session_state.messages.append({"role": "user", "content": {"texto": prompt}})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # >> CAMBIO AQUÍ: Pasa el historial de mensajes al orquestador
            res = orquestador(prompt, st.session_state.messages)
            
            st.markdown(f"### IANA responde a: '{prompt}'")
            if res.get("df") is not None and not res["df"].empty:
                st.dataframe(res["df"])
            
            if res.get("texto"):
                 st.markdown(res["texto"])
            
            if res.get("analisis"):
                st.markdown("---")
                st.markdown("### 🧠 Análisis de IANA para Farmacapsulas")
                st.markdown(res["analisis"])
                
            st.session_state.messages.append({"role": "assistant", "content": res})

















