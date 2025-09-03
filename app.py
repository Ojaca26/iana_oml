# app.py

import streamlit as st
import pandas as pd
import re
from sqlalchemy import text
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.chains import create_sql_query_chain


# ============================================
# 0) Configuración de la Página y Título
# ============================================
st.set_page_config(page_title="IANA para OML", page_icon="👩‍💻", layout="wide")
st.title("👩‍💻 IANA: Tu Asistente IA para Análisis de Datos")
st.markdown("Soy **IANA**, la red de agentes IA de **OML**. Hazme una pregunta sobre los datos de **Farmacapsulas**.")


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
            llm_sql = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro", temperature=0.1, google_api_key=api_key)
            llm_analista = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro", temperature=0.3, google_api_key=api_key)
            llm_orq = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro", temperature=0.0, google_api_key=api_key)
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
        agent = create_sql_agent(llm=_llm, toolkit=toolkit, verbose=False)
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
    # >> CAMBIO: Eliminada la mención a 'data_farma' que podría ver el usuario.
    prompt_con_instrucciones = f"""
    Tu tarea es generar una consulta SQL para los datos de Farmacapsulas basada en la pregunta del usuario.
    Aquí están las columnas más importantes y sus significados:
    - `FECHA_SOLICITUD`: La fecha en que se solicitó el servicio (DATE).
    - `CATEGORIA_SERVICIO`: La categoría principal del servicio (TEXT).
    - `TIPO`: Un subtipo o clasificación del servicio (TEXT).
    - `CANTIDAD_SERVICIOS`: El conteo de servicios realizados.
    - `TOTAL_HORAS`: El total de horas dedicadas a un servicio.
    REGLA CLAVE: Nunca agregues un 'LIMIT' a la consulta a menos que el usuario lo pida explícitamente.
    Pregunta original del usuario: "{pregunta_usuario}"
    """
    try:
        query_chain = create_sql_query_chain(llm_sql, db)
        sql_query = query_chain.invoke({"question": prompt_con_instrucciones})
        st.code(sql_query.strip(), language='sql')
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
    prompt_sql = (
        "Responde consultando la BD. Devuelve un resultado legible en tabla/resumen. "
        "Responde siempre en español. "
        "Pregunta: "
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
    df_resumen = _df_preview(df, 20)
    prompt_analisis = f"""
    Tu nombre es IANA. Eres un analista de datos senior de OML para su cliente Farmacapsulas.
    Los datos tratan sobre la prestación de servicios, con métricas como `CANTIDAD_SERVICIOS` y `TOTAL_HORAS`, y categorías como `CATEGORIA_SERVICIO` y `TIPO`.
    Responde siempre en español.

    Pregunta original del usuario: {pregunta_usuario}
    Datos/Resultados disponibles para tu análisis:
    TEXTO: {datos_texto}
    TABLA (primeras filas): {df_resumen}
    
    Inicia tu respuesta con un título: "Análisis Ejecutivo de Datos para Farmacapsulas".
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

def orquestador(pregunta_usuario: str):
    with st.expander("⚙️ Ver Proceso de IANA", expanded=False):
        st.info(f"🚀 Recibido: '{pregunta_usuario}'")
        with st.spinner("🔍 IANA está analizando tu pregunta..."):
            clasificacion = clasificar_intencion(pregunta_usuario)
        st.success(f"✅ ¡Intención detectada! Tarea: {clasificacion.upper()}.")

        if clasificacion == "conversacional":
            return responder_conversacion(pregunta_usuario)

        res_datos = obtener_datos_sql(pregunta_usuario)
        resultado = {"tipo": clasificacion, **res_datos, "analisis": None}
        
        # >> CAMBIO: Lógica mejorada para evitar el doble mensaje de error.
        if clasificacion == "analista":
            if res_datos.get("df") is not None and not res_datos["df"].empty:
                analisis = analizar_con_datos(pregunta_usuario, res_datos.get("texto", ""), res_datos["df"])
                resultado["analisis"] = analisis
            else:
                # Si el análisis falla por falta de datos, mostramos un solo mensaje claro.
                resultado["texto"] = "Para poder realizar un análisis, primero necesito datos. Por favor, haz una pregunta más específica para obtener la información que quieres analizar."
                resultado["df"] = None # Nos aseguramos de que no haya tabla de datos
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
            res = orquestador(prompt)
            
            st.markdown(f"### IANA responde a: '{prompt}'")
            # La lógica de visualización ahora es más simple
            if res.get("df") is not None and not res["df"].empty:
                st.dataframe(res["df"])
            
            if res.get("texto"):
                 st.markdown(res["texto"])
            
            if res.get("analisis"):
                st.markdown("---")
                st.markdown("### 🧠 Análisis de IANA para Farmacapsulas")
                st.markdown(res["analisis"])
                
            st.session_state.messages.append({"role": "assistant", "content": res})
