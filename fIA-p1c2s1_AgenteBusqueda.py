# ==============================================================================
# APP AGENTE DE BUSQUEDA CON LANGGRAPH
# Última Actualización: 2026.09.19 | V2.0
# 
# * La librería google-generativeai fue deprecada el 30.11.2025.
# * La librería google-genai es la actualización para este código.
# * Módulo 3: Agentes IA con LangGraph - ONE AI FOR TECH
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. Conectando librerías LLM (LangChain, LangGraph)
# ------------------------------------------------------------------------------

# 1.1 Instalando librerías necesarias en la aplicación
%pip install -q \
langchain \
langchain-core \
langchain-community \
langchain-google-genai \
langgraph \
arxiv

# 1.2 Importando módulos, mapeando librerías y credenciales
import os
from google.colab import userdata

# Obtener la API key desde los Secrets de Colab
api_key = userdata.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError(
        "No se encontró GEMINI_API_KEY. Abrí el ícono de llave (Secrets) "
        "en la barra lateral, creá el secreto y activá 'Acceso al notebook'."
    )

# Asignar la clave a la variable de entorno que consumen los integradores
os.environ["GOOGLE_API_KEY"] = api_key


# 1.3 Inicializando el modelo LLM
# Instanciación del modelo de lenguaje dentro de la estructura de LangChain.
# gemini-3.6-flash usa sampling fijo: no acepta temperature ni top_p.
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")


# 1.4 Funciones auxiliares para renderizado y aplanado de texto
from IPython.display import Markdown, display

def extraer_texto(respuesta):
    """Aplana el content de un AIMessage.
    
    Los modelos con razonamiento devuelven una lista de bloques
    ({'type': 'text', ...}, firmas de pensamiento, etc.).
    Los modelos clásicos devuelven un string plano.
    """
    if isinstance(respuesta.content, str):
        return respuesta.content

    return "".join(
        bloque["text"]
        for bloque in respuesta.content
        if isinstance(bloque, dict) and bloque.get("type") == "text"
    )

def mostrar(texto):
    """Renderiza Markdown en la salida de Colab."""
    display(Markdown(texto))


# 1.5 Diseño del PromptTemplate
# Se construye una plantilla reutilizable definiendo explícitamente las variables de entrada dinámicas.
from langchain_core.prompts import PromptTemplate

prompt_template = (
    "Eres un experto en {tema} y debes entregar una explicación clara "
    "y actualizada sobre los impactos de la Inteligencia Artificial (IA) "
    "en esta área."
)

modelo_de_prompt = PromptTemplate(
    input_variables=["tema"],
    template=prompt_template,
)


# 1.6 Ejecución de la cadena
# Se invoca la ejecución pasando un diccionario con el valor requerido por la plantilla.
cadena = modelo_de_prompt | llm
respuesta = cadena.invoke({"tema": "educación"})
mostrar(extraer_texto(respuesta))


# 1.7 Mejora de texto plano para ser almacenado (Parser)
from langchain_core.output_parsers import StrOutputParser

# El parser aplana los bloques automáticamente: ya no hace falta extraer_texto
cadena_texto = modelo_de_prompt | llm | StrOutputParser()

mostrar(cadena_texto.invoke({"tema": "educación"}))


# 1.8 Ingresándole otros temas al prompt en bucle
for tema in ["educación", "salud", "logística"]:
    mostrar(f"## Tema: {tema}")
    mostrar(cadena_texto.invoke({"tema": tema}))
