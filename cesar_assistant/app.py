import json
import os
import re
import importlib
from datetime import datetime, timezone
from functools import lru_cache

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gradio as gr
import httpx
import pandas as pd
from dotenv import load_dotenv

from python_mermaid.diagram import MermaidDiagram, Node, Link
from rag_manager import RAGManager

load_dotenv()

MASTER_PROMPT = """Role: You are an advanced Analytical AI designed for deep reasoning, NLP tasks, data structuring, and business assistance.

Operational Directives:
- Language: Respond in Spanish by default unless the user explicitly asks for another language. Do not translate code unless asked.
- Deep Thinking: Break down complex requests step-by-step before answering.
- Data Generation: If the user asks for data, statistics, tables, or charts, you MUST format your response as strictly valid JSON.
- Personality NLP: Infer communication style and business behavior patterns from the user's wording, but present them as hypotheses rather than facts.
- Business Context: Consider persistent profile memory and recent conversations when available.
- JSON Output Schema (use only when data or charts are requested):
{
  "response_text": "Your explanation in Spanish.",
  "personality_profile": "Optional short reading of communication style and likely personality traits.",
  "has_data": true,
  "table_data": {
    "columns": ["Header1", "Header2"],
    "rows": [["Row1Data1", "Row1Data2"], ["Row2Data1", "Row2Data2"]]
  },
  "plot_config": {
    "title": "Chart Title",
    "x_label": "X Axis Label",
    "y_label": "Y Axis Label",
    "type": "bar"
  }
}
- Never use markdown code fences around JSON.
"""

CONVERSATIONAL_PROMPT = """Role: You are Cesar Assistant in conversational NLP mode.

Operational Directives:
- Respond in Spanish by default unless the user asks for another language.
- Maintain a warm, useful, business-ready conversation.
- Detect communication style, urgency, tolerance for detail, decision style, risk posture, and main business focus.
- Treat personality recognition as a probabilistic reading, not a diagnosis or certainty.
- Adapt your tone to the user while remaining clear, practical, and respectful.
- If the user asks for data, tables, or charts, you may still return valid JSON using the shared schema.
- Never use markdown code fences around JSON.
"""

MODEL_FALLBACKS = [
    "models/gemini-2.5-flash",
    "models/gemini-flash-latest",
    "models/gemini-2.0-flash",
    "models/gemini-2.0-flash-001",
    "models/gemini-2.0-flash-lite",
    "models/gemini-pro-latest",
]

PROFILE_TABLE = os.getenv("SUPABASE_PROFILE_TABLE", "assistant_profiles")
CONVERSATION_TABLE = os.getenv("SUPABASE_CONVERSATION_TABLE", "assistant_conversations")
DEFAULT_PROFILE_ID = os.getenv("CESAR_PROFILE_ID", "cesar-mora")


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def slugify_profile_id(raw_value):
    cleaned = (raw_value or DEFAULT_PROFILE_ID).strip().lower()
    cleaned = re.sub(r"[^a-z0-9_-]+", "-", cleaned)
    cleaned = re.sub(r"-+", "-", cleaned).strip("-")
    return cleaned or DEFAULT_PROFILE_ID


def normalize_model_name(model_name):
    if not model_name:
        return None
    return model_name if model_name.startswith("models/") else f"models/{model_name}"


def strip_model_prefix(model_name):
    if not model_name:
        return ""
    return model_name.replace("models/", "", 1)


def count_matches(lowered_text, terms):
    return sum(lowered_text.count(term) for term in terms)


def safe_json_dumps(value):
    return json.dumps(value, ensure_ascii=False)


def build_master_prompt(
    who_are_you,
    what_do_you_want,
    objectives,
    problems,
    kpis,
    tasks,
    schedule,
    integrations,
    financial_objectives,
    financial_pain_points,
    budget_range,
    revenue_sources,
    cost_structure,
    reports_needed,
    risk_tolerance,
    tone,
    constraints,
):
    sections = [
        "Rol: Eres Cesar Assistant, un agente analitico y operativo creado para Cesar Mora.",
        f"Perfil del usuario: {who_are_you}",
        f"Resultado deseado: {what_do_you_want}",
        f"Objetivos principales: {objectives}",
        f"Problemas a resolver: {problems}",
        f"KPIs o metricas de exito: {kpis}",
        f"Tareas y rutinas: {tasks}",
        f"Rutinas programadas: {schedule}",
        f"Integraciones necesarias: {integrations}",
        f"Objetivos financieros: {financial_objectives}",
        f"Dolores financieros actuales: {financial_pain_points}",
        f"Rango de presupuesto: {budget_range}",
        f"Fuentes de ingreso o unidades de negocio: {revenue_sources}",
        f"Estructura de costos o gastos clave: {cost_structure}",
        f"Reportes o tableros requeridos: {reports_needed}",
        f"Tolerancia al riesgo financiero: {risk_tolerance}",
        f"Tono preferido y restricciones: {tone} | {constraints}",
        "Idioma principal de respuesta: espanol, excepto cuando el usuario pida otro idioma o un bloque de codigo especifico.",
        "Arquitectura sugerida: Gradio como interfaz, Gemini como motor analitico, pandas para tablas, matplotlib para graficos, y Supabase para memoria persistente de perfiles y conversaciones.",
        "SDLC sugerido: discovery de requerimientos, diseno de flujos, desarrollo iterativo, pruebas con casos reales, despliegue con variables de entorno seguras y mejora recursiva basada en feedback.",
        "Prompt maestro operativo: piensa paso a paso, responde con precision, usa JSON valido cuando el usuario pida datos o graficos, y conserva contexto persistente para mejorar decisiones futuras.",
    ]
    return "\n\n".join(sections)


@lru_cache(maxsize=1)
def get_model_candidates():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None, None, "Falta GEMINI_API_KEY en el archivo .env."
    try:
        genai_module = importlib.import_module("google.genai")
    except ImportError:
        return None, None, "Falta instalar google-genai."

    client = genai_module.Client(api_key=api_key)
    available_models = []
    for model in client.models.list():
        supported_actions = getattr(model, "supported_actions", []) or []
        if "generateContent" in supported_actions:
            available_models.append(model.name)

    if not available_models:
        return client, [], "No hay modelos Gemini compatibles con generateContent para esta clave."

    preferred_models = []
    env_model = normalize_model_name(os.getenv("GEMINI_MODEL"))
    if env_model:
        preferred_models.append(env_model)
    preferred_models.extend(MODEL_FALLBACKS)

    candidates = []
    seen_models = set()
    for model_name in preferred_models + available_models:
        normalized_name = normalize_model_name(model_name)
        if normalized_name in available_models and normalized_name not in seen_models:
            candidates.append(normalized_name)
            seen_models.add(normalized_name)

    return client, candidates, None


@lru_cache(maxsize=1)
def get_supabase_client():
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = (
        os.getenv("SUPABASE_KEY")
        or os.getenv("SUPABASE_SERVICE_KEY")
        or os.getenv("SUPABASE_ANON_KEY")
    )
    if not supabase_url or not supabase_key:
        return None, "Faltan SUPABASE_URL y SUPABASE_SERVICE_KEY, SUPABASE_ANON_KEY o SUPABASE_KEY."
    return {
        "rest_url": f"{supabase_url.rstrip('/')}/rest/v1",
        "headers": {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
            "Content-Type": "application/json",
        },
    }, None


def supabase_request(config, method, table_name, params=None, payload=None, prefer=None):
    headers = dict(config["headers"])
    if prefer:
        headers["Prefer"] = prefer
    response = httpx.request(
        method,
        f"{config['rest_url']}/{table_name}",
        headers=headers,
        params=params,
        json=payload,
        timeout=20.0,
    )
    response.raise_for_status()
    if not response.text:
        return []
    return response.json()


def format_supabase_error(error):
    message = str(error)
    lowered = message.lower()
    if "404 not found" in lowered and (PROFILE_TABLE in lowered or CONVERSATION_TABLE in lowered):
        return (
            "Supabase conectado con SUPABASE_KEY, pero faltan las tablas de memoria. "
            f"Ejecuta el esquema en {PROFILE_TABLE} y {CONVERSATION_TABLE} usando el archivo supabase_schema.sql."
        )
    if "401" in lowered or "403" in lowered:
        return "Supabase respondio con permisos insuficientes. Revisa SUPABASE_KEY y las politicas RLS."
    return f"Error de Supabase: {error}"


def is_model_compatibility_error(error):
    message = str(error).lower()
    return (
        "not found for api version" in message
        or "is not supported for generatecontent" in message
        or "404 models/" in message
        or ("404" in message and "model" in message)
    )


def build_model_choices():
    _, candidates, _ = get_model_candidates()
    if candidates:
        return [strip_model_prefix(candidate) for candidate in candidates]
    return [strip_model_prefix(candidate) for candidate in MODEL_FALLBACKS]


def analyze_personality_signals(user_text):
    text = (user_text or "").strip()
    lowered = text.lower()
    if not text:
        return {
            "summary": "Sin suficiente texto para estimar estilo de comunicacion.",
            "traits": {},
            "business_rules": [],
        }

    direct_score = count_matches(lowered, ["quiero", "haz", "necesito", "directo", "rapido", "resumen", "ejecuta"])
    analytical_score = count_matches(lowered, ["analiza", "compara", "datos", "tabla", "grafico", "estrategia", "proceso", "indicador"])
    warm_score = count_matches(lowered, ["gracias", "por favor", "ayuda", "hola", "amable", "equipo", "acompanar"])
    urgency_score = count_matches(lowered, ["urgente", "ahora", "hoy", "ya", "inmediato", "asap"]) + text.count("!")
    detail_score = count_matches(lowered, ["detalle", "explica", "paso a paso", "profundo", "desglosa"])
    executive_score = count_matches(lowered, ["breve", "resumen", "puntos", "concreto", "accionable"])
    risk_score = count_matches(lowered, ["seguro", "riesgo", "prudente", "control", "garantia"])
    aggressive_score = count_matches(lowered, ["crecer", "acelerar", "agresivo", "dominar", "maximizar"])
    finance_score = count_matches(lowered, ["flujo de caja", "margen", "rentabilidad", "costos", "presupuesto", "ingresos", "gastos"])
    operations_score = count_matches(lowered, ["proceso", "operacion", "logistica", "sistema", "rutina", "automatiza"])
    sales_score = count_matches(lowered, ["ventas", "cliente", "pipeline", "conversion", "prospecto", "oferta"])
    people_score = count_matches(lowered, ["equipo", "persona", "liderazgo", "colaboracion", "comunicacion", "delegar"])

    communication_style = "equilibrado"
    if direct_score >= max(analytical_score, warm_score) and direct_score >= 2:
        communication_style = "directivo"
    elif analytical_score >= max(direct_score, warm_score) and analytical_score >= 2:
        communication_style = "analitico"
    elif warm_score >= 2:
        communication_style = "colaborativo"

    detail_preference = "equilibrada"
    if detail_score > executive_score and detail_score >= 2:
        detail_preference = "detallada"
    elif executive_score >= 2:
        detail_preference = "ejecutiva"

    decision_style = "balanceado"
    if analytical_score >= 2:
        decision_style = "basado en datos"
    elif direct_score >= 2 or urgency_score >= 2:
        decision_style = "rapido y orientado a accion"

    tone_preference = "profesional cercano"
    if warm_score >= 2:
        tone_preference = "cercano y colaborativo"
    elif direct_score >= 2:
        tone_preference = "directo y eficiente"
    elif analytical_score >= 2:
        tone_preference = "tecnico y estructurado"

    risk_posture = "balanceada"
    if risk_score > aggressive_score and risk_score >= 2:
        risk_posture = "conservadora"
    elif aggressive_score > risk_score and aggressive_score >= 2:
        risk_posture = "expansiva"

    business_focus = "general"
    scores = {
        "finanzas": finance_score,
        "operaciones": operations_score,
        "ventas": sales_score,
        "equipo": people_score,
    }
    focus_key = "general"
    focus_score = 0
    for candidate, candidate_score in scores.items():
        if candidate_score > focus_score:
            focus_key = candidate
            focus_score = candidate_score
    if focus_score >= 1:
        business_focus = focus_key

    urgency_level = "alta" if urgency_score >= 3 else "media" if urgency_score >= 1 else "baja"

    business_rules = []
    if communication_style == "directivo":
        business_rules.append("Responder primero con acciones concretas y decisiones recomendadas.")
    if communication_style == "analitico":
        business_rules.append("Incluir criterios, comparativas y trade-offs antes de concluir.")
    if detail_preference == "ejecutiva":
        business_rules.append("Entregar resumen ejecutivo antes del detalle.")
    if risk_posture == "conservadora":
        business_rules.append("Priorizar control de riesgo, validacion y escenarios defensivos.")
    if business_focus == "finanzas":
        business_rules.append("Resaltar impacto en caja, margen, costos y retorno esperado.")
    if business_focus == "operaciones":
        business_rules.append("Resaltar eficiencia, SLA, cuellos de botella y automatizacion.")
    if business_focus == "ventas":
        business_rules.append("Priorizar conversion, pipeline, oferta y seguimiento comercial.")
    if business_focus == "equipo":
        business_rules.append("Considerar coordinacion, liderazgo y alineacion de equipo.")

    traits = {
        "estilo_comunicacion": communication_style,
        "preferencia_detalle": detail_preference,
        "estilo_decision": decision_style,
        "tono_preferido": tone_preference,
        "postura_riesgo": risk_posture,
        "foco_negocio": business_focus,
        "urgencia": urgency_level,
    }
    summary = (
        f"Estilo: {communication_style}. "
        f"Detalle: {detail_preference}. "
        f"Decision: {decision_style}. "
        f"Tono: {tone_preference}. "
        f"Riesgo: {risk_posture}. "
        f"Foco negocio: {business_focus}. "
        f"Urgencia: {urgency_level}."
    )
    return {
        "summary": summary,
        "traits": traits,
        "business_rules": business_rules,
    }


def format_personality_analysis(analysis):
    if not analysis or not analysis.get("summary"):
        return "Sin lectura de personalidad disponible."
    lines = [analysis["summary"]]
    if analysis.get("business_rules"):
        lines.append("Reglas sugeridas: " + " | ".join(analysis["business_rules"]))
    return "\n".join(lines)


def build_request_contents(history, user_message, persistent_summary):
    contents = []
    if persistent_summary:
        contents.append(
            {
                "role": "user",
                "parts": [
                    {
                        "text": "Contexto persistente de perfil y conversaciones previas para usar como referencia: "
                        + persistent_summary
                    }
                ],
            }
        )
    for message in history or []:
        role = "model" if message.get("role") == "assistant" else "user"
        content = message.get("content", "")
        if content:
            contents.append({"role": role, "parts": [{"text": content}]})
    contents.append({"role": "user", "parts": [{"text": user_message}]})
    return contents


def build_generation_config(chat_mode, personality_analysis, persistent_summary, temperature=None):
    types_module = importlib.import_module("google.genai.types")
    system_prompt = MASTER_PROMPT if chat_mode == "Analitica" else CONVERSATIONAL_PROMPT
    
    # Determinar temperatura por defecto si no se pasa
    if temperature is None:
        temperature = 0.0 if chat_mode == "Analitica" else 0.7

    if personality_analysis.get("summary") and chat_mode == "Conversacional":
        system_prompt += f"\n- Hipotesis actual de personalidad y estilo: {personality_analysis['summary']}"
    if personality_analysis.get("business_rules"):
        system_prompt += "\n- Reglas de respuesta sugeridas: " + " | ".join(personality_analysis["business_rules"])
    if persistent_summary:
        system_prompt += f"\n- Memoria persistente disponible: {persistent_summary}"
    
    return types_module.GenerateContentConfig(
        systemInstruction=system_prompt,
        temperature=temperature,
        top_p=0.95,
        max_output_tokens=2048
    )


def resolve_model_order(selected_model, model_candidates):
    normalized_selected = normalize_model_name(selected_model)
    ordered_candidates = []
    seen_models = set()
    if normalized_selected and normalized_selected in model_candidates:
        ordered_candidates.append(normalized_selected)
        seen_models.add(normalized_selected)
    for candidate in model_candidates:
        if candidate not in seen_models:
            ordered_candidates.append(candidate)
            seen_models.add(candidate)
    return ordered_candidates


def parse_structured_output(output_text):
    """
    Detecta y extrae JSON incrustado en la respuesta (con o sin bloques de código markdown).
    Prioriza bloques de código ```json o ``` y luego busca estructuras {}.
    """
    try:
        # 1. Buscar bloques de código markdown ```json o ``` 
        code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", output_text, re.DOTALL)
        if code_block_match:
            return json.loads(code_block_match.group(1))
        
        # 2. Si no hay bloques, buscar la primera estructura JSON bruta { ... }
        json_match = re.search(r"(\{.*\})", output_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                # Si falló, puede que haya texto después del JSON, intentar búsqueda no-greedy
                json_match_minimal = re.search(r"(\{.*?\})", output_text, re.DOTALL)
                if json_match_minimal:
                    return json.loads(json_match_minimal.group(1))
    except Exception:
        pass
    return None


def build_plot(dataframe, plot_config):
    if dataframe.empty or len(dataframe.columns) < 2:
        return None

    try:
        figure, axis = plt.subplots(figsize=(8, 5))
        chart_type = plot_config.get("type", "bar")
        
        # Selección dinámica de columnas (especificada en config o por defecto 0 y 1)
        # Soportar tanto x_axis/y_axis como x_data_column/y_data_column
        x_col = plot_config.get("x_axis") or plot_config.get("x_data_column") or dataframe.columns[0]
        y_col = plot_config.get("y_axis") or plot_config.get("y_data_column") or dataframe.columns[1]
        
        x_values = dataframe[x_col]
        y_values = pd.to_numeric(dataframe[y_col], errors="coerce")

        if y_values.isna().all():
            plt.close(figure)
            return None

        if chart_type == "line":
            axis.plot(x_values, y_values, marker="o")
        elif chart_type == "scatter":
            axis.scatter(x_values, y_values)
        else:
            axis.bar(x_values, y_values)

        axis.set_title(plot_config.get("title", f"Generated {chart_type.capitalize()} Chart"))
        axis.set_xlabel(plot_config.get("x_label", str(x_col)))
        axis.set_ylabel(plot_config.get("y_label", str(y_col)))
        plt.xticks(rotation=45)
        plt.tight_layout()
        return figure
    except Exception as e:
        print(f"Error building plot: {e}")
        if 'figure' in locals():
            plt.close(figure)
        return None


def generate_mermaid_html(mermaid_code):
    """
    Renders Mermaid code into HTML that Gradio can display.
    """
    if not mermaid_code or not mermaid_code.strip():
        return ""
    
    # We wrap the mermaid code in a div with class 'mermaid'
    # and include the mermaid.js library via CDN.
    html = f"""
    <div class="mermaid">
    {mermaid_code}
    </div>
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{ startOnLoad: true }});
    </script>
    """
    return html


def build_roadmap_diagram(tasks_list):
    """
    Example usage of python_mermaid to build a simple flowchart from a list of tasks.
    """
    if not tasks_list:
        return ""
    
    try:
        nodes = []
        links = []
        prev_node = None
        
        for i, task in enumerate(tasks_list):
            node_id = f"node_{i}"
            current_node = Node(node_id, task)
            nodes.append(current_node)
            if prev_node:
                links.append(Link(prev_node, current_node))
            prev_node = current_node
        
        chart = MermaidDiagram(
            title="Roadmap de Ejecución",
            nodes=nodes,
            links=links
        )
        return str(chart)
    except Exception as e:
        print(f"Error with python_mermaid: {e}")
        return ""


def summarize_profile_record(profile_record):
    if not profile_record:
        return ""
    summary_parts = []
    for key in [
        "display_name",
        "goal_summary",
        "business_context",
        "financial_objectives",
        "personality_summary",
        "constraints",
    ]:
        value = profile_record.get(key)
        if value:
            summary_parts.append(f"{key}: {value}")
    return " | ".join(summary_parts)


def load_persistent_memory(profile_id):
    normalized_profile = slugify_profile_id(profile_id)
    client, error_message = get_supabase_client()
    if error_message:
        return {
            "profile_id": normalized_profile,
            "summary": "",
            "status": f"Memoria persistente no disponible: {error_message}",
            "profile": {},
            "conversations": [],
        }
    if client is None:
        return {
            "profile_id": normalized_profile,
            "summary": "",
            "status": "Cliente de Supabase no disponible.",
            "profile": {},
            "conversations": [],
        }
    assert client is not None
    try:
        profile_response = supabase_request(
            client,
            "GET",
            PROFILE_TABLE,
            params={"select": "*", "profile_id": f"eq.{normalized_profile}", "limit": 1},
        )
        conversation_records = supabase_request(
            client,
            "GET",
            CONVERSATION_TABLE,
            params={
                "select": "user_message,assistant_reply,chat_mode,personality_summary,created_at",
                "profile_id": f"eq.{normalized_profile}",
                "order": "created_at.desc",
                "limit": 5,
            },
        )
        profile_record = (profile_response or [None])[0] or {}
        recent_lines = []
        for row in reversed(conversation_records):
            recent_lines.append(
                f"[{row.get('chat_mode', 'chat')}] Usuario: {row.get('user_message', '')} | Asistente: {row.get('assistant_reply', '')}"
            )
        summary_sections = [summarize_profile_record(profile_record)] if profile_record else []
        if recent_lines:
            summary_sections.append("Conversaciones recientes: " + " || ".join(recent_lines))
        summary = " -- ".join(section for section in summary_sections if section)
        status = "Memoria persistente cargada desde Supabase." if summary else "Sin memoria previa para este perfil."
        return {
            "profile_id": normalized_profile,
            "summary": summary,
            "status": status,
            "profile": profile_record,
            "conversations": conversation_records,
        }
    except Exception as error:
        return {
            "profile_id": normalized_profile,
            "summary": "",
            "status": format_supabase_error(error),
            "profile": {},
            "conversations": [],
        }


def persist_profile_memory(profile_id, prompt_text, form_payload, personality_analysis=None):
    normalized_profile = slugify_profile_id(profile_id)
    client, error_message = get_supabase_client()
    if error_message:
        return f"No se guardo el perfil en Supabase: {error_message}"
    if client is None:
        return "No se guardo el perfil en Supabase: cliente no disponible."
    assert client is not None

    profile_payload = {
        "profile_id": normalized_profile,
        "display_name": form_payload.get("who_are_you") or normalized_profile,
        "goal_summary": form_payload.get("what_do_you_want") or form_payload.get("objectives", ""),
        "master_prompt": prompt_text,
        "financial_objectives": form_payload.get("financial_objectives", ""),
        "business_context": form_payload.get("integrations", ""),
        "constraints": form_payload.get("constraints", ""),
        "baseline_tone": form_payload.get("tone", "Directo"),
        "personality_summary": (personality_analysis or {}).get("summary", ""),
        "personality_traits": (personality_analysis or {}).get("traits", {}),
        "updated_at": utc_now_iso(),
    }
    try:
        supabase_request(
            client,
            "POST",
            PROFILE_TABLE,
            params={"on_conflict": "profile_id"},
            payload=[profile_payload],
            prefer="resolution=merge-duplicates,return=representation",
        )
        return f"Perfil {normalized_profile} guardado en Supabase."
    except Exception as error:
        return format_supabase_error(error)


def persist_conversation_memory(profile_id, chat_mode, model_name, user_message, assistant_reply, personality_analysis, dataframe):
    normalized_profile = slugify_profile_id(profile_id)
    client, error_message = get_supabase_client()
    if error_message:
        return f"Memoria no guardada: {error_message}"
    if client is None:
        return "Memoria no guardada: cliente de Supabase no disponible."
    assert client is not None

    profile_payload = {
        "profile_id": normalized_profile,
        "personality_summary": personality_analysis.get("summary", ""),
        "personality_traits": personality_analysis.get("traits", {}),
        "updated_at": utc_now_iso(),
    }
    conversation_payload = {
        "profile_id": normalized_profile,
        "chat_mode": chat_mode,
        "model_name": normalize_model_name(model_name) or "",
        "user_message": user_message,
        "assistant_reply": assistant_reply,
        "personality_summary": personality_analysis.get("summary", ""),
        "personality_traits": personality_analysis.get("traits", {}),
        "data_rows_count": int(len(dataframe.index)) if not dataframe.empty else 0,
        "created_at": utc_now_iso(),
    }
    try:
        supabase_request(
            client,
            "POST",
            PROFILE_TABLE,
            params={"on_conflict": "profile_id"},
            payload=[profile_payload],
            prefer="resolution=merge-duplicates,return=representation",
        )
        supabase_request(
            client,
            "POST",
            CONVERSATION_TABLE,
            payload=[conversation_payload],
            prefer="return=representation",
        )
        return f"Conversacion guardada para el perfil {normalized_profile}."
    except Exception as error:
        return format_supabase_error(error)


def save_profile_wrapper(
    profile_id,
    who_are_you,
    what_do_you_want,
    objectives,
    problems,
    kpis,
    tasks,
    schedule,
    integrations,
    financial_objectives,
    financial_pain_points,
    budget_range,
    revenue_sources,
    cost_structure,
    reports_needed,
    risk_tolerance,
    tone,
    constraints,
):
    prompt_text = build_master_prompt(
        who_are_you,
        what_do_you_want,
        objectives,
        problems,
        kpis,
        tasks,
        schedule,
        integrations,
        financial_objectives,
        financial_pain_points,
        budget_range,
        revenue_sources,
        cost_structure,
        reports_needed,
        risk_tolerance,
        tone,
        constraints,
    )
    personality_analysis = analyze_personality_signals(
        " ".join(
            [
                who_are_you,
                what_do_you_want,
                objectives,
                problems,
                tasks,
                financial_objectives,
                financial_pain_points,
            ]
        )
    )
    form_payload = {
        "who_are_you": who_are_you,
        "what_do_you_want": what_do_you_want,
        "objectives": objectives,
        "integrations": integrations,
        "financial_objectives": financial_objectives,
        "constraints": constraints,
        "tone": tone,
    }
    status = persist_profile_memory(profile_id, prompt_text, form_payload, personality_analysis)
    return prompt_text, status


rag_engine = None

def get_rag_engine(client):
    global rag_engine
    if rag_engine is None:
        rag_engine = RAGManager(client)
    return rag_engine


def build_request_contents(history, user_message, persistent_summary, uploaded_files=None, rag_results=None):
    types_module = importlib.import_module("google.genai.types")
    contents = []
    
    if persistent_summary:
        contents.append({"role": "user", "parts": [{"text": f"Contexto persistente: {persistent_summary}"}]})

    if rag_results:
        context_text = "\n".join([f"- {res}" for res in rag_results])
        contents.append({"role": "user", "parts": [{"text": f"Contexto relevante recuperado de documentos (RAG):\n{context_text}"}]})

    for message in history or []:
        role = "model" if message.get("role") == "assistant" else "user"
        content = message.get("content", "")
        if content:
            contents.append({"role": role, "parts": [{"text": content}]})

    user_parts = [{"text": user_message}]
    
    if uploaded_files:
        for file in uploaded_files:
            file_path = file.name
            ext = os.path.splitext(file_path)[1].lower()
            if ext in [".png", ".jpg", ".jpeg"]:
                with open(file_path, "rb") as f:
                    user_parts.append(types_module.Part.from_bytes(data=f.read(), mime_type=f"image/{ext[1:]}"))
            elif ext == ".pdf":
                with open(file_path, "rb") as f:
                    user_parts.append(types_module.Part.from_bytes(data=f.read(), mime_type="application/pdf"))
            elif ext == ".md":
                with open(file_path, "r", encoding="utf-8") as f:
                    user_parts.append({"text": f"\n[Documento Markdown: {os.path.basename(file_path)}]\n{f.read()}"})

    contents.append({"role": "user", "parts": user_parts})
    return contents


def process_request(user_message, chat_mode, selected_model, history, profile_id, uploaded_files=None):
    client, model_candidates, error_message = get_model_candidates()
    if error_message:
        return error_message, pd.DataFrame(), None, {"summary": "", "traits": {}, "business_rules": []}, error_message, "", ""
    if not model_candidates:
        return "No se encontro un modelo Gemini compatible para esta clave.", pd.DataFrame(), None, {"summary": "", "traits": {}, "business_rules": []}, "Sin modelos compatibles.", "", ""
    if client is None:
        return "No se pudo inicializar el cliente Gemini.", pd.DataFrame(), None, {"summary": "", "traits": {}, "business_rules": []}, "Cliente Gemini no disponible.", "", ""

    rag = get_rag_engine(client)
    rag_status = ""
    if uploaded_files:
        for f in uploaded_files:
            if f.name.endswith((".pdf", ".md")):
                rag_status += rag.process_file(f.name) + " "

    rag_context = rag.search(user_message) if rag.index.ntotal > 0 else []

    persistent_memory = load_persistent_memory(profile_id)
    personality_analysis = analyze_personality_signals(user_message) if chat_mode == "Conversacional" else {"summary": "", "traits": {}, "business_rules": []}
    
    # Determinar temperatura dinámica
    temp = 0.0 if chat_mode == "Analitica" else 0.7
    
    contents = build_request_contents(history, user_message, persistent_memory["summary"], uploaded_files, rag_context)
    generation_config = build_generation_config(chat_mode, personality_analysis, persistent_memory["summary"], temperature=temp)
    resolved_models = resolve_model_order(selected_model, model_candidates)
    compatibility_errors = []

    for model_name in resolved_models:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=generation_config,
            )
            output_text = (response.text or "").strip()
            parsed = parse_structured_output(output_text)

            if not parsed:
                # Soporte para Mermaid (extrayendo bloques de código mermaid)
                mermaid_match = re.search(r"```mermaid\s*(.*?)\s*```", output_text, re.DOTALL)
                mermaid_code = mermaid_match.group(1).strip() if mermaid_match else ""
                
                # Si no hay bloque formal de código mermaid, intentamos generar uno de roadmap 
                # a partir del texto si el usuario pidió roadmap o pasos claros.
                if not mermaid_code and ("roadmap" in user_message.lower() or "pasos" in user_message.lower()):
                    steps = re.findall(r"(?:\d+\.|\-)\s+(.+)", output_text)
                    if steps:
                        mermaid_code = build_roadmap_diagram(steps[:10])
                
                mermaid_html = generate_mermaid_html(mermaid_code)
                
                python_code = ""
                if "run" in user_message.lower():
                    code_match = re.search(r"```python\s*(.*?)\s*```", output_text, re.DOTALL)
                    if code_match:
                        python_code = code_match.group(1).strip()

                memory_status = persist_conversation_memory(
                    profile_id,
                    chat_mode,
                    model_name,
                    user_message,
                    output_text,
                    personality_analysis,
                    pd.DataFrame(),
                )
                return output_text, pd.DataFrame(), None, personality_analysis, memory_status, mermaid_html, python_code

            dataframe = pd.DataFrame()
            if parsed.get("has_data") and "table_data" in parsed:
                columns = parsed["table_data"].get("columns", [])
                rows = parsed["table_data"].get("rows", [])
                dataframe = pd.DataFrame(rows, columns=columns)

            figure = None
            if "plot_config" in parsed:
                figure = build_plot(dataframe, parsed["plot_config"])

            if parsed.get("personality_profile"):
                personality_analysis["summary"] = parsed["personality_profile"]

            assistant_reply = parsed.get("response_text", "Analisis completado.")
            
            # Soporte para Mermaid (extrayendo bloques de código mermaid)
            mermaid_match = re.search(r"```mermaid\s*(.*?)\s*```", output_text, re.DOTALL)
            mermaid_code = mermaid_match.group(1).strip() if mermaid_match else ""

            # Si el usuario pidió un roadmap o hay un plan de pasos claro en la respuesta,
            # intentamos usar python_mermaid para formalizarlo si no hay mermaid_code aun.
            if not mermaid_code and ("roadmap" in user_message.lower() or "pasos" in user_message.lower()):
                # Intento simple: extraer líneas que empiecen con numero o viñeta
                steps = re.findall(r"(?:\d+\.|\-)\s+(.+)", assistant_reply)
                if steps:
                    mermaid_code = build_roadmap_diagram(steps[:10]) # Limitar a 10 nodos

            mermaid_html = generate_mermaid_html(mermaid_code)

            # Soporte para ejecución de código Python
            python_code = ""
            if "run" in user_message.lower():
                code_match = re.search(r"```python\s*(.*?)\s*```", output_text, re.DOTALL)
                if code_match:
                    python_code = code_match.group(1).strip()

            memory_status = persist_conversation_memory(
                profile_id,
                chat_mode,
                model_name,
                user_message,
                assistant_reply,
                personality_analysis,
                dataframe,
            )
            return assistant_reply, dataframe, figure, personality_analysis, memory_status, mermaid_html, python_code
        except Exception as error:
            if is_model_compatibility_error(error):
                compatibility_errors.append(f"{model_name}: {error}")
                continue
            return (
                f"Error procesando la solicitud con {model_name}: {error}",
                pd.DataFrame(),
                None,
                personality_analysis,
                persistent_memory["status"],
                "", ""
            )

    if compatibility_errors:
        return (
            "No fue posible generar respuesta con los modelos compatibles detectados. " + " | ".join(compatibility_errors[:3]),
            pd.DataFrame(),
            None,
            personality_analysis,
            persistent_memory["status"],
            "", ""
        )

    return "No se pudo inicializar un modelo Gemini utilizable.", pd.DataFrame(), None, personality_analysis, persistent_memory["status"], "", ""


def chat_wrapper(user_text, chat_mode, selected_model, history, profile_id, uploaded_files):
    history = history or []
    if not user_text or not user_text.strip():
        if uploaded_files:
            user_text = "Procesa los archivos cargados."
        else:
            return history, history, "", pd.DataFrame(), None, "", "", "", ""

    bot_reply, dataframe, figure, personality_analysis, memory_status, mermaid_code, python_code = process_request(
        user_text,
        chat_mode,
        selected_model,
        history,
        profile_id,
        uploaded_files=uploaded_files,
    )
    updated_history = history + [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": bot_reply},
    ]
    return (
        updated_history,
        updated_history,
        "",
        dataframe,
        figure,
        format_personality_analysis(personality_analysis),
        memory_status,
        mermaid_code,
        python_code
    )


def load_memory_wrapper(profile_id):
    memory = load_persistent_memory(profile_id)
    return memory["summary"] or "Sin memoria almacenada para este perfil.", memory["status"]


def clear_chat():
    return [], [], "", pd.DataFrame(), None, "", "", "", ""


loaded_env = [
    key
    for key in ["GEMINI_API_KEY", "SUPABASE_URL", "SUPABASE_KEY", "SUPABASE_SERVICE_KEY", "HF_TOKEN"]
    if os.getenv(key)
]
env_preview = ", ".join(loaded_env) if loaded_env else "ninguna"
model_choices = build_model_choices()
default_model = strip_model_prefix(normalize_model_name(os.getenv("GEMINI_MODEL"))) or (model_choices[0] if model_choices else "")

with gr.Blocks(title="Cesar Assistant") as demo:
    gr.Markdown("# Cesar Assistant")
    gr.Markdown(
        "Formulario para definir funciones, objetivos y perfil de Cesar Mora, con memoria persistente en Supabase y un modulo analitico para razonamiento, tablas, graficos y NLP de personalidad."
    )

    with gr.Tabs():
        with gr.Tab("Discovery"):
            profile_id_input = gr.Textbox(label="Profile ID", value=DEFAULT_PROFILE_ID, lines=1)
            with gr.Row():
                with gr.Column(scale=1):
                    who_are_you = gr.Textbox(label="Quien eres", lines=2, placeholder="Describe tu perfil, rol o negocio")
                    what_do_you_want = gr.Textbox(label="Que quieres lograr", lines=2, placeholder="Resultados esperados con Cesar Assistant")
                    objectives = gr.Textbox(label="Objetivo principal", lines=2)
                    problems = gr.Textbox(label="Problemas principales a resolver", lines=3)
                    kpis = gr.Textbox(label="KPIs o metricas de exito", lines=2)
                    tasks = gr.Textbox(label="Tareas repetitivas y custom tasks", lines=3)
                    schedule = gr.Textbox(label="Rutinas y horarios", lines=2)
                    integrations = gr.Textbox(label="Integraciones requeridas", lines=2)
                with gr.Column(scale=1):
                    financial_objectives = gr.Textbox(label="Objetivos financieros", lines=2, placeholder="Margen, crecimiento, flujo de caja, ahorro")
                    financial_pain_points = gr.Textbox(label="Problemas financieros actuales", lines=3, placeholder="Cobranza, costos, rentabilidad, deuda")
                    budget_range = gr.Textbox(label="Rango de presupuesto", lines=2)
                    revenue_sources = gr.Textbox(label="Fuentes de ingreso", lines=2)
                    cost_structure = gr.Textbox(label="Estructura de costos", lines=2)
                    reports_needed = gr.Textbox(label="Reportes o dashboards deseados", lines=2)
                    risk_tolerance = gr.Dropdown(label="Tolerancia al riesgo financiero", choices=["Baja", "Media", "Alta"], value="Media")
                    tone = gr.Dropdown(label="Tono preferido", choices=["Formal", "Directo", "Tecnico", "Amigable"], value="Directo")
                    constraints = gr.Textbox(label="Reglas o restricciones", lines=2)

            with gr.Row():
                prompt_button = gr.Button("Generar Prompt Maestro", variant="primary")
                save_profile_button = gr.Button("Guardar Perfil en Supabase")
            prompt_output = gr.Textbox(label="Prompt Maestro generado", lines=24)
            discovery_status = gr.Textbox(label="Estado de memoria de perfil", lines=3, interactive=False)

            prompt_button.click(
                fn=build_master_prompt,
                inputs=[
                    who_are_you,
                    what_do_you_want,
                    objectives,
                    problems,
                    kpis,
                    tasks,
                    schedule,
                    integrations,
                    financial_objectives,
                    financial_pain_points,
                    budget_range,
                    revenue_sources,
                    cost_structure,
                    reports_needed,
                    risk_tolerance,
                    tone,
                    constraints,
                ],
                outputs=prompt_output,
            )
            save_profile_button.click(
                fn=save_profile_wrapper,
                inputs=[
                    profile_id_input,
                    who_are_you,
                    what_do_you_want,
                    objectives,
                    problems,
                    kpis,
                    tasks,
                    schedule,
                    integrations,
                    financial_objectives,
                    financial_pain_points,
                    budget_range,
                    revenue_sources,
                    cost_structure,
                    reports_needed,
                    risk_tolerance,
                    tone,
                    constraints,
                ],
                outputs=[prompt_output, discovery_status],
            )

        with gr.Tab("Analitica"):
            gr.Markdown(
                "Usa Gemini para razonamiento paso a paso o conversacion NLP. Puedes subir archivos (PDF, MD, Imágenes) para análisis multimodal y RAG."
            )
            analytics_profile_id = gr.Textbox(label="Profile ID para memoria", value=DEFAULT_PROFILE_ID, lines=1)
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Row():
                        chat_mode = gr.Dropdown(label="Modo", choices=["Analitica", "Conversacional"], value="Analitica")
                        model_selector = gr.Dropdown(label="Modelo Gemini", choices=model_choices, value=default_model, allow_custom_value=True)
                    
                    file_upload = gr.File(label="Subir Documentos (PDF, MD) o Imágenes", file_types=[".pdf", ".md", ".png", ".jpg", ".jpeg"], file_count="multiple")
                    
                    chatbot_ui = gr.Chatbot(label="Cesar Assistant", height=420)
                    msg_input = gr.Textbox(
                        label="Pregunta o instruccion",
                        placeholder="Pide una explicacion, una tabla, un grafico o sube archivos para investigar...",
                    )
                    with gr.Row():
                        submit_btn = gr.Button("Enviar", variant="primary")
                        clear_btn = gr.Button("Limpiar")
                        load_memory_btn = gr.Button("Cargar memoria")
                with gr.Column(scale=1):
                    personality_output = gr.Textbox(label="Perfil NLP y reglas de negocio", lines=6, interactive=False)
                    memory_output = gr.Textbox(label="Memoria persistente", lines=6, interactive=False)
                    data_output = gr.Dataframe(label="Pandas DataFrame Viewer", interactive=False)
                    plot_output = gr.Plot(label="Matplotlib Visualization")
                    mermaid_output = gr.HTML(label="Mermaid Diagram Render")
                    code_execute_view = gr.Code(label="Python Code Executed", language="python", interactive=False)

            state_history = gr.State([])
            submit_btn.click(
                chat_wrapper,
                inputs=[msg_input, chat_mode, model_selector, state_history, analytics_profile_id, file_upload],
                outputs=[chatbot_ui, state_history, msg_input, data_output, plot_output, personality_output, memory_output, mermaid_output, code_execute_view],
            )
            msg_input.submit(
                chat_wrapper,
                inputs=[msg_input, chat_mode, model_selector, state_history, analytics_profile_id, file_upload],
                outputs=[chatbot_ui, state_history, msg_input, data_output, plot_output, personality_output, memory_output, mermaid_output, code_execute_view],
            )
            clear_btn.click(
                clear_chat,
                outputs=[chatbot_ui, state_history, msg_input, data_output, plot_output, personality_output, memory_output, mermaid_output, code_execute_view],
            )
            load_memory_btn.click(
                load_memory_wrapper,
                inputs=[analytics_profile_id],
                outputs=[memory_output, personality_output],
            )

    gr.Markdown(f"Variables de entorno detectadas: {env_preview}")

if __name__ == "__main__":
    server_port = int(os.getenv("PORT", "7860"))
    enable_share = os.getenv("GRADIO_SHARE", "false").lower() == "true"
    demo.launch(server_name="0.0.0.0", server_port=server_port, share=enable_share)
