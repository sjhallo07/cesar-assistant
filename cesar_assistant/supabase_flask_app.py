import os
from flask import Flask
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)


# Try to import supabase client; if not available, provide a graceful fallback
try:
    from supabase import create_client, Client  # type: ignore
    SUPABASE_AVAILABLE = True
except Exception:
    create_client = None
    Client = None
    SUPABASE_AVAILABLE = False


def get_supabase_client():
    if not SUPABASE_AVAILABLE:
        return None
    return create_client(
        os.environ.get("SUPABASE_URL"),
        os.environ.get("SUPABASE_KEY"),
    )


@app.route('/')
def index():
    if not SUPABASE_AVAILABLE:
        msg = (
            '<h1>Supabase no instalado</h1>'
            '<p>El paquete <strong>supabase</strong> no está disponible en este entorno.</p>'
            '<p>Si quieres usar Supabase, instala las Build Tools de Microsoft y luego instala el paquete:</p>'
            '<pre>pip install supabase</pre>'
        )
        return msg

    try:
        supabase = get_supabase_client()
        if supabase is None:
            return '<h1>Supabase no configurado</h1><p>Faltan SUPABASE_URL o SUPABASE_KEY en el entorno.</p>'

        response = supabase.table('todos').select("*").execute()
        todos = getattr(response, 'data', []) or []

        if not todos:
            return '<h1>No hay tareas pendientes</h1>'

        html = '<h1>Lista de Tareas (Todos)</h1><ul>'
        for todo in todos:
            name = todo.get("name", "Sin nombre") if isinstance(todo, dict) else str(todo)
            html += f'<li>{name}</li>'
        html += '</ul>'
        return html
    except Exception as e:
        return f'<h1>Error al conectar con Supabase</h1><p>{str(e)}</p>', 500


if __name__ == '__main__':
    app.run(debug=True)
