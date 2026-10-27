
import os
import re
import sqlite3
import requests

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "querymind.db")

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:1.5b"


def get_db():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    with get_db() as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS employees (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                department TEXT NOT NULL,
                salary INTEGER NOT NULL,
                city TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sales (
                id INTEGER PRIMARY KEY,
                employee_id INTEGER,
                product TEXT NOT NULL,
                amount REAL NOT NULL,
                month TEXT NOT NULL,
                FOREIGN KEY(employee_id) REFERENCES employees(id)
            );
        """)

        employee_count = db.execute(
            "SELECT COUNT(*) FROM employees"
        ).fetchone()[0]

        if employee_count == 0:
            db.executemany(
                """INSERT INTO employees
                   (name, department, salary, city)
                   VALUES (?, ?, ?, ?)""",
                [
                    ("Aarav Sharma", "Engineering", 75000, "Delhi"),
                    ("Priya Singh", "Marketing", 55000, "Mumbai"),
                    ("Rahul Verma", "Engineering", 82000, "Noida"),
                    ("Sneha Gupta", "Sales", 48000, "Delhi"),
                    ("Kabir Khan", "Finance", 68000, "Lucknow"),
                    ("Ananya Das", "Sales", 62000, "Mumbai"),
                    ("Rohan Mehta", "Engineering", 90000, "Pune"),
                    ("Isha Patel", "HR", 50000, "Ahmedabad"),
                ]
            )

        sales_count = db.execute(
            "SELECT COUNT(*) FROM sales"
        ).fetchone()[0]

        if sales_count == 0:
            db.executemany(
                """INSERT INTO sales
                   (employee_id, product, amount, month)
                   VALUES (?, ?, ?, ?)""",
                [
                    (1, "Laptop", 65000, "January"),
                    (2, "Software", 25000, "January"),
                    (3, "Cloud Service", 45000, "February"),
                    (4, "Laptop", 70000, "February"),
                    (5, "Software", 18000, "March"),
                    (6, "Cloud Service", 52000, "March"),
                    (7, "Laptop", 85000, "April"),
                    (8, "Software", 22000, "April"),
                    (1, "Cloud Service", 38000, "May"),
                    (4, "Software", 15000, "May"),
                ]
            )


SCHEMA = """
Database tables:

employees(id, name, department, salary, city)
sales(id, employee_id, product, amount, month)

Relationship:
sales.employee_id references employees.id.
"""


def generate_sql(question):
    prompt = f"""
You are a careful SQLite query generator.

{SCHEMA}

Convert the user's question into exactly one SQLite SELECT query.
Rules:
- Return SQL only, without markdown or explanation.
- Only SELECT queries are allowed.
- Never use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE,
  PRAGMA, ATTACH, or other modifying statements.
- Use only the tables and columns listed above.
- Use LIMIT 100 unless the query already has a smaller limit.

User question: {question}
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "Return one safe SQLite SELECT query only."
                },
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "options": {"temperature": 0}
        },
        timeout=120
    )

    response.raise_for_status()
    sql = response.json()["message"]["content"].strip()

    # Remove optional Markdown code fences.
    sql = re.sub(r"^```(?:sql)?\s*", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\s*```$", "", sql).strip()

    return sql


def run_read_only_query(sql):
    # Accept only a single SELECT statement.
    if not re.match(r"^\s*SELECT\b", sql, re.IGNORECASE):
        raise ValueError("Only SELECT queries are allowed.")

    if ";" in sql.rstrip().rstrip(";"):
        raise ValueError("Only one SQL statement is allowed.")

    blocked = re.compile(
        r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|"
        r"ATTACH|DETACH|PRAGMA|VACUUM|REINDEX|TRIGGER)\b",
        re.IGNORECASE
    )

    if blocked.search(sql):
        raise ValueError("The generated SQL contains a blocked keyword.")

    db = get_db()

    try:
        # SQLite-level read-only enforcement.
        def authorize(action, arg1, arg2, database, trigger):
            allowed_actions = {
                sqlite3.SQLITE_SELECT,
                sqlite3.SQLITE_READ,
                sqlite3.SQLITE_FUNCTION,
            }
            if action in allowed_actions:
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY

        db.set_authorizer(authorize)
        cursor = db.execute(sql)
        rows = cursor.fetchmany(100)
        columns = [item[0] for item in cursor.description or []]

        return columns, [
            {key: row[key] for key in columns}
            for row in rows
        ]
    finally:
        db.close()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    try:
        response = requests.get(
            "http://localhost:11434/api/tags",
            timeout=5
        )
        response.raise_for_status()
        models = [
            item["name"]
            for item in response.json().get("models", [])
        ]

        return jsonify({
            "status": "online",
            "ollama": "connected",
            "model": MODEL,
            "model_available": any(
                name == MODEL or name.startswith(MODEL + ":")
                for name in models
            )
        })
    except requests.RequestException:
        return jsonify({
            "status": "offline",
            "ollama": "disconnected",
            "message": "Start Ollama and check that the model is installed."
        }), 503


@app.route("/api/query", methods=["POST"])
def query():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "Please enter a question."}), 400

    if len(question) > 500:
        return jsonify({"error": "Question must be under 500 characters."}), 400

    try:
        sql = generate_sql(question)
        columns, rows = run_read_only_query(sql)

        return jsonify({
            "question": question,
            "sql": sql,
            "columns": columns,
            "rows": rows,
            "row_count": len(rows)
        })

    except requests.ConnectionError:
        return jsonify({
            "error": "Ollama is not running. Open Ollama and try again."
        }), 503

    except requests.Timeout:
        return jsonify({
            "error": "AI took too long to respond. Please try again."
        }), 504

    except requests.HTTPError:
        return jsonify({
            "error": "Ollama returned an error. Check that the model is installed."
        }), 502

    except (ValueError, sqlite3.Error) as error:
        return jsonify({"error": str(error)}), 400

    except Exception:
        app.logger.exception("Unexpected query error")
        return jsonify({
            "error": "Something went wrong. Check the VS Code terminal."
        }), 500


if __name__ == "__main__":
    init_db()
    app.run(debug=True)
