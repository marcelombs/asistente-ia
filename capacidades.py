"""Las capabilities de tu asistente: inventario de chompas, gastos de la tienda y documentos.

Una Capability junta tools + instrucciones en una pieza que se enchufa al agente (asistente.py).
El docstring y los tipos de cada tool son lo único que el modelo lee para decidir cuándo usarla.
"""

import csv
import sqlite3
from datetime import date
from functools import cache
from typing import Literal

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pydantic_ai.capabilities import Capability
from pypdf import PdfReader

from config import DATOS, settings

Modelo = Literal["CH-001", "CH-002", "CH-003", "CH-004", "CH-005"]
Talla = Literal["S", "M", "L", "XL", "4", "6", "8", "10", "12"]

# ═══ Inventario: stock, precios, ventas y compras (lectura) ════════════════════════════
inventario = Capability(
    id="inventario",
    description="Stock, precios, ventas y compras registradas de las chompas CH-001 a CH-005",
    instructions=(
        "Usa la base de datos de la tienda para responder sobre disponibilidad, tallas, precios, "
        "ventas y compras. No inventes datos: si algo no está, dilo."
    ),
)


def _asegurar_base() -> None:
    if not DB.exists():
        crear_base()


@inventario.tool_plain
def consultar_stock(modelo: Modelo | None = None, talla: Talla | None = None) -> list[dict] | dict:
    """Stock y precio de venta de las chompas. Filtra por modelo (por ejemplo CH-001) y/o talla.

    Args:
        modelo: código del modelo, por ejemplo CH-001. Opcional: sin él, devuelve todo el stock
        talla: talla, por ejemplo M o 12. Opcional
    """
    _asegurar_base()
    conexion = sqlite3.connect(DB.resolve().as_uri() + "?mode=ro", uri=True)
    conexion.row_factory = sqlite3.Row
    sql = "SELECT modelo, talla, cantidad, precio FROM stock"
    condiciones: list[str] = []
    parametros: list = []
    if modelo:
        condiciones.append("modelo = ?")
        parametros.append(modelo)
    if talla:
        condiciones.append("talla = ?")
        parametros.append(talla)
    if condiciones:
        sql += " WHERE " + " AND ".join(condiciones)
    filas = conexion.execute(sql, parametros).fetchall()
    conexion.close()
    if not filas:
        return {"error": f"No encontré en el catálogo modelo={modelo or '*'} talla={talla or '*'}"}
    return [dict(f) for f in filas]


@inventario.tool_plain
def consultar_ventas(fecha: str | None = None, modelo: Modelo | None = None) -> list[dict] | dict:
    """Ventas registradas. Filtra por fecha (AAAA-MM-DD) y/o modelo.

    Args:
        fecha: fecha exacta, por ejemplo 2026-11-07. Opcional
        modelo: código del modelo, por ejemplo CH-001. Opcional
    """
    _asegurar_base()
    conexion = sqlite3.connect(DB.resolve().as_uri() + "?mode=ro", uri=True)
    conexion.row_factory = sqlite3.Row
    sql = "SELECT fecha, modelo, talla, cantidad, precio_unitario, total FROM ventas"
    condiciones: list[str] = []
    parametros: list = []
    if fecha:
        condiciones.append("fecha = ?")
        parametros.append(fecha)
    if modelo:
        condiciones.append("modelo = ?")
        parametros.append(modelo)
    if condiciones:
        sql += " WHERE " + " AND ".join(condiciones)
    sql += " ORDER BY fecha"
    filas = conexion.execute(sql, parametros).fetchall()
    conexion.close()
    if not filas:
        return {"error": "No hay ventas registradas con ese filtro"}
    return [dict(f) for f in filas]


@inventario.tool_plain
def consultar_compras(fecha: str | None = None, modelo: Modelo | None = None) -> list[dict] | dict:
    """Compras al proveedor registradas. Filtra por fecha (AAAA-MM-DD) y/o modelo.

    Args:
        fecha: fecha exacta, por ejemplo 2026-11-07. Opcional
        modelo: código del modelo, por ejemplo CH-001. Opcional
    """
    _asegurar_base()
    conexion = sqlite3.connect(DB.resolve().as_uri() + "?mode=ro", uri=True)
    conexion.row_factory = sqlite3.Row
    sql = "SELECT fecha, modelo, talla, cantidad, costo_unitario, total FROM compras"
    condiciones: list[str] = []
    parametros: list = []
    if fecha:
        condiciones.append("fecha = ?")
        parametros.append(fecha)
    if modelo:
        condiciones.append("modelo = ?")
        parametros.append(modelo)
    if condiciones:
        sql += " WHERE " + " AND ".join(condiciones)
    sql += " ORDER BY fecha"
    filas = conexion.execute(sql, parametros).fetchall()
    conexion.close()
    if not filas:
        return {"error": "No hay compras registradas con ese filtro"}
    return [dict(f) for f in filas]


# ═══ Registro de ventas (escritura) ═══════════════════════════════════════════════════
def hoy() -> str:
    """Instrucción dinámica: el modelo no sabe qué día es; se lo decimos en cada mensaje."""
    return f"Hoy es {date.today().isoformat()}."


ventas = Capability(
    id="registro_ventas",
    description="Registra ventas de chompas y descuenta el stock",
    instructions=[
        "Registra cada venta con fecha, modelo, talla y cantidad. "
        "Si no hay stock suficiente, di cuánto queda y no registres la venta.",
        hoy,
    ],
)


@ventas.tool_plain
def registrar_venta(fecha: str, modelo: Modelo, talla: Talla, cantidad: int) -> str:
    """Registra una venta y descuenta el stock. La fecha va como AAAA-MM-DD y la cantidad en unidades.

    Args:
        fecha: fecha de la venta, por ejemplo 2026-11-07
        modelo: código del modelo, por ejemplo CH-001
        talla: talla vendida, por ejemplo M
        cantidad: unidades vendidas, por ejemplo 2
    """
    _asegurar_base()
    conexion = sqlite3.connect(DB)
    try:
        fila = conexion.execute(
            "SELECT precio, cantidad FROM stock WHERE modelo = ? AND talla = ?", (modelo, talla)
        ).fetchone()
        if fila is None:
            return f"No se registró: {modelo} talla {talla} no está en el catálogo."
        precio, en_stock = fila
        if cantidad > en_stock:
            return f"No se registró: solo quedan {en_stock} unidades de {modelo} talla {talla}."
        total = precio * cantidad
        with conexion:  # transacción: descuenta y anota, todo o nada
            conexion.execute(
                "UPDATE stock SET cantidad = cantidad - ? WHERE modelo = ? AND talla = ?",
                (cantidad, modelo, talla),
            )
            conexion.execute(
                "INSERT INTO ventas VALUES (?, ?, ?, ?, ?, ?)",
                (fecha, modelo, talla, cantidad, precio, total),
            )
    except sqlite3.IntegrityError:  # el CHECK de la tabla no deja stock negativo
        return f"No se registró: stock insuficiente de {modelo} talla {talla}."
    finally:
        conexion.close()
    return (
        f"Venta registrada: {cantidad} x {modelo} talla {talla} a {precio} Bs "
        f"= {total} Bs el {fecha}. Quedan {en_stock - cantidad} unidades."
    )


# ═══ Registro de compras al proveedor (escritura) ══════════════════════════════════════
compras = Capability(
    id="registro_compras",
    description="Registra compras de chompas al proveedor y suma el stock",
    instructions=[
        "Registra cada compra al proveedor con fecha, modelo, talla y cantidad. "
        "Si no llevas el costo unitario, usa el del contrato de proveedor (o déjalo vacío y la tool lo toma del stock).",
        hoy,
    ],
)


@compras.tool_plain
def registrar_compra(
    fecha: str, modelo: Modelo, talla: Talla, cantidad: int, costo_unitario: float | None = None
) -> str:
    """Registra una compra al proveedor y suma el stock. La fecha va como AAAA-MM-DD.

    Args:
        fecha: fecha de la entrega, por ejemplo 2026-11-07
        modelo: código del modelo, por ejemplo CH-001
        talla: talla comprada, por ejemplo M
        cantidad: unidades compradas, por ejemplo 24
        costo_unitario: costo por unidad en Bs. Si no se pasa, se usa el del contrato de proveedor
    """
    _asegurar_base()
    conexion = sqlite3.connect(DB)
    try:
        fila = conexion.execute(
            "SELECT costo FROM stock WHERE modelo = ? AND talla = ?", (modelo, talla)
        ).fetchone()
        if fila is None:
            return f"No se registró: {modelo} talla {talla} no está en el catálogo."
        costo = costo_unitario if costo_unitario is not None else fila[0]
        if costo <= 0:
            return f"No se registró: el costo unitario debe ser mayor que 0 (recibí {costo})."
        total = costo * cantidad
        with conexion:  # transacción: suma y anota, todo o nada
            conexion.execute(
                "UPDATE stock SET cantidad = cantidad + ? WHERE modelo = ? AND talla = ?",
                (cantidad, modelo, talla),
            )
            conexion.execute(
                "INSERT INTO compras VALUES (?, ?, ?, ?, ?, ?)",
                (fecha, modelo, talla, cantidad, costo, total),
            )
            restante = conexion.execute(
                "SELECT cantidad FROM stock WHERE modelo = ? AND talla = ?", (modelo, talla)
            ).fetchone()[0]
    except sqlite3.IntegrityError:
        return f"No se registró: cantidad inválida ({cantidad})."
    finally:
        conexion.close()
    return (
        f"Compra registrada: {cantidad} x {modelo} talla {talla} a {costo} Bs "
        f"= {total} Bs el {fecha}. Quedan {restante} unidades."
    )


# ═══ Gastos de la tienda: una tool que lee y otra que escribe ══════════════════════════
Categoria = Literal[
    "transporte", "insumos", "alquiler", "mercaderia", "servicios", "marketing", "prestamo", "mantenimiento"
]
CSV = DATOS / "gastos_tienda.csv"  # gastos ficticios de la tienda (octubre de 2026)
STOCK_CSV = DATOS / "stock.csv"  # stock y precios del catálogo al 31/10/2026
DB = DATOS / "tienda.db"  # la base: se crea desde los CSV la primera vez


def crear_base() -> None:
    """Crea tienda.db desde cero con los gastos y el stock de los CSV."""
    DB.unlink(missing_ok=True)
    with CSV.open(encoding="utf-8") as archivo:
        filas_gastos = [
            (g["fecha"], g["categoria"], g["descripcion"], float(g["monto"])) for g in csv.DictReader(archivo)
        ]
    with STOCK_CSV.open(encoding="utf-8") as archivo:
        filas_stock = [
            (g["modelo"], g["talla"], int(g["cantidad"]), float(g["precio"]), float(g["costo"]))
            for g in csv.DictReader(archivo)
        ]
    conexion = sqlite3.connect(DB)
    with conexion:
        conexion.execute(
            "CREATE TABLE gastos (fecha TEXT, categoria TEXT, descripcion TEXT, monto REAL CHECK (monto > 0))"
        )
        conexion.execute(
            "CREATE TABLE stock ("
            "modelo TEXT, talla TEXT, cantidad INTEGER CHECK (cantidad >= 0), "
            "precio REAL, costo REAL, PRIMARY KEY (modelo, talla))"
        )
        conexion.execute(
            "CREATE TABLE ventas ("
            "fecha TEXT, modelo TEXT, talla TEXT, cantidad INTEGER CHECK (cantidad > 0), "
            "precio_unitario REAL, total REAL)"
        )
        conexion.execute(
            "CREATE TABLE compras ("
            "fecha TEXT, modelo TEXT, talla TEXT, cantidad INTEGER CHECK (cantidad > 0), "
            "costo_unitario REAL, total REAL)"
        )
        conexion.executemany("INSERT INTO gastos VALUES (?, ?, ?, ?)", filas_gastos)
        conexion.executemany("INSERT INTO stock VALUES (?, ?, ?, ?, ?)", filas_stock)
    conexion.close()


# Lectura y escritura van en capabilities separadas: a otro agente le podrías dar solo la primera
consulta = Capability(
    id="consulta_gastos",
    description="Consulta gastos de la tienda almacenados en una base de datos SQLite",
    instructions="Usa la base de datos para responder preguntas sobre los gastos. No inventes datos.",
)
registro = Capability(
    id="registro_gastos",
    description="Registra nuevos gastos de la tienda en la base de datos SQLite",
    instructions=["Agrega nuevos gastos a la base de datos. Asegúrate de que los datos sean válidos.", hoy],
)


@consulta.tool_plain
def consultar_gastos(categoria: Categoria) -> dict:
    """Gastos de una categoría y su total en bolivianos."""
    _asegurar_base()
    conexion = sqlite3.connect(DB.resolve().as_uri() + "?mode=ro", uri=True)  # solo lectura
    conexion.row_factory = sqlite3.Row  # cada fila como diccionario
    filas = conexion.execute(
        "SELECT fecha, descripcion, monto FROM gastos WHERE categoria = ?",  # SQL fijo; el valor va en ?
        (categoria,),
    ).fetchall()
    conexion.close()
    return {"categoria": categoria, "total_bs": sum(f["monto"] for f in filas), "gastos": [dict(f) for f in filas]}


@registro.tool_plain
def registrar_gasto(fecha: str, categoria: Categoria, descripcion: str, monto: float) -> str:
    """Anota un gasto nuevo. La fecha va como AAAA-MM-DD y el monto en bolivianos."""
    _asegurar_base()
    conexion = sqlite3.connect(DB)  # sin mode=ro: esta tool sí escribe
    try:
        with conexion:  # transacción: guarda todo o nada
            conexion.execute("INSERT INTO gastos VALUES (?, ?, ?, ?)", (fecha, categoria, descripcion, monto))
    except sqlite3.IntegrityError:  # el CHECK de la tabla rechaza montos de 0 o negativos
        return f"No se guardó: el monto debe ser mayor que 0 (recibí {monto})."
    finally:
        conexion.close()
    return f"Gasto registrado: {descripcion}, {monto} Bs en {categoria} el {fecha}."


# ═══ Documentos: buscar en mis PDFs (catálogo, contratos, garantías) y citar la fuente ══
DOCUMENTOS = DATOS / "documentos"  # PDFs ficticios de la tienda


def embeddings_de(textos: list[str]) -> list[list[float]]:
    """Convierte cada texto en un vector: textos de significado parecido dan vectores cercanos."""
    # Gemini también ofrece una API compatible con la de OpenAI; la usamos para los embeddings
    cliente = OpenAI(
        api_key=settings.gemini_api_key_local.get_secret_value(),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    respuesta = cliente.embeddings.create(model="gemini-embedding-001", input=textos)
    return [item.embedding for item in respuesta.data]


@cache  # se indexa la primera vez que alguien busca, y una sola vez mientras el programa corre
def indice():
    # Base vectorial en memoria (se pierde al terminar); "cosine" mide la cercanía por el ángulo
    coleccion = chromadb.EphemeralClient().get_or_create_collection(
        "documentos", configuration={"hnsw": {"space": "cosine"}}
    )
    for pdf in sorted(DOCUMENTOS.glob("*.pdf")):
        texto = "\n".join(pagina.extract_text() for pagina in PdfReader(pdf).pages)
        # Trozos de ~300 caracteres que se solapan 50, para no cortar una idea por la mitad
        trozos = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50).split_text(texto)
        coleccion.upsert(
            ids=[f"{pdf.stem}-{i}" for i in range(len(trozos))],  # el id dice de qué PDF sale: sirve para citar
            documents=trozos,
            embeddings=embeddings_de(trozos),
        )
    return coleccion


def buscar(pregunta: str, k: int = 3) -> list[dict]:
    """Los k trozos más parecidos a la pregunta, con su fuente y su similitud (1 = idéntico)."""
    r = indice().query(query_embeddings=embeddings_de([pregunta]), n_results=k)
    return [
        {"fuente": fuente, "similitud": round(1 - distancia, 2), "texto": texto}
        for fuente, distancia, texto in zip(r["ids"][0], r["distances"][0], r["documents"][0])
    ]


documentos = Capability(
    id="mis_documentos",
    description="Documentos de la tienda: contratos, catálogo de tallas y precios, garantías y préstamo",
    # Citar la fuente y admitir cuando no hay datos: así se ve qué sale de los documentos y qué no
    instructions=(
        "Para preguntas sobre contratos, catálogo, garantías o el préstamo usa buscar_documentos "
        "y cita la fuente entre corchetes, por ejemplo [06_catalogo_tallas_precios-1]. "
        "Si no hay información suficiente para responder, di que no lo encuentras, no lo inventes."
    ),
)


@documentos.tool_plain
def buscar_documentos(pregunta: str) -> list[dict]:
    """Busca en los documentos los fragmentos que responden a la pregunta."""
    resultados = buscar(pregunta, k=3)
    for r in resultados:  # en la terminal (o en los Logs de Render) se ve qué recuperó
        print(f"  📄 {r['fuente']} (similitud {r['similitud']})", flush=True)
    return resultados
