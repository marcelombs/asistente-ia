"""Pruebas sin modelo real ni internet: uv run pytest

Un modelo falso (FunctionModel) pide la tool que queremos, y comprobamos lo que hace el código.
"""

import os
from types import SimpleNamespace

os.environ.setdefault("GEMINI_API_KEY_LOCAL", "clave-de-prueba")  # para importar sin .env (no se llama a Gemini)

import httpx  # noqa: E402
import pytest  # noqa: E402
from pydantic import SecretStr  # noqa: E402
from pydantic_ai import ModelResponse, TextPart, ToolCallPart, ToolReturnPart  # noqa: E402
from pydantic_ai.models.function import FunctionModel  # noqa: E402

import api  # noqa: E402
import asistente  # noqa: E402
import bot  # noqa: E402
import capacidades  # noqa: E402
from config import settings  # noqa: E402


@pytest.fixture(autouse=True)
def base_temporal(tmp_path, monkeypatch):
    """Cada prueba usa su propia tienda.db, no la de datos/."""
    monkeypatch.setattr(capacidades, "DB", tmp_path / "tienda.db")


def pide(tool: str, args: dict) -> FunctionModel:
    """Un modelo falso: la 1.ª vez pide la tool; cuando recibe su resultado, responde texto."""

    def responder(mensajes, info):
        ultimo = mensajes[-1].parts[-1]
        if isinstance(ultimo, ToolReturnPart):
            return ModelResponse(parts=[TextPart(f"Resultado: {ultimo.content}")])
        return ModelResponse(parts=[ToolCallPart(tool, args)])

    return FunctionModel(responder)


# ── Gastos de la tienda ──


def test_consultar_gastos_sin_modelo():
    gastos = capacidades.consultar_gastos("transporte")  # una tool es una función normal
    assert gastos["total_bs"] == 60


async def test_registrar_gasto_escribe_en_la_base():
    with asistente.asistente.override(model=pide("registrar_gasto", {
        "fecha": "2026-11-07", "categoria": "marketing", "descripcion": "Flyers", "monto": 30,
    })):
        respuesta = await asistente.responder("Anota 30 Bs de marketing.")
    assert "Gasto registrado" in respuesta
    assert capacidades.consultar_gastos("marketing")["total_bs"] == 150  # 120 del CSV + 30


async def test_monto_negativo_no_se_guarda():
    with asistente.asistente.override(model=pide("registrar_gasto", {
        "fecha": "2026-11-07", "categoria": "marketing", "descripcion": "Error", "monto": -5,
    })):
        respuesta = await asistente.responder("Anota -5 Bs.")
    assert "No se guardó" in respuesta
    assert capacidades.consultar_gastos("marketing")["total_bs"] == 120


# ── Inventario: stock, ventas y compras ──


def test_consultar_stock():
    stock = capacidades.consultar_stock("CH-001", "M")
    assert stock[0]["cantidad"] == 12
    assert stock[0]["precio"] == 180


def test_consultar_stock_sin_coincidencia():
    resultado = capacidades.consultar_stock("CH-001", "XXL")
    assert "error" in resultado


async def test_registrar_venta_descuenta_stock():
    with asistente.asistente.override(model=pide("registrar_venta", {
        "fecha": "2026-11-07", "modelo": "CH-001", "talla": "M", "cantidad": 2,
    })):
        respuesta = await asistente.responder("Vendí 2 chompas CH-001 talla M.")
    assert "Venta registrada" in respuesta
    assert capacidades.consultar_stock("CH-001", "M")[0]["cantidad"] == 10  # 12 - 2
    ventas = capacidades.consultar_ventas(modelo="CH-001")
    assert ventas[0]["total"] == 360  # 2 x 180


async def test_venta_sin_stock_no_se_registra():
    with asistente.asistente.override(model=pide("registrar_venta", {
        "fecha": "2026-11-07", "modelo": "CH-003", "talla": "L", "cantidad": 99,
    })):
        respuesta = await asistente.responder("Vendí 99 CH-003 talla L.")
    assert "No se registró" in respuesta
    assert capacidades.consultar_stock("CH-003", "L")[0]["cantidad"] == 5  # sin cambios
    assert "error" in capacidades.consultar_ventas(modelo="CH-003")  # no se anotó nada


async def test_registrar_compra_suma_stock():
    with asistente.asistente.override(model=pide("registrar_compra", {
        "fecha": "2026-11-07", "modelo": "CH-005", "talla": "S", "cantidad": 12,
    })):
        respuesta = await asistente.responder("Llegó un pedido de 12 CH-005 talla S.")
    assert "Compra registrada" in respuesta
    assert capacidades.consultar_stock("CH-005", "S")[0]["cantidad"] == 17  # 5 + 12
    compras = capacidades.consultar_compras(modelo="CH-005")
    assert compras[0]["total"] == 3600  # 12 x 300 (costo del proveedor)


# ── Bot de Telegram ──


async def test_chat_ajeno_se_rechaza_sin_llamar_al_modelo(monkeypatch):
    monkeypatch.setattr(settings, "telegram_permitidos", "123")
    respuestas = []

    async def reply_text(texto):
        respuestas.append(texto)

    def modelo_prohibido(mensajes, info):
        raise AssertionError("no debía llamarse al modelo")

    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=999), message=SimpleNamespace(text="hola", reply_text=reply_text)
    )
    with asistente.asistente.override(model=FunctionModel(modelo_prohibido)):
        await bot.mensaje(update, None)
    assert "No autorizado" in respuestas[0] and "999" in respuestas[0]


def test_permitidos(monkeypatch):
    monkeypatch.setattr(settings, "telegram_permitidos", " 123, 456 ,")
    assert settings.permitidos == {"123", "456"}
    monkeypatch.setattr(settings, "telegram_permitidos", "")
    assert settings.permitidos == set()  # vacío = nadie


# ── Extra opcional: la API (API.md) ──


async def post_chat(cuerpo: dict, token: str | None = None) -> httpx.Response:
    """Llama a la API en el mismo proceso, sin servidor ni red."""
    cabeceras = {"Authorization": f"Bearer {token}"} if token else {}
    transporte = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transporte, base_url="http://prueba") as cliente:
        return await cliente.post("/chat", json=cuerpo, headers=cabeceras)


async def test_api_sin_token_valido_no_llama_al_modelo(monkeypatch):
    def modelo_prohibido(mensajes, info):
        raise AssertionError("no debía llamarse al modelo")

    with asistente.asistente.override(model=FunctionModel(modelo_prohibido)):
        monkeypatch.setattr(settings, "api_tokens", SecretStr(""))
        assert (await post_chat({"mensaje": "hola"}, "cualquiera")).status_code == 401  # vacío = nadie
        monkeypatch.setattr(settings, "api_tokens", SecretStr("abc, def"))
        assert (await post_chat({"mensaje": "hola"})).status_code == 401
        assert (await post_chat({"mensaje": "hola"}, "xyz")).status_code == 401
        assert (await post_chat({"mensaje": ""}, "abc")).status_code == 422  # Pydantic valida el pedido


async def test_api_con_token_responde(monkeypatch):
    monkeypatch.setattr(settings, "api_tokens", SecretStr("abc,def"))
    with asistente.asistente.override(model=pide("consultar_gastos", {"categoria": "transporte"})):
        r = await post_chat({"mensaje": "¿Cuánto gasté en transporte?"}, "def")
    assert r.status_code == 200
    assert "60" in r.json()["texto"]
