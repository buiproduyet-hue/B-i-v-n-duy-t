#!/usr/bin/env python3
"""LocalMind: a small, dependency-free web app and llama.cpp gateway."""

from __future__ import annotations

import base64
import hashlib
import hmac
import io
import ipaddress
import json
import mimetypes
import os
import re
import secrets
import sqlite3
import threading
import time
import uuid
import zipfile
from datetime import datetime, timezone
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import chain
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, unquote, urlencode, urlsplit
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("LOCALMIND_DATA_DIR", ROOT / "data")).resolve()
DB_PATH = DATA_DIR / "localmind.sqlite3"
DEFAULT_LLM_BASE = os.environ.get("LLAMA_BASE_URL", "http://127.0.0.1:8090").rstrip("/")
HOST = os.environ.get("LOCALMIND_HOST", "0.0.0.0")
PORT = int(os.environ.get("LOCALMIND_PORT", "8091"))
SESSION_DAYS = 14
MAX_BODY = 16 * 1024 * 1024
MAX_IMAGE_BYTES = 10 * 1024 * 1024
PASSWORD_ITERATIONS = 310_000
LANGUAGE_NAMES = {
    "auto": "Auto (the user's language)", "vi": "Vietnamese", "en": "English", "zh-CN": "Chinese (Simplified)",
    "zh-TW": "Chinese (Traditional)", "ja": "Japanese", "ko": "Korean", "th": "Thai", "fr": "French",
    "de": "German", "es": "Spanish", "pt": "Portuguese", "it": "Italian", "ru": "Russian", "ar": "Arabic",
    "hi": "Hindi", "id": "Indonesian", "ms": "Malay", "tr": "Turkish", "nl": "Dutch", "pl": "Polish",
    "uk": "Ukrainian", "fa": "Persian", "bn": "Bengali", "ur": "Urdu",
}
SENSITIVE_ARGUMENT_KEY = re.compile(r"password|passwd|secret|token|api[_-]?key|authorization|cookie|private[_-]?key", re.I)
SETUP_LOCK = threading.Lock()
SETUP_TOKEN = os.environ.get("LOCALMIND_SETUP_TOKEN") or secrets.token_urlsafe(9)
ACTIVE_GENERATIONS: dict[str, dict] = {}
ACTIVE_GENERATIONS_LOCK = threading.Lock()
MCP_PROTOCOL_VERSION = "2026-07-28"
MCP_MAX_RESPONSE = 16 * 1024 * 1024
GITHUB_API_VERSION = "2026-03-10"
GITHUB_DEVICE_FLOWS: dict[str, dict] = {}
GITHUB_DEVICE_FLOWS_LOCK = threading.Lock()
GITHUB_SYNC_TIMERS: dict[str, threading.Timer] = {}
GITHUB_SYNC_LOCK = threading.Lock()
GITHUB_SYNC_ACTIVE: set[str] = set()
GITHUB_SYNC_PENDING: dict[str, tuple[str, bool]] = {}
GITHUB_TOKEN_LOCKS: dict[str, threading.Lock] = {}
GITHUB_TOKEN_LOCKS_GUARD = threading.Lock()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_context_messages(history_rows, memory_items: list[str], global_settings: dict) -> list[dict]:
    messages = []
    global_prompt = (global_settings.get("ai_system_prompt") or "").strip()
    if global_prompt:
        messages.append({"role": "system", "content": global_prompt})
    language = global_settings.get("ai_language", "auto")
    if language != "auto":
        language_name = LANGUAGE_NAMES.get(language, "Auto (the user's language)")
        messages.append({"role": "system", "content": f"Ngôn ngữ phản hồi bắt buộc trên toàn hệ thống: {language_name}. Hãy trả lời bằng ngôn ngữ này trong mọi lượt, kể cả khi user nhập ngôn ngữ khác. Chỉ dùng ngôn ngữ khác cho phần nội dung cụ thể khi user yêu cầu dịch hoặc viết nội dung bằng ngôn ngữ đó."})
    if memory_items:
        messages.append({"role": "system", "content": "Thông tin ghi nhớ lâu dài riêng của tài khoản hiện tại. Dùng khi phù hợp; nếu mâu thuẫn với yêu cầu mới nhất của user thì ưu tiên yêu cầu mới. Không tự nhắc lại thông tin này nếu không liên quan.\n- " + "\n- ".join(memory_items)})
    for row in history_rows:
        content = row["content"]
        attachments = json.loads(row["attachments"] or "[]")
        if attachments:
            parts = []
            if content:
                parts.append({"type": "text", "text": content})
            for image in attachments:
                parts.append({"type": "image_url", "image_url": {"url": f"data:{image['type']};base64,{image['data']}"}})
            content = parts
        messages.append({"role": row["role"], "content": content})
    return messages


def model_context_limit(api_base: str) -> int | None:
    _, props = read_upstream_json(api_base + "/props", timeout=3)
    if not isinstance(props, dict):
        return None
    settings = props.get("default_generation_settings") or {}
    params = settings.get("params") or {}
    value = settings.get("n_ctx") or params.get("n_ctx")
    try:
        return int(value) if int(value) > 0 else None
    except (TypeError, ValueError):
        return None


def count_model_input_tokens(api_base: str, model_id: str, messages: list[dict], tools: list[dict] | None = None) -> int | None:
    payload = {"model": model_id, "messages": messages}
    if tools:
        payload["tools"] = tools
    request = Request(
        api_base + "/v1/chat/completions/input_tokens",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=3) as response:
            result = json.loads(response.read(1024 * 1024).decode("utf-8", "replace"))
        value = result.get("input_tokens") if isinstance(result, dict) else None
        return int(value) if value is not None and int(value) >= 0 else None
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def redact_tool_arguments(raw: str) -> str:
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError:
        value = str(raw or "")[:4000]

    def redact(item):
        if isinstance(item, dict):
            return {key: "[đã ẩn]" if SENSITIVE_ARGUMENT_KEY.search(str(key)) else redact(child) for key, child in item.items()}
        if isinstance(item, list):
            return [redact(child) for child in item]
        if isinstance(item, str):
            return re.sub(r"(?i)\b(bearer\s+)[^\s,;]+|\b(password|passwd|secret|token|api[_-]?key)\s*[:=]\s*[^\s,;]+", lambda match: (match.group(1) or "") + "[đã ẩn]" if match.group(1) else match.group(2) + "=[đã ẩn]", item)[:4000]
        return item

    return json.dumps(redact(value), ensure_ascii=False, indent=2)[:4000]


def summarize_tool_result(content: str | list[dict]) -> tuple[str, str | None]:
    texts = []
    images = []
    if isinstance(content, list):
        for part in content:
            if part.get("type") == "text":
                texts.append(str(part.get("text", "")))
            elif part.get("type") == "image_url":
                image_url = (part.get("image_url") or {}).get("url")
                if isinstance(image_url, str) and image_url.startswith("data:image/"):
                    images.append(image_url)
    else:
        texts.append(str(content))
    description = "\n".join(text for text in texts if text).strip()
    if images:
        description = f"Tool trả về {len(images)} ảnh; ảnh đã được chuyển riêng cho model để quan sát." + (f"\n{description}" if description else "")
    safe_preview = redact_tool_arguments(json.dumps({"output": description or "Tool hoàn tất nhưng không có nội dung văn bản."}, ensure_ascii=False))[:1800]
    preview_image = images[0] if images and len(images[0]) <= 1_500_000 else None
    if images and not preview_image:
        safe_preview += "\nẢnh quá lớn để hiển thị trong nhật ký chat."
    return safe_preview, preview_image


def db() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=20)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_db() -> None:
    with db() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                display_name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('admin','user')),
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token_hash TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                expires_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS conversations_by_user ON conversations(user_id, updated_at DESC);
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                role TEXT NOT NULL CHECK(role IN ('user','assistant')),
                content TEXT NOT NULL,
                attachments TEXT NOT NULL DEFAULT '[]',
                mode TEXT,
                status TEXT NOT NULL DEFAULT 'complete',
                feedback INTEGER,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS messages_by_conversation ON messages(conversation_id, created_at);
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS user_memories (
                user_id TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                content TEXT NOT NULL DEFAULT '[]',
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS tool_registry (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE COLLATE NOCASE,
                server TEXT NOT NULL,
                description TEXT NOT NULL,
                input_schema TEXT NOT NULL DEFAULT '{}',
                enabled INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS mcp_servers (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE COLLATE NOCASE,
                url TEXT NOT NULL UNIQUE,
                session_id TEXT,
                protocol_version TEXT NOT NULL DEFAULT '2025-11-25',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS tool_activity (
                id TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                server_name TEXT NOT NULL,
                status TEXT NOT NULL,
                duration_ms INTEGER NOT NULL DEFAULT 0,
                error_kind TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS projects_user_recent ON projects(user_id,updated_at DESC);
            CREATE TABLE IF NOT EXISTS project_files (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                path TEXT NOT NULL,
                content TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(project_id,path)
            );
            CREATE TABLE IF NOT EXISTS project_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                path TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'edit'
            );
            CREATE INDEX IF NOT EXISTS project_versions_recent ON project_versions(project_id,path,id DESC);
            CREATE TABLE IF NOT EXISTS github_accounts (
                user_id TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                login TEXT NOT NULL COLLATE NOCASE UNIQUE,
                connected_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS github_projects (
                project_id TEXT PRIMARY KEY REFERENCES projects(id) ON DELETE CASCADE,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                owner TEXT NOT NULL,
                repo TEXT NOT NULL,
                branch TEXT NOT NULL DEFAULT 'main',
                repo_url TEXT NOT NULL,
                auto_sync INTEGER NOT NULL DEFAULT 1,
                last_commit_sha TEXT NOT NULL DEFAULT '',
                last_synced_at TEXT,
                sync_status TEXT NOT NULL DEFAULT 'idle',
                sync_error TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS tool_activity_recent ON tool_activity(created_at DESC);
            """
        )
        connection.execute(
            "INSERT OR IGNORE INTO settings(key,value) VALUES('llama_base_url',?)",
            (DEFAULT_LLM_BASE,),
        )
        connection.execute("INSERT OR IGNORE INTO settings(key,value) VALUES('app_name','LocalMind')")
        connection.execute("INSERT OR IGNORE INTO settings(key,value) VALUES('accent_color','#b3a0ff')")
        connection.execute("INSERT OR IGNORE INTO settings(key,value) VALUES('ai_display_name','Gemma 4 26B')")
        connection.execute("INSERT OR IGNORE INTO settings(key,value) VALUES('ai_language','auto')")
        connection.execute("INSERT OR IGNORE INTO settings(key,value) VALUES('ai_system_prompt','')")
        connection.execute("INSERT OR IGNORE INTO settings(key,value) VALUES('ai_tools_enabled','0')")
        connection.execute("INSERT OR IGNORE INTO settings(key,value) VALUES('github_app_client_id','')")
        tool_columns = {row["name"] for row in connection.execute("PRAGMA table_info(tool_registry)").fetchall()}
        if "input_schema" not in tool_columns:
            connection.execute("ALTER TABLE tool_registry ADD COLUMN input_schema TEXT NOT NULL DEFAULT '{}' ")
        if "mcp_tool_name" not in tool_columns:
            connection.execute("ALTER TABLE tool_registry ADD COLUMN mcp_tool_name TEXT")
        server_columns = {row["name"] for row in connection.execute("PRAGMA table_info(mcp_servers)").fetchall()}
        if "session_id" not in server_columns:
            connection.execute("ALTER TABLE mcp_servers ADD COLUMN session_id TEXT")
        if "protocol_version" not in server_columns:
            connection.execute("ALTER TABLE mcp_servers ADD COLUMN protocol_version TEXT NOT NULL DEFAULT '2025-11-25'")
        connection.execute("UPDATE messages SET status='partial' WHERE status='pending'")


def password_hash(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS)
    return f"pbkdf2_sha256${PASSWORD_ITERATIONS}${base64.b64encode(salt).decode()}${base64.b64encode(digest).decode()}"


def password_matches(password: str, stored: str) -> bool:
    try:
        scheme, iterations, salt64, hash64 = stored.split("$", 3)
        if scheme != "pbkdf2_sha256":
            return False
        salt = base64.b64decode(salt64)
        expected = base64.b64decode(hash64)
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(iterations))
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


def user_public(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "username": row["username"],
        "display_name": row["display_name"],
        "role": row["role"],
        "active": bool(row["active"]),
        "created_at": row["created_at"],
    }


def safe_base_url(value: str) -> str:
    value = value.strip().rstrip("/")
    parsed = urlsplit(value)
    if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("Địa chỉ API phải bắt đầu bằng http:// hoặc https:// và không chứa thông tin đăng nhập.")
    if parsed.path.endswith("/v1"):
        value = value[:-3]
    return value.rstrip("/")


def validate_mcp_url(value: str) -> str:
    value = str(value or "").strip()
    parsed = urlsplit(value)
    if parsed.scheme != "http" or not parsed.hostname or parsed.username or parsed.password or parsed.fragment:
        raise ValueError("MCP cần URL HTTP nội bộ dạng http://IP:PORT/mcp.")
    try:
        address = ipaddress.ip_address(parsed.hostname)
    except ValueError as exc:
        raise ValueError("Dùng địa chỉ IP nội bộ trực tiếp, không dùng hostname.") from exc
    allowed_networks = (ipaddress.ip_network("10.0.0.0/8"), ipaddress.ip_network("172.16.0.0/12"), ipaddress.ip_network("192.168.0.0/16")) if address.version == 4 else (ipaddress.ip_network("fc00::/7"),)
    if not any(address in network for network in allowed_networks):
        raise ValueError("MCP chỉ được kết nối tới IP riêng trong mạng nội bộ.")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("Cổng MCP không hợp lệ.") from exc
    if port is None or not 1 <= port <= 65535:
        raise ValueError("URL MCP cần chỉ rõ cổng.")
    return value.rstrip("/")


def _mcp_read_response(response) -> tuple[dict, str | None]:
    content_type = response.headers.get("Content-Type", "")
    raw = response.read(MCP_MAX_RESPONSE + 1)
    if len(raw) > MCP_MAX_RESPONSE:
        raise ValueError("Phản hồi MCP vượt quá giới hạn 16 MB.")
    if response.status == 202 and not raw:
        return {}, response.headers.get("Mcp-Session-Id")
    text = raw.decode("utf-8", "replace")
    if "text/event-stream" in content_type:
        candidates = []
        for block in re.split(r"\r?\n\r?\n", text):
            data_lines = [line[5:].lstrip() for line in block.splitlines() if line.startswith("data:")]
            if data_lines:
                candidates.append("\n".join(data_lines))
        payload = None
        for candidate in reversed(candidates):
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict) and ("result" in parsed or "error" in parsed):
                    payload = parsed
                    break
            except json.JSONDecodeError:
                continue
        if payload is None:
            raise ValueError("MCP SSE không chứa JSON-RPC response.")
    else:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("MCP trả về phản hồi không phải JSON hợp lệ.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Phản hồi MCP không đúng định dạng JSON-RPC.")
    return payload, response.headers.get("Mcp-Session-Id")


def _mcp_header_value(value) -> str:
    rendered = "true" if value is True else "false" if value is False else str(value)
    if rendered.isascii() and all(0x20 <= ord(char) <= 0x7E or char == "\t" for char in rendered) and rendered == rendered.strip() and not (rendered.startswith("=?base64?") and rendered.endswith("?=")):
        return rendered
    encoded = base64.b64encode(rendered.encode("utf-8")).decode("ascii")
    return f"=?base64?{encoded}?="


def mcp_post(url: str, message: dict, session_id: str | None = None, protocol_version: str = MCP_PROTOCOL_VERSION, active_state: dict | None = None, param_headers: dict | None = None) -> tuple[dict, str | None]:
    url = validate_mcp_url(url)
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream", "MCP-Protocol-Version": protocol_version}
    if protocol_version == "2026-07-28":
        headers["Mcp-Method"] = str(message.get("method", ""))
        params = message.setdefault("params", {})
        if not isinstance(params, dict):
            raise ValueError("MCP request params không hợp lệ.")
        if message.get("method") == "tools/call" and params.get("name"):
            headers["Mcp-Name"] = _mcp_header_value(params["name"])
        for header_name, header_value in (param_headers or {}).items():
            headers[f"Mcp-Param-{header_name}"] = _mcp_header_value(header_value)
        params["_meta"] = {
            "io.modelcontextprotocol/protocolVersion": protocol_version,
            "io.modelcontextprotocol/clientInfo": {"name": "LocalMind", "version": "1.0"},
            "io.modelcontextprotocol/clientCapabilities": {},
        }
    if session_id:
        headers["Mcp-Session-Id"] = session_id
    request = Request(url, data=json.dumps(message, ensure_ascii=False).encode("utf-8"), headers=headers, method="POST")
    with urlopen(request, timeout=25) as response:
        if active_state is not None:
            active_state["tool_response"] = response
            if not get_tools_enabled():
                response.close()
                raise RuntimeError("Tools bị Admin tắt trong lúc gửi request MCP.")
        try:
            return _mcp_read_response(response)
        finally:
            if active_state is not None:
                active_state["tool_response"] = None


def mcp_rpc(server: dict, method: str, params: dict | None = None, active_state: dict | None = None, param_headers: dict | None = None) -> tuple[dict, str | None]:
    request_id = uuid.uuid4().hex
    message = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        message["params"] = params
    payload, session = mcp_post(server["url"], message, server.get("session_id"), server.get("protocol_version") or MCP_PROTOCOL_VERSION, active_state, param_headers)
    if payload.get("error"):
        error = payload["error"]
        raise RuntimeError(str(error.get("message") or "MCP JSON-RPC error"))
    if payload.get("id") not in (request_id, None):
        raise RuntimeError("MCP trả về sai request id.")
    return payload.get("result") or {}, session


def mcp_initialize(url: str) -> tuple[dict, str, str]:
    version = "2025-11-25"
    result, session = mcp_rpc({"url": url, "session_id": None, "protocol_version": version}, "initialize", {
        "protocolVersion": version,
        "capabilities": {},
        "clientInfo": {"name": "LocalMind", "version": "1.0"},
    })
    version = result.get("protocolVersion") or version
    if not isinstance(version, str) or len(version) > 40:
        raise ValueError("MCP server trả protocol version không hợp lệ.")
    # Initialized is a notification and has no JSON-RPC response body.
    notify = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    mcp_post(url, notify, session, version)
    return result, session or "", version


def mcp_list_all_tools(server: dict) -> dict:
    tools = []
    cursor = None
    result = {}
    for _ in range(20):
        params = {"cursor": cursor} if cursor else None
        result, session = mcp_rpc(server, "tools/list", params)
        if session:
            server["session_id"] = session
        page = result.get("tools") or []
        if not isinstance(page, list):
            raise ValueError("MCP trả danh sách tools không hợp lệ.")
        tools.extend(page)
        if len(tools) > 500:
            raise ValueError("MCP server có quá 500 tools; giới hạn dò là 500.")
        cursor = result.get("nextCursor")
        if not cursor:
            break
    else:
        raise ValueError("MCP server phân trang quá nhiều lần.")
    result["tools"] = tools
    return result


def mcp_argument_headers(schema: dict, arguments: dict) -> dict:
    collected = {}

    def visit(node, values):
        if not isinstance(node, dict):
            return
        properties = node.get("properties") or {}
        if not isinstance(properties, dict):
            return
        for key, child in properties.items():
            if not isinstance(child, dict):
                continue
            header_name = child.get("x-mcp-header")
            if header_name:
                if not isinstance(header_name, str) or not re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", header_name):
                    raise ValueError("MCP tool có tên x-mcp-header không hợp lệ.")
                value = values.get(key) if isinstance(values, dict) else None
                if value is not None:
                    if not isinstance(value, (str, int, bool)) or isinstance(value, float):
                        raise ValueError("Giá trị x-mcp-header phải là chuỗi, số nguyên hoặc boolean.")
                    if header_name.lower() in {name.lower() for name in collected}:
                        raise ValueError("MCP tool khai báo trùng x-mcp-header.")
                    collected[header_name] = value
            elif isinstance(child.get("properties"), dict):
                visit(child, values.get(key, {}) if isinstance(values, dict) else {})

    visit(schema, arguments)
    return collected


def get_tools_enabled() -> bool:
    with db() as connection:
        row = connection.execute("SELECT value FROM settings WHERE key='ai_tools_enabled'").fetchone()
    return bool(row and row["value"] == "1")


def workspace_tool_definitions(include_create: bool = False, include_github: bool = False) -> list[dict]:
    defs = []
    if include_create:
        defs.append({"type":"function","function":{"name":"workspace_create_project","description":"Create a project in the user's private LocalMind workspace when no project is open.","parameters":{"type":"object","properties":{"name":{"type":"string","description":"Short project name"}},"required":["name"]}}})
    defs.extend([
        {"type":"function","function":{"name":"workspace_list_files","description":"List files in the currently selected private project.","parameters":{"type":"object","properties":{}}}},
        {"type":"function","function":{"name":"workspace_read_file","description":"Read a text file from the currently selected private project. Large files can be read in chunks using offset and limit.","parameters":{"type":"object","properties":{"path":{"type":"string"},"offset":{"type":"integer","minimum":0},"limit":{"type":"integer","minimum":1,"maximum":120000}},"required":["path"]}}},
        {"type":"function","function":{"name":"workspace_write_file","description":"Create or update one text file in the currently selected private project. The app autosaves and keeps a version snapshot.","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}},
    ])
    if include_github:
        defs.extend([
            {"type":"function","function":{"name":"workspace_github_status","description":"Check whether the current LocalMind user linked GitHub and whether this selected workspace project has its own private repository. Only reports the current user's connection and selected project.","parameters":{"type":"object","properties":{}}}},
            {"type":"function","function":{"name":"workspace_github_sync","description":"Queue an immediate sync of this selected workspace project to its already-linked GitHub repository. Does not create or delete repositories. Only works with the current user's linked private repository.","parameters":{"type":"object","properties":{}}}},
        ])
    return defs


def safe_project_path(value: str) -> str:
    raw = str(value or "").replace("\\", "/").strip()
    parts = [part for part in raw.split("/") if part not in ("", ".")]
    if not parts or len(raw) > 240 or raw.startswith("/") or ":" in raw or any(part == ".." or part.casefold() == ".git" for part in parts):
        raise ValueError("Đường dẫn tệp không hợp lệ.")
    path = "/".join(parts)
    if any(ord(ch) < 32 for ch in path):
        raise ValueError("Đường dẫn tệp chứa ký tự không hợp lệ.")
    return path


def trim_project_versions(connection: sqlite3.Connection, project_id: str, path: str) -> None:
    connection.execute("DELETE FROM project_versions WHERE project_id=? AND path=? AND id NOT IN (SELECT id FROM project_versions WHERE project_id=? AND path=? ORDER BY id DESC LIMIT 30)", (project_id,path,project_id,path))
    while True:
        totals = connection.execute("SELECT COUNT(*) count,COALESCE(SUM(length(CAST(content AS BLOB))),0) bytes FROM project_versions WHERE project_id=?", (project_id,)).fetchone()
        if totals["count"] <= 100 and totals["bytes"] <= 100_000_000:
            break
        oldest = connection.execute("SELECT id FROM project_versions WHERE project_id=? ORDER BY id ASC LIMIT 1", (project_id,)).fetchone()
        if not oldest:
            break
        connection.execute("DELETE FROM project_versions WHERE id=?", (oldest["id"],))


def execute_workspace_tool(tool_call: dict, active_state: dict) -> tuple[str, str, dict | None]:
    fn = tool_call.get("function") or {}
    name = str(fn.get("name", ""))
    try:
        args = json.loads(fn.get("arguments") or "{}")
        if not isinstance(args, dict):
            raise ValueError("Arguments phải là object.")
        user_id = active_state["user_id"]
        project_id = active_state.get("project_id")
        if name == "workspace_create_project":
            if project_id:
                return "Đã có dự án đang mở; tiếp tục dùng dự án đó.", "complete", None
            title = str(args.get("name", "Dự án mới")).strip()[:80] or "Dự án mới"
            project_id = uuid.uuid4().hex
            stamp = now_iso()
            with db() as connection:
                connection.execute("INSERT INTO projects(id,user_id,name,created_at,updated_at) VALUES(?,?,?,?,?)", (project_id,user_id,title,stamp,stamp))
            active_state["project_id"] = project_id
            project = {"id": project_id, "name": title}
            return f"Đã tạo dự án '{title}' trong workspace riêng của user. Hãy dùng workspace_write_file để lưu các tệp.", "complete", project
        if not project_id:
            return "Chưa có workspace dự án đang mở. Hãy yêu cầu user tạo/chọn dự án trước.", "blocked", None
        with db() as connection:
            owned = connection.execute("SELECT id FROM projects WHERE id=? AND user_id=?", (project_id,user_id)).fetchone()
        if not owned:
            return "Không tìm thấy dự án trong workspace của user hiện tại.", "blocked", None
        if name in ("workspace_github_status", "workspace_github_sync"):
            with db() as connection:
                account = connection.execute("SELECT login FROM github_accounts WHERE user_id=?", (user_id,)).fetchone()
                repository = connection.execute(
                    "SELECT owner,repo,branch,repo_url,auto_sync,sync_status,sync_error,last_synced_at "
                    "FROM github_projects WHERE project_id=? AND user_id=?", (project_id,user_id)
                ).fetchone()
            if name == "workspace_github_status":
                status = {"connected": bool(account), "login": account["login"] if account else None,
                          "repository": dict(repository) if repository else None,
                          "scope": "current LocalMind user and selected private workspace project"}
                return json.dumps(status, ensure_ascii=False), "complete", None
            if not account:
                return "GitHub chưa liên kết với tài khoản LocalMind hiện tại.", "blocked", None
            if not repository:
                return "Dự án hiện tại chưa liên kết repo GitHub. Hãy liên kết GitHub và tạo repo riêng trong Workspace trước.", "blocked", None
            if not schedule_github_sync(project_id, user_id, force=True):
                return "Không thể đưa dự án vào hàng đợi đồng bộ GitHub.", "failed", None
            return "Đã đưa dự án hiện tại vào hàng đợi đồng bộ GitHub. Backend sẽ kiểm tra repo vẫn ở chế độ private trước khi ghi.", "complete", None
        with db() as connection:
            if name == "workspace_list_files":
                rows = connection.execute("SELECT path,length(CAST(content AS BLOB)) size FROM project_files WHERE project_id=? ORDER BY path", (project_id,)).fetchall()
                return json.dumps([dict(row) for row in rows], ensure_ascii=False), "complete", None
            path = safe_project_path(args.get("path", ""))
            if name == "workspace_read_file":
                row = connection.execute("SELECT content FROM project_files WHERE project_id=? AND path=?", (project_id,path)).fetchone()
                if not row:
                    return f"Không tìm thấy tệp {path}.", "failed", None
                content = row["content"]
                offset = max(0, min(len(content), int(args.get("offset", 0))))
                limit = max(1, min(120_000, int(args.get("limit", 120_000))))
                chunk = content[offset:offset+limit]
                suffix = f"\n[Đang hiển thị ký tự {offset}–{offset+len(chunk)} / {len(content)}; gọi lại với offset để đọc phần sau.]" if offset + len(chunk) < len(content) else ""
                return chunk + suffix, "complete", None
            if name == "workspace_write_file":
                content = args.get("content")
                if not isinstance(content, str) or len(content.encode("utf-8")) > 1_000_000:
                    return "Tệp cần là văn bản và không vượt quá 1 MB.", "blocked", None
                used = connection.execute("SELECT COALESCE(SUM(length(CAST(content AS BLOB))),0) FROM project_files WHERE project_id=?", (project_id,)).fetchone()[0]
                old = connection.execute("SELECT id,length(CAST(content AS BLOB)) size FROM project_files WHERE project_id=? AND path=?", (project_id,path)).fetchone()
                file_count = connection.execute("SELECT COUNT(*) FROM project_files WHERE project_id=?", (project_id,)).fetchone()[0]
                if not old and file_count >= 500:
                    return "Dự án đạt giới hạn 500 tệp.", "blocked", None
                if used - (old["size"] if old else 0) + len(content.encode("utf-8")) > 20_000_000:
                    return "Dự án đã đạt giới hạn lưu trữ 20 MB.", "blocked", None
                stamp = now_iso()
                file_id = old["id"] if old else uuid.uuid4().hex
                connection.execute("INSERT INTO project_files(id,project_id,path,content,updated_at) VALUES(?,?,?,?,?) ON CONFLICT(project_id,path) DO UPDATE SET content=excluded.content,updated_at=excluded.updated_at", (file_id,project_id,path,content,stamp))
                connection.execute("INSERT INTO project_versions(project_id,path,content,created_at,source) VALUES(?,?,?,?,?)", (project_id,path,content,stamp,"ai"))
                trim_project_versions(connection, project_id, path)
                connection.execute("UPDATE projects SET updated_at=? WHERE id=?", (stamp,project_id))
                write_result = (f"Đã lưu {path} vào workspace; phiên bản này đã được ghi lại.", "complete", {"id":project_id,"name":"","path":path})
        if name == "workspace_write_file":
            schedule_github_sync(project_id, user_id)
            return write_result
        return "Workspace tool không hợp lệ.", "blocked", None
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        return f"Workspace tool báo lỗi: {str(exc)[:240]}", "failed", None


def iter_model_stream(api_base: str, payload: dict, active_state: dict):
    request = Request(
        api_base + "/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    calls: dict[int, dict] = {}
    with urlopen(request, timeout=300) as upstream:
        active_state["response"] = upstream
        try:
            if active_state["stop"].is_set():
                upstream.close()
                return
            while not active_state["stop"].is_set():
                # Keep urllib's buffered SSE reader intact. A 1s socket timeout here can
                # poison its buffered stream and raise "cannot read from timed out object".
                raw_line = upstream.readline()
                if not raw_line:
                    break
                line = raw_line.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                value = line[5:].strip()
                if value == "[DONE]":
                    break
                try:
                    event = json.loads(value)
                except json.JSONDecodeError:
                    continue
                if event.get("error"):
                    raise RuntimeError(str(event["error"]))
                if isinstance(event.get("usage"), dict):
                    yield "usage", {"usage": event["usage"], "timings": event.get("timings") or {}}
                choices = event.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                if reasoning:
                    yield "thinking", reasoning
                token = delta.get("content")
                if isinstance(token, str) and token:
                    yield "token", token
                for item in delta.get("tool_calls") or []:
                    if not isinstance(item, dict):
                        continue
                    index = int(item.get("index", 0))
                    call = calls.setdefault(index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                    if item.get("id"):
                        call["id"] = item["id"]
                    function = item.get("function") or {}
                    if function.get("name"):
                        call["function"]["name"] += function["name"]
                    if function.get("arguments"):
                        call["function"]["arguments"] += function["arguments"]
        finally:
            active_state["response"] = None
    yield "complete", [calls[i] for i in sorted(calls)]


def record_tool_activity(active_state: dict, tool_name: str, server_name: str, status: str, started: float, error_kind: str = "") -> None:
    try:
        with db() as connection:
            connection.execute(
                "INSERT INTO tool_activity(id,username,conversation_id,tool_name,server_name,status,duration_ms,error_kind,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (uuid.uuid4().hex, active_state.get("username", ""), active_state.get("conversation_id", ""), tool_name[:128], server_name[:80], status, int((time.monotonic() - started) * 1000), error_kind[:80], now_iso()),
            )
            connection.execute("DELETE FROM tool_activity WHERE id NOT IN (SELECT id FROM tool_activity ORDER BY created_at DESC LIMIT 500)")
    except sqlite3.Error:
        pass


def execute_mcp_tool(tool_call: dict, active_state: dict) -> tuple[str | list[dict], str]:
    function = tool_call.get("function") or {}
    public_name = str(function.get("name", ""))
    started = time.monotonic()
    try:
        arguments = json.loads(function.get("arguments") or "{}")
        if not isinstance(arguments, dict):
            raise ValueError()
    except (json.JSONDecodeError, ValueError, TypeError):
        record_tool_activity(active_state, public_name, "", "blocked", started, "Invalid arguments")
        return "Tool arguments không phải JSON object hợp lệ; không thực hiện.", "blocked"
    with db() as connection:
        tool = connection.execute(
            "SELECT t.*,s.url,s.name server_name,s.session_id,s.protocol_version FROM tool_registry t JOIN mcp_servers s ON s.id=t.server WHERE t.name=? AND t.mcp_tool_name IS NOT NULL",
            (public_name,),
        ).fetchone()
    if not tool:
        record_tool_activity(active_state, public_name, "", "blocked", started, "Not in registry")
        return "Tool không có trong danh sách MCP được LocalMind cho phép.", "blocked"
    server_name = tool["server_name"]
    tool_name = tool["mcp_tool_name"]
    if not get_tools_enabled():
        record_tool_activity(active_state, tool_name, server_name, "blocked", started, "Tools disabled")
        return "Admin vừa tắt Tools toàn hệ thống. Tool call bị chặn.", "blocked"
    if active_state["stop"].is_set():
        record_tool_activity(active_state, tool_name, server_name, "blocked", started, "Chat stopped")
        return "Lượt chat đã bị dừng; tool không được thực hiện.", "blocked"
    # Read the central switch again immediately before dispatch. This is the kill-switch boundary.
    server = {"url": tool["url"], "session_id": tool["session_id"], "protocol_version": tool["protocol_version"]}
    try:
        schema = json.loads(tool["input_schema"] or "{}")
        param_headers = mcp_argument_headers(schema, arguments) if server["protocol_version"] == "2026-07-28" else None
        result, _ = mcp_rpc(server, "tools/call", {"name": tool_name, "arguments": arguments}, active_state, param_headers)
        if not get_tools_enabled():
            record_tool_activity(active_state, tool_name, server_name, "blocked", started, "Tools disabled during call")
            return "Admin đã tắt Tools trong lúc chạy. Kết nối đã bị ngắt; trạng thái tác vụ từ MCP server có thể không xác định.", "blocked"
        content = result.get("content") or []
        parts = []
        text_parts = []
        total_image_bytes = 0
        for item in content:
            if item.get("type") == "text":
                value = str(item.get("text", ""))[:20000]
                text_parts.append(value)
                parts.append({"type": "text", "text": value})
            elif item.get("type") == "image":
                mime = str(item.get("mimeType", ""))
                encoded = item.get("data", "")
                if mime not in ("image/png", "image/jpeg", "image/webp", "image/gif") or not isinstance(encoded, str):
                    continue
                try:
                    decoded = base64.b64decode(encoded, validate=True)
                except Exception:
                    continue
                total_image_bytes += len(decoded)
                if len(decoded) > MAX_IMAGE_BYTES or total_image_bytes > MAX_IMAGE_BYTES:
                    raise ValueError("Ảnh trả về từ MCP vượt giới hạn 10 MB.")
                parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}})
            else:
                value = json.dumps(item, ensure_ascii=False)[:20000]
                text_parts.append(value)
                parts.append({"type": "text", "text": value})
        if not parts:
            fallback = json.dumps(result.get("structuredContent") or result, ensure_ascii=False)[:20000]
            parts = [{"type": "text", "text": fallback}]
            text_parts.append(fallback)
        if result.get("isError"):
            prefix = {"type": "text", "text": "[Tool báo lỗi]"}
            parts.insert(0, prefix)
            text_parts.insert(0, "[Tool báo lỗi]")
        has_image = any(part["type"] == "image_url" for part in parts)
        rendered: str | list[dict] = parts if has_image else "\n".join(text_parts)[:20000]
        status = "failed" if result.get("isError") else "complete"
        record_tool_activity(active_state, tool_name, server_name, status, started, "Tool returned isError" if result.get("isError") else "")
        return rendered, status
    except Exception as exc:
        if not get_tools_enabled():
            record_tool_activity(active_state, tool_name, server_name, "blocked", started, "Tools disabled during call")
            return "Admin đã tắt Tools; MCP request bị ngắt. Tác vụ phía MCP có thể đã bắt đầu trước khi kết nối bị đóng.", "blocked"
        record_tool_activity(active_state, tool_name, server_name, "failed", started, type(exc).__name__)
        return f"MCP tool lỗi ({type(exc).__name__}): {str(exc)[:1000]}", "failed"


def clean_memory_list(value):
    if not isinstance(value, list) or len(value) > 30:
        return None
    cleaned = []
    for item in value:
        if not isinstance(item, str):
            return None
        item = re.sub(r"\s+", " ", item).strip()[:300]
        if item and item not in cleaned:
            cleaned.append(item)
    return cleaned


def base_url() -> str:
    with db() as connection:
        row = connection.execute("SELECT value FROM settings WHERE key='llama_base_url'").fetchone()
    return safe_base_url(row["value"] if row else DEFAULT_LLM_BASE)


def read_upstream_json(url: str, timeout: int = 5) -> tuple[int, dict | list | str]:
    request = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read(2 * 1024 * 1024).decode("utf-8", "replace")
            try:
                return response.status, json.loads(raw)
            except json.JSONDecodeError:
                return response.status, raw
    except HTTPError as exc:
        try:
            detail = exc.read(4096).decode("utf-8", "replace")
        except Exception:
            detail = str(exc)
        return exc.code, {"error": detail}
    except (URLError, TimeoutError, OSError) as exc:
        return 0, {"error": str(exc)}


class GitHubError(Exception):
    def __init__(self, message: str, status: int = 0):
        super().__init__(message)
        self.status = status


def github_json_request(url: str, method: str = "GET", payload: dict | None = None,
                        access_token: str = "", form: bool = False) -> dict:
    headers = {"Accept": "application/json", "User-Agent": "LocalMind"}
    data = None
    if payload is not None:
        if form:
            data = urlencode(payload).encode("utf-8")
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        else:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    if url.startswith("https://api.github.com/"):
        headers["X-GitHub-Api-Version"] = GITHUB_API_VERSION
    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=25) as response:
            raw = response.read(24 * 1024 * 1024).decode("utf-8", "replace")
    except HTTPError as exc:
        try:
            raw = exc.read(4096).decode("utf-8", "replace")
            detail = json.loads(raw) if raw else {}
            message = detail.get("message", "") if isinstance(detail, dict) else ""
        except Exception:
            message = ""
        raise GitHubError(str(message or f"GitHub trả HTTP {exc.code}")[:300], exc.code) from None
    except (URLError, TimeoutError, OSError) as exc:
        raise GitHubError(f"Không kết nối được GitHub ({type(exc).__name__}).") from None
    if not raw.strip():
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        raise GitHubError("GitHub trả phản hồi không hợp lệ.") from None
    if not isinstance(value, dict):
        raise GitHubError("GitHub trả định dạng dữ liệu không hợp lệ.")
    return value


def github_token_path(user_id: str) -> Path:
    token_dir = DATA_DIR / "github-tokens"
    token_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(token_dir, 0o700)
    except OSError:
        pass
    return token_dir / (hashlib.sha256(user_id.encode("utf-8")).hexdigest() + ".json")


def write_github_token(user_id: str, token: dict) -> None:
    path = github_token_path(user_id)
    temporary = path.with_name(path.name + "." + secrets.token_hex(6) + ".tmp")
    try:
        temporary.write_text(json.dumps(token, separators=(",", ":")), encoding="utf-8")
        try:
            os.chmod(temporary, 0o600)
        except OSError:
            pass
        os.replace(temporary, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def github_client_id() -> str:
    with db() as connection:
        row = connection.execute("SELECT value FROM settings WHERE key='github_app_client_id'").fetchone()
    return str(row["value"] if row else "").strip()


def github_user_token_lock(user_id: str) -> threading.Lock:
    with GITHUB_TOKEN_LOCKS_GUARD:
        return GITHUB_TOKEN_LOCKS.setdefault(user_id, threading.Lock())


def github_access_token(user_id: str) -> str:
    with github_user_token_lock(user_id):
        return _github_access_token_locked(user_id)


def _github_access_token_locked(user_id: str) -> str:
    path = github_token_path(user_id)
    try:
        token = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise GitHubError("Tài khoản GitHub chưa được liên kết hoặc cần liên kết lại.") from None
    access = str(token.get("access_token", ""))
    expires_at = float(token.get("expires_at", 0) or 0)
    if not access:
        raise GitHubError("Phiên GitHub không hợp lệ; hãy liên kết lại tài khoản.")
    if not expires_at or expires_at > time.time() + 60:
        return access
    refresh = str(token.get("refresh_token", ""))
    client_id = github_client_id()
    if not refresh or not client_id:
        raise GitHubError("Phiên GitHub đã hết hạn; hãy ngắt kết nối rồi liên kết lại.")
    refreshed = github_json_request("https://github.com/login/oauth/access_token", "POST", {
        "client_id": client_id, "grant_type": "refresh_token", "refresh_token": refresh,
    }, form=True)
    if not refreshed.get("access_token"):
        raise GitHubError("Không làm mới được phiên GitHub; hãy liên kết lại tài khoản.")
    token["access_token"] = refreshed["access_token"]
    token["refresh_token"] = refreshed.get("refresh_token", refresh)
    token["expires_at"] = time.time() + int(refreshed["expires_in"]) if refreshed.get("expires_in") else 0
    write_github_token(user_id, token)
    return str(token["access_token"])


def github_api(user_id: str, method: str, endpoint: str, payload: dict | None = None) -> dict:
    token = github_access_token(user_id)
    return github_json_request("https://api.github.com" + endpoint, method, payload, token)


def set_github_project_status(project_id: str, status: str, error: str = "",
                              commit_sha: str | None = None, synced_at: str | None = None) -> None:
    with db() as connection:
        connection.execute(
            "UPDATE github_projects SET sync_status=?,sync_error=?,"
            "last_commit_sha=COALESCE(?,last_commit_sha),last_synced_at=COALESCE(?,last_synced_at),updated_at=? WHERE project_id=?",
            (status, error[:500], commit_sha, synced_at, now_iso(), project_id),
        )


def _run_github_project_sync(project_id: str, user_id: str, force: bool = False) -> None:
    try:
        with db() as connection:
            mapping = connection.execute(
                "SELECT gp.*,p.name project_name FROM github_projects gp JOIN projects p ON p.id=gp.project_id "
                "WHERE gp.project_id=? AND gp.user_id=? AND p.user_id=?", (project_id, user_id, user_id)
            ).fetchone()
            files = connection.execute("SELECT path,content FROM project_files WHERE project_id=? ORDER BY path", (project_id,)).fetchall() if mapping else []
        if not mapping or (not force and not mapping["auto_sync"]):
            return
        set_github_project_status(project_id, "syncing")
        if not files:
            set_github_project_status(project_id, "complete", synced_at=now_iso())
            return
        owner, repo = mapping["owner"], mapping["repo"]
        remote = github_api(user_id, "GET", f"/repos/{quote(owner, safe='')}/{quote(repo, safe='')}")
        if remote.get("private") is not True:
            raise GitHubError("Đồng bộ bị chặn: repo GitHub không còn ở chế độ private.")
        tree_entries = [{"path": row["path"], "mode": "100644", "type": "blob", "content": row["content"]} for row in files]
        ref_url = f"/repos/{quote(owner, safe='')}/{quote(repo, safe='')}/git/ref/heads/{quote(mapping['branch'], safe='')}"
        tree_url = f"/repos/{quote(owner, safe='')}/{quote(repo, safe='')}/git/trees"
        commits_url = f"/repos/{quote(owner, safe='')}/{quote(repo, safe='')}/git/commits"
        update_url = f"/repos/{quote(owner, safe='')}/{quote(repo, safe='')}/git/refs/heads/{quote(mapping['branch'], safe='')}"
        for attempt in range(2):
            ref = github_api(user_id, "GET", ref_url)
            parent_sha = str((ref.get("object") or {}).get("sha", ""))
            if not parent_sha:
                raise GitHubError("Không đọc được nhánh GitHub của repo.")
            parent = github_api(user_id, "GET", f"/repos/{quote(owner, safe='')}/{quote(repo, safe='')}/git/commits/{quote(parent_sha, safe='')}")
            base_tree = str(((parent.get("tree") or {}).get("sha")) or "")
            tree = github_api(user_id, "POST", tree_url, {"base_tree": base_tree, "tree": tree_entries})
            tree_sha = str(tree.get("sha", ""))
            if not tree_sha:
                raise GitHubError("GitHub không tạo được cây tệp cho dự án.")
            if tree_sha == base_tree:
                set_github_project_status(project_id, "complete", commit_sha=parent_sha, synced_at=now_iso())
                return
            commit = github_api(user_id, "POST", commits_url, {
                "message": f"LocalMind: đồng bộ {mapping['project_name'][:100]}",
                "tree": tree_sha, "parents": [parent_sha],
            })
            commit_sha = str(commit.get("sha", ""))
            if not commit_sha:
                raise GitHubError("GitHub không tạo được commit.")
            try:
                github_api(user_id, "PATCH", update_url, {"sha": commit_sha, "force": False})
                set_github_project_status(project_id, "complete", commit_sha=commit_sha, synced_at=now_iso())
                return
            except GitHubError as exc:
                if attempt == 0 and exc.status in (409, 422):
                    continue
                raise
        raise GitHubError("Repo GitHub thay đổi liên tục; thử đồng bộ lại sau vài giây.")
    except Exception as exc:
        message = str(exc)[:500] or type(exc).__name__
        set_github_project_status(project_id, "error", message)


def run_github_project_sync(project_id: str, user_id: str, force: bool = False) -> None:
    with GITHUB_SYNC_LOCK:
        if project_id in GITHUB_SYNC_ACTIVE:
            pending = GITHUB_SYNC_PENDING.get(project_id)
            GITHUB_SYNC_PENDING[project_id] = (user_id, force or (pending[1] if pending else False))
            return
        GITHUB_SYNC_ACTIVE.add(project_id)
    try:
        _run_github_project_sync(project_id, user_id, force)
    finally:
        with GITHUB_SYNC_LOCK:
            GITHUB_SYNC_ACTIVE.discard(project_id)
            pending = GITHUB_SYNC_PENDING.pop(project_id, None)
        if pending:
            schedule_github_sync(project_id, pending[0], pending[1])


def schedule_github_sync(project_id: str, user_id: str, force: bool = False) -> bool:
    with db() as connection:
        mapping = connection.execute("SELECT auto_sync FROM github_projects WHERE project_id=? AND user_id=?", (project_id, user_id)).fetchone()
        if not mapping or (not force and not mapping["auto_sync"]):
            return False
    set_github_project_status(project_id, "queued", "")
    with GITHUB_SYNC_LOCK:
        previous = GITHUB_SYNC_TIMERS.pop(project_id, None)
        if previous:
            previous.cancel()
        timer = None
        def fire() -> None:
            with GITHUB_SYNC_LOCK:
                if GITHUB_SYNC_TIMERS.get(project_id) is timer:
                    GITHUB_SYNC_TIMERS.pop(project_id, None)
            run_github_project_sync(project_id, user_id, force)
        timer = threading.Timer(2.5, fire)
        timer.daemon = True
        GITHUB_SYNC_TIMERS[project_id] = timer
        timer.start()
    return True


def resume_pending_github_syncs() -> None:
    with db() as connection:
        rows = connection.execute("SELECT project_id,user_id FROM github_projects WHERE auto_sync=1 AND sync_status IN ('queued','syncing')").fetchall()
    for row in rows:
        schedule_github_sync(row["project_id"], row["user_id"])


class LocalMindHandler(BaseHTTPRequestHandler):
    server_version = "LocalMind/0.1"
    sys_version = ""

    def log_message(self, fmt: str, *args) -> None:
        # Keep routine prompt and account data out of the access log.
        print(f"[{self.log_date_time_string()}] {self.address_string()} {fmt % args}")

    def _send_json(self, status: int, payload: dict | list) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length < 0 or length > MAX_BODY:
            raise ValueError("Yêu cầu quá lớn.")
        raw = self.rfile.read(length) if length else b"{}"
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Dữ liệu gửi lên không hợp lệ.")
        return value

    def _body_or_error(self) -> dict | None:
        try:
            return self._read_json()
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc) or "JSON không hợp lệ."})
            return None

    def _session(self) -> sqlite3.Row | None:
        cookie = SimpleCookie()
        try:
            cookie.load(self.headers.get("Cookie", ""))
            raw_token = cookie.get("localmind_session").value if cookie.get("localmind_session") else ""
        except Exception:
            raw_token = ""
        if not raw_token:
            return None
        token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
        with db() as connection:
            return connection.execute(
                "SELECT u.* FROM sessions s JOIN users u ON u.id=s.user_id "
                "WHERE s.token_hash=? AND s.expires_at>? AND u.active=1",
                (token_hash, int(time.time())),
            ).fetchone()

    def _require_user(self) -> sqlite3.Row | None:
        user = self._session()
        if not user:
            self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "Bạn cần đăng nhập lại."})
        return user

    def _require_admin(self) -> sqlite3.Row | None:
        user = self._require_user()
        if user and user["role"] != "admin":
            self._send_json(HTTPStatus.FORBIDDEN, {"error": "Tài khoản không có quyền quản trị."})
            return None
        return user

    def _cookie(self, token: str, max_age: int = SESSION_DAYS * 86400) -> None:
        secure = "; Secure" if self.headers.get("X-Forwarded-Proto", "").lower() == "https" else ""
        self.send_header(
            "Set-Cookie",
            f"localmind_session={token}; Path=/; HttpOnly; SameSite=Strict; Max-Age={max_age}{secure}",
        )

    def _start_session(self, user_id: str) -> str:
        token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        with db() as connection:
            connection.execute("INSERT INTO sessions(token_hash,user_id,expires_at) VALUES(?,?,?)",
                               (token_hash, user_id, int(time.time()) + SESSION_DAYS * 86400))
        return token

    def _github_start_device(self, user: sqlite3.Row) -> None:
        client_id = github_client_id()
        if not client_id:
            self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "Admin chưa cấu hình GitHub App Client ID."})
            return
        try:
            response = github_json_request("https://github.com/login/device/code", "POST", {"client_id": client_id, "scope": "offline_access"}, form=True)
        except GitHubError as exc:
            self._send_json(HTTPStatus.BAD_GATEWAY, {"error": f"Không bắt đầu được đăng nhập GitHub: {exc}"})
            return
        device_code = response.get("device_code")
        user_code = response.get("user_code")
        if not device_code or not user_code:
            self._send_json(HTTPStatus.BAD_GATEWAY, {"error": "GitHub không cấp mã đăng nhập thiết bị. Kiểm tra GitHub App đã bật Device Flow chưa."})
            return
        flow_id = secrets.token_urlsafe(24)
        interval = max(5, int(response.get("interval", 5)))
        with GITHUB_DEVICE_FLOWS_LOCK:
            for old_flow_id in [key for key, value in GITHUB_DEVICE_FLOWS.items() if value["expires_at"] <= time.time()]:
                GITHUB_DEVICE_FLOWS.pop(old_flow_id, None)
            GITHUB_DEVICE_FLOWS[flow_id] = {
                "user_id": user["id"], "device_code": str(device_code), "client_id": client_id,
                "interval": interval, "next_poll": time.time() + interval,
                "expires_at": time.time() + max(60, int(response.get("expires_in", 900))),
            }
        self._send_json(HTTPStatus.OK, {
            "flow_id": flow_id, "user_code": user_code,
            "verification_uri": response.get("verification_uri", "https://github.com/login/device"),
            "interval": interval, "expires_in": int(response.get("expires_in", 900)),
        })

    def _github_poll_device(self, user: sqlite3.Row, body: dict) -> None:
        flow_id = str(body.get("flow_id", ""))
        with GITHUB_DEVICE_FLOWS_LOCK:
            flow = GITHUB_DEVICE_FLOWS.get(flow_id)
            if not flow or flow["user_id"] != user["id"]:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Phiên đăng nhập GitHub không tồn tại hoặc đã hết hạn."})
                return
            if time.time() >= flow["expires_at"]:
                GITHUB_DEVICE_FLOWS.pop(flow_id, None)
                self._send_json(HTTPStatus.GONE, {"error": "Mã GitHub đã hết hạn. Hãy bắt đầu liên kết lại."})
                return
            wait = flow["next_poll"] - time.time()
            if wait > 0:
                self._send_json(HTTPStatus.TOO_MANY_REQUESTS, {"error": "Đang chờ tới lượt kiểm tra GitHub.", "retry_after": round(wait, 1)})
                return
            flow["next_poll"] = time.time() + flow["interval"]
        try:
            token_response = github_json_request("https://github.com/login/oauth/access_token", "POST", {
                "client_id": flow["client_id"], "device_code": flow["device_code"],
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            }, form=True)
        except GitHubError as exc:
            self._send_json(HTTPStatus.BAD_GATEWAY, {"error": f"Không kiểm tra được trạng thái đăng nhập GitHub: {exc}"})
            return
        oauth_error = str(token_response.get("error", ""))
        if oauth_error == "authorization_pending":
            self._send_json(HTTPStatus.OK, {"pending": True, "interval": flow["interval"]})
            return
        if oauth_error == "slow_down":
            with GITHUB_DEVICE_FLOWS_LOCK:
                current = GITHUB_DEVICE_FLOWS.get(flow_id)
                if current:
                    current["interval"] += 5
                    current["next_poll"] = time.time() + current["interval"]
                    interval = current["interval"]
                else:
                    interval = flow["interval"] + 5
            self._send_json(HTTPStatus.OK, {"pending": True, "interval": interval})
            return
        if oauth_error:
            with GITHUB_DEVICE_FLOWS_LOCK:
                GITHUB_DEVICE_FLOWS.pop(flow_id, None)
            messages = {
                "expired_token": "Mã GitHub đã hết hạn. Hãy bắt đầu liên kết lại.",
                "access_denied": "Đăng nhập GitHub đã bị từ chối.",
                "incorrect_device_code": "Mã thiết bị GitHub không hợp lệ.",
            }
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": messages.get(oauth_error, "GitHub không hoàn tất được đăng nhập thiết bị.")})
            return
        access_token = str(token_response.get("access_token", ""))
        if not access_token:
            self._send_json(HTTPStatus.BAD_GATEWAY, {"error": "GitHub chưa trả access token hợp lệ."})
            return
        try:
            profile = github_json_request("https://api.github.com/user", access_token=access_token)
        except GitHubError as exc:
            self._send_json(HTTPStatus.BAD_GATEWAY, {"error": f"Không đọc được tài khoản GitHub: {exc}"})
            return
        login = str(profile.get("login", "")).strip()
        if not login or not re.fullmatch(r"[A-Za-z0-9-]{1,39}", login):
            self._send_json(HTTPStatus.BAD_GATEWAY, {"error": "GitHub trả về tên tài khoản không hợp lệ."})
            return
        stamp = now_iso()
        with db() as connection:
            other = connection.execute("SELECT user_id FROM github_accounts WHERE login=? COLLATE NOCASE AND user_id<>?", (login, user["id"])).fetchone()
            if other:
                with GITHUB_DEVICE_FLOWS_LOCK:
                    GITHUB_DEVICE_FLOWS.pop(flow_id, None)
                self._send_json(HTTPStatus.CONFLICT, {"error": "Tài khoản GitHub này đã được liên kết với một tài khoản LocalMind khác."})
                return
        token_record = {
            "access_token": access_token,
            "refresh_token": str(token_response.get("refresh_token", "")),
            "expires_at": time.time() + int(token_response.get("expires_in", 0) or 0) if token_response.get("expires_in") else 0,
        }
        previous_token = None
        try:
            with github_user_token_lock(user["id"]):
                try:
                    previous_token = json.loads(github_token_path(user["id"]).read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    previous_token = None
                write_github_token(user["id"], token_record)
                with db() as connection:
                    connection.execute(
                        "INSERT INTO github_accounts(user_id,login,connected_at) VALUES(?,?,?) "
                        "ON CONFLICT(user_id) DO UPDATE SET login=excluded.login,connected_at=excluded.connected_at",
                        (user["id"], login, stamp),
                    )
                    connection.execute(
                        "UPDATE github_projects SET auto_sync=0,sync_status='account_changed',sync_error='Tài khoản GitHub đã đổi; liên kết lại repo trước khi đồng bộ.' "
                        "WHERE user_id=? AND owner<>? COLLATE NOCASE", (user["id"], login),
                    )
        except (OSError, sqlite3.IntegrityError):
            try:
                with github_user_token_lock(user["id"]):
                    if previous_token:
                        write_github_token(user["id"], previous_token)
                    else:
                        github_token_path(user["id"]).unlink(missing_ok=True)
            except OSError:
                pass
            self._send_json(HTTPStatus.CONFLICT, {"error": "Không lưu được liên kết GitHub. Tài khoản này có thể đã được liên kết ở nơi khác."})
            return
        with GITHUB_DEVICE_FLOWS_LOCK:
            GITHUB_DEVICE_FLOWS.pop(flow_id, None)
        self._send_json(HTTPStatus.OK, {"pending": False, "connected": True, "login": login})

    def _github_create_project_repo(self, user: sqlite3.Row, project_id: str, body: dict) -> None:
        with db() as connection:
            project = connection.execute("SELECT id,name FROM projects WHERE id=? AND user_id=?", (project_id, user["id"])).fetchone()
            account = connection.execute("SELECT login FROM github_accounts WHERE user_id=?", (user["id"],)).fetchone()
            existing = connection.execute("SELECT repo_url FROM github_projects WHERE project_id=? AND user_id=?", (project_id, user["id"])).fetchone()
        if not project:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Không tìm thấy dự án."})
            return
        if not account:
            self._send_json(HTTPStatus.CONFLICT, {"error": "Hãy liên kết tài khoản GitHub trước."})
            return
        if existing:
            self._send_json(HTTPStatus.CONFLICT, {"error": "Dự án này đã có repo GitHub được liên kết."})
            return
        base = re.sub(r"[^a-z0-9]+", "-", project["name"].lower()).strip("-")[:48] or "project"
        repo_name = str(body.get("repo_name", f"localmind-{base}-{project_id[:6]}")).strip()
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,100}", repo_name):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Tên repo chỉ được gồm chữ, số, dấu chấm, gạch ngang và gạch dưới; tối đa 100 ký tự."})
            return
        try:
            repo = github_api(user["id"], "POST", "/user/repos", {
                "name": repo_name, "private": True, "auto_init": True,
                "description": f"LocalMind workspace: {project['name'][:120]}",
            })
        except GitHubError as exc:
            self._send_json(HTTPStatus.BAD_GATEWAY, {"error": f"Không tạo được repo riêng tư: {exc}"})
            return
        owner = str((repo.get("owner") or {}).get("login") or account["login"])
        actual_repo = str(repo.get("name") or repo_name)
        branch = str(repo.get("default_branch") or "main")
        if repo.get("private") is not True:
            self._send_json(HTTPStatus.BAD_GATEWAY, {"error": "GitHub không xác nhận repo được tạo ở chế độ private; LocalMind không bật đồng bộ."})
            return
        repo_url = str(repo.get("html_url") or f"https://github.com/{owner}/{actual_repo}")
        if not repo_url.startswith("https://github.com/"):
            repo_url = f"https://github.com/{owner}/{actual_repo}"
        stamp = now_iso()
        try:
            with db() as connection:
                connection.execute(
                    "INSERT INTO github_projects(project_id,user_id,owner,repo,branch,repo_url,auto_sync,sync_status,updated_at) "
                    "VALUES(?,?,?,?,?,?,1,'queued',?)",
                    (project_id, user["id"], owner, actual_repo, branch, repo_url, stamp),
                )
        except sqlite3.IntegrityError:
            self._send_json(HTTPStatus.CONFLICT, {"error": "Repo đã được gắn với dự án khác; kiểm tra GitHub trước khi tạo repo mới."})
            return
        schedule_github_sync(project_id, user["id"], force=True)
        self._send_json(HTTPStatus.CREATED, {"repository": {"owner": owner, "repo": actual_repo, "branch": branch, "repo_url": repo_url, "auto_sync": True, "sync_status": "queued"}})

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        path = unquote(parsed.path)
        if path == "/api/status":
            current_user = self._require_user()
            if not current_user:
                return
            code, health = read_upstream_json(base_url() + "/health", timeout=3)
            models_code, models = read_upstream_json(base_url() + "/v1/models", timeout=5) if code == 200 else (0, {})
            model_name = None
            if isinstance(models, dict):
                data = models.get("data") or models.get("models") or []
                if data and isinstance(data[0], dict):
                    model_name = data[0].get("id") or data[0].get("name")
            with db() as connection:
                display_row = connection.execute("SELECT value FROM settings WHERE key='ai_display_name'").fetchone()
            self._send_json(HTTPStatus.OK, {
                "online": code == 200,
                "health": health.get("status") if isinstance(health, dict) else None,
                "model": model_name,
                "display_name": display_row["value"] if display_row and display_row["value"] else None,
                "error": health.get("error") if isinstance(health, dict) and code != 200 else None,
                "models_ok": models_code == 200,
            })
            return
        if path == "/api/session":
            with db() as connection:
                users_exist = connection.execute("SELECT 1 FROM users LIMIT 1").fetchone() is not None
                settings = dict(connection.execute("SELECT key,value FROM settings WHERE key IN ('app_name','accent_color')").fetchall())
            user = self._session()
            self._send_json(HTTPStatus.OK, {
                "setup_required": not users_exist,
                "setup_token_required": not users_exist,
                "authenticated": bool(user),
                "user": user_public(user) if user else None,
                "branding": {"app_name": settings.get("app_name", "LocalMind"), "accent_color": settings.get("accent_color", "#b3a0ff")},
            })
            return
        if path == "/api/ping":
            if not self._require_user():
                return
            self._send_json(HTTPStatus.OK, {"ok": True})
            return
        if path.startswith("/api/"):
            user = self._require_user()
            if not user:
                return
            if path == "/api/github/status":
                with db() as connection:
                    account = connection.execute("SELECT login,connected_at FROM github_accounts WHERE user_id=?", (user["id"],)).fetchone()
                    settings = connection.execute("SELECT value FROM settings WHERE key='github_app_client_id'").fetchone()
                self._send_json(HTTPStatus.OK, {"configured": bool(settings and settings["value"].strip()), "connected": bool(account), "account": dict(account) if account else None})
                return
            if path == "/api/projects":
                with db() as connection:
                    rows = connection.execute("SELECT id,name,created_at,updated_at FROM projects WHERE user_id=? ORDER BY updated_at DESC", (user["id"],)).fetchall()
                self._send_json(HTTPStatus.OK, {"projects":[dict(row) for row in rows]})
                return
            project_match = re.fullmatch(r"/api/projects/([a-f0-9]{32})", path)
            if project_match:
                with db() as connection:
                    project = connection.execute("SELECT id,name,created_at,updated_at FROM projects WHERE id=? AND user_id=?", (project_match.group(1),user["id"])).fetchone()
                    if not project:
                        self._send_json(HTTPStatus.NOT_FOUND, {"error":"Không tìm thấy dự án."})
                        return
                    files = connection.execute("SELECT path,length(CAST(content AS BLOB)) size,updated_at FROM project_files WHERE project_id=? ORDER BY path", (project["id"],)).fetchall()
                self._send_json(HTTPStatus.OK, {"project":dict(project),"files":[dict(row) for row in files]})
                return
            project_github_match = re.fullmatch(r"/api/projects/([a-f0-9]{32})/github", path)
            if project_github_match:
                project_id = project_github_match.group(1)
                with db() as connection:
                    project = connection.execute("SELECT id FROM projects WHERE id=? AND user_id=?", (project_id, user["id"])).fetchone()
                    mapping = connection.execute("SELECT owner,repo,branch,repo_url,auto_sync,last_commit_sha,last_synced_at,sync_status,sync_error,updated_at FROM github_projects WHERE project_id=? AND user_id=?", (project_id, user["id"])).fetchone() if project else None
                if not project:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "Không tìm thấy dự án."})
                    return
                self._send_json(HTTPStatus.OK, {"repository": dict(mapping) if mapping else None})
                return
            project_contents_match = re.fullmatch(r"/api/projects/([a-f0-9]{32})/contents", path)
            if project_contents_match:
                with db() as connection:
                    owned=connection.execute("SELECT id FROM projects WHERE id=? AND user_id=?",(project_contents_match.group(1),user["id"])).fetchone()
                    rows=connection.execute("SELECT path,content FROM project_files WHERE project_id=? ORDER BY path",(project_contents_match.group(1),)).fetchall() if owned else []
                if not owned:
                    self._send_json(HTTPStatus.NOT_FOUND,{"error":"Không tìm thấy dự án."}); return
                self._send_json(HTTPStatus.OK,{"files":[dict(row) for row in rows]}); return
            versions_match = re.fullmatch(r"/api/projects/([a-f0-9]{32})/versions", path)
            if versions_match:
                project_id = versions_match.group(1)
                try:
                    file_path = safe_project_path((parse_qs(parsed.query).get("path") or [""])[0])
                except ValueError as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST,{"error":str(exc)}); return
                with db() as connection:
                    owned = connection.execute("SELECT 1 FROM projects WHERE id=? AND user_id=?", (project_id,user["id"])).fetchone()
                    if not owned:
                        self._send_json(HTTPStatus.NOT_FOUND,{"error":"Không tìm thấy dự án."}); return
                    rows=connection.execute("SELECT id,created_at,source FROM project_versions WHERE project_id=? AND path=? ORDER BY id DESC LIMIT 30",(project_id,file_path)).fetchall()
                self._send_json(HTTPStatus.OK,{"versions":[dict(row) for row in rows]}); return
            file_match = re.fullmatch(r"/api/projects/([a-f0-9]{32})/file", path)
            if file_match:
                project_id=file_match.group(1)
                try:
                    file_path=safe_project_path((parse_qs(parsed.query).get("path") or [""])[0])
                except ValueError as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST,{"error":str(exc)}); return
                with db() as connection:
                    row=connection.execute("SELECT f.content FROM project_files f JOIN projects p ON p.id=f.project_id WHERE p.id=? AND p.user_id=? AND f.path=?",(project_id,user["id"],file_path)).fetchone()
                if not row:
                    self._send_json(HTTPStatus.NOT_FOUND,{"error":"Không tìm thấy tệp."}); return
                self._send_json(HTTPStatus.OK,{"path":file_path,"content":row["content"]}); return
            download_match = re.fullmatch(r"/api/projects/([a-f0-9]{32})/download", path)
            if download_match:
                project_id=download_match.group(1)
                with db() as connection:
                    project=connection.execute("SELECT name FROM projects WHERE id=? AND user_id=?",(project_id,user["id"])).fetchone()
                    files=connection.execute("SELECT path,content FROM project_files WHERE project_id=? ORDER BY path",(project_id,)).fetchall() if project else []
                if not project:
                    self._send_json(HTTPStatus.NOT_FOUND,{"error":"Không tìm thấy dự án."}); return
                archive=io.BytesIO()
                with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as bundle:
                    for file_row in files:
                        bundle.writestr(file_row["path"],file_row["content"])
                safe_name=re.sub(r"[^A-Za-z0-9._-]+","-",project["name"]).strip("-.") or "localmind-project"
                payload=archive.getvalue()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type","application/zip")
                self.send_header("Content-Length",str(len(payload)))
                self.send_header("Content-Disposition",f'attachment; filename="{safe_name}.zip"')
                self.send_header("Cache-Control","no-store")
                self.end_headers(); self.wfile.write(payload); return
            if path == "/api/conversations":
                query = (parse_qs(parsed.query).get("q") or [""])[0].strip()
                with db() as connection:
                    if query:
                        rows = connection.execute(
                            "SELECT DISTINCT c.* FROM conversations c LEFT JOIN messages m ON m.conversation_id=c.id "
                            "WHERE c.user_id=? AND (c.title LIKE ? OR m.content LIKE ?) "
                            "ORDER BY c.updated_at DESC LIMIT 100",
                            (user["id"], f"%{query}%", f"%{query}%"),
                        ).fetchall()
                    else:
                        rows = connection.execute(
                            "SELECT * FROM conversations WHERE user_id=? ORDER BY updated_at DESC LIMIT 100",
                            (user["id"],),
                        ).fetchall()
                self._send_json(HTTPStatus.OK, {"conversations": [dict(row) for row in rows]})
                return
            if path == "/api/memory":
                with db() as connection:
                    row = connection.execute("SELECT content,updated_at FROM user_memories WHERE user_id=?", (user["id"],)).fetchone()
                self._send_json(HTTPStatus.OK, {"memories": json.loads(row["content"]) if row else [], "updated_at": row["updated_at"] if row else None})
                return
            sync_match = re.fullmatch(r"/api/conversations/([a-f0-9]{32})/updates", path)
            if sync_match:
                conversation = self._owned_conversation(sync_match.group(1), user["id"])
                if not conversation:
                    return
                query = parse_qs(parsed.query)
                try:
                    after = min(2**63 - 1, max(0, int((query.get("after") or ["0"])[0])))
                    active_offset = min(2**63 - 1, max(0, int((query.get("active_offset") or ["0"])[0])))
                except (TypeError, ValueError):
                    after, active_offset = 0, 0
                known_active_id = (query.get("active_id") or [""])[0]
                known_assistant_id = (query.get("assistant_id") or [""])[0]
                known_assistant_status = (query.get("assistant_status") or [""])[0]
                with db() as connection:
                    rows = connection.execute(
                        "SELECT rowid AS seq,id,role,content,attachments,mode,status,feedback,created_at FROM messages "
                        "WHERE conversation_id=? AND rowid>? ORDER BY rowid",
                        (conversation["id"], after),
                    ).fetchall()
                    cursor = connection.execute(
                        "SELECT COALESCE(MAX(rowid),0) AS seq FROM messages WHERE conversation_id=?",
                        (conversation["id"],),
                    ).fetchone()["seq"]
                    latest_row = connection.execute(
                        "SELECT rowid AS seq,id,role,content,attachments,mode,status,feedback,created_at FROM messages "
                        "WHERE conversation_id=? AND role='assistant' ORDER BY rowid DESC LIMIT 1",
                        (conversation["id"],),
                    ).fetchone()

                def message_payload(row):
                    if not row:
                        return None
                    item = dict(row)
                    item["attachments"] = json.loads(item["attachments"] or "[]")
                    return item

                active = None
                with ACTIVE_GENERATIONS_LOCK:
                    state = ACTIVE_GENERATIONS.get(conversation["id"])
                    if state:
                        text = state.get("text", "")
                        same_stream = known_active_id == state["assistant_id"]
                        offset = min(active_offset if same_stream else 0, len(text))
                        active = {
                            "message_id": state["assistant_id"],
                            "user_message_id": state.get("user_message_id"),
                            "replace_message_id": state.get("replace_message_id"),
                            "mode": state.get("mode", "medium"),
                            "text_delta": text[offset:],
                            "text_length": len(text),
                        }
                latest = message_payload(latest_row)
                if latest and latest["id"] == known_assistant_id and latest["status"] == known_assistant_status:
                    latest = None
                self._send_json(HTTPStatus.OK, {
                    "cursor": cursor,
                    "messages": [message_payload(row) for row in rows],
                    "active": active,
                    "latest": latest,
                })
                return
            context_match = re.fullmatch(r"/api/conversations/([a-f0-9]{32})/context", path)
            if context_match:
                conversation = self._owned_conversation(context_match.group(1), user["id"])
                if not conversation:
                    return
                with db() as connection:
                    history_rows = connection.execute(
                        "SELECT role,content,attachments FROM messages WHERE conversation_id=? ORDER BY rowid",
                        (conversation["id"],),
                    ).fetchall()
                    memory_row = connection.execute("SELECT content FROM user_memories WHERE user_id=?", (user["id"],)).fetchone()
                    global_settings = dict(connection.execute("SELECT key,value FROM settings WHERE key IN ('ai_language','ai_system_prompt')").fetchall())
                    tool_rows = connection.execute(
                        "SELECT t.name,t.description,t.input_schema,s.name server_name FROM tool_registry t "
                        "JOIN mcp_servers s ON s.id=t.server WHERE t.mcp_tool_name IS NOT NULL ORDER BY t.name COLLATE NOCASE"
                    ).fetchall() if get_tools_enabled() else []
                api_base = base_url()
                _, model_payload = read_upstream_json(api_base + "/v1/models", timeout=8)
                models = (model_payload.get("data") or model_payload.get("models") or []) if isinstance(model_payload, dict) else []
                model_id = (models[0].get("id") or models[0].get("name")) if models and isinstance(models[0], dict) else "local-model"
                tools_for_model = [{"type": "function", "function": {"name": row["name"], "description": row["description"], "parameters": json.loads(row["input_schema"] or "{}")}} for row in tool_rows]
                messages = build_context_messages(history_rows, json.loads(memory_row["content"]) if memory_row else [], global_settings)
                if tools_for_model:
                    has_display = any(str(row["server_name"]).casefold() == "display" for row in tool_rows)
                    desktop_rule = " Với thao tác GUI hiển thị, chỉ dùng tool của MCP server DISPLAY; không dùng CMD/shell để mở GUI vì tiến trình shell có thể không gắn vào desktop đang được quan sát. Chụp màn hình trước, thao tác bằng click/keyboard, rồi chụp lại để xác nhận." if has_display else ""
                    messages.insert(0, {"role": "system", "content": "LocalMind có thể chuyển tool calls qua MCP ở mạng nội bộ. Nếu user yêu cầu thao tác thật, hãy gọi tool phù hợp thay vì chỉ kể kế hoạch. Chỉ nói đã làm xong khi tool call thực sự được chạy và kết quả trả về xác nhận; nếu tool không được gọi, bị chặn hoặc lỗi thì nói rõ thao tác chưa hoàn tất. Khi tool trả ảnh, hãy quan sát ảnh đó trước khi kết luận." + desktop_rule})
                used = count_model_input_tokens(api_base, model_id, messages, tools_for_model)
                limit = model_context_limit(api_base)
                self._send_json(HTTPStatus.OK, {"used": used, "limit": limit, "remaining": max(0, limit - used) if limit and used is not None else None})
                return
            match = re.fullmatch(r"/api/conversations/([a-f0-9]{32})", path)
            if match:
                conversation = self._owned_conversation(match.group(1), user["id"])
                if not conversation:
                    return
                with db() as connection:
                    messages = connection.execute(
                        "SELECT rowid AS seq,id,role,content,attachments,mode,status,feedback,created_at FROM messages "
                        "WHERE conversation_id=? ORDER BY rowid", (conversation["id"],)
                    ).fetchall()
                    cursor = connection.execute(
                        "SELECT COALESCE(MAX(rowid),0) AS seq FROM messages WHERE conversation_id=?",
                        (conversation["id"],),
                    ).fetchone()["seq"]
                result = []
                for row in messages:
                    item = dict(row)
                    item["attachments"] = json.loads(item["attachments"] or "[]")
                    result.append(item)
                self._send_json(HTTPStatus.OK, {"conversation": dict(conversation), "messages": result, "cursor": cursor})
                return
            if path == "/api/admin/users" or path.startswith("/api/admin/"):
                if not self._require_admin():
                    return
                self._admin_get(path, parsed)
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Không tìm thấy API."})
            return
        self._serve_static(path)

    def _serve_static(self, path: str) -> None:
        if path == "/":
            path = "/index.html"
        target = (ROOT / path.lstrip("/")).resolve()
        try:
            target.relative_to(ROOT)
        except ValueError:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Không tìm thấy trang."})
            return
        if not target.is_file() or target.name == "server.py" or "data" in target.parts:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Không tìm thấy trang."})
            return
        content_type, _ = mimetypes.guess_type(str(target))
        body = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", (content_type or "application/octet-stream") + ("; charset=utf-8" if content_type and content_type.startswith(("text/", "application/javascript")) else ""))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.end_headers()
        self.wfile.write(body)

    def _owned_conversation(self, conversation_id: str, user_id: str) -> sqlite3.Row | None:
        with db() as connection:
            row = connection.execute("SELECT * FROM conversations WHERE id=? AND user_id=?", (conversation_id, user_id)).fetchone()
        if not row:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Không tìm thấy cuộc trò chuyện."})
        return row

    def _discover_tools(self, server: dict, result: dict) -> list[dict]:
        discovered = result.get("tools") or []
        if not isinstance(discovered, list) or len(discovered) > 500:
            raise ValueError("Danh sách tool MCP không hợp lệ hoặc quá lớn.")
        output = []
        active_names = []
        with db() as connection:
            for tool in discovered:
                if not isinstance(tool, dict):
                    continue
                remote_name = str(tool.get("name", "")).strip()
                if not remote_name or len(remote_name) > 128 or any(ord(ch) < 32 for ch in remote_name):
                    continue
                safe_remote = re.sub(r"[^A-Za-z0-9_.-]", "_", remote_name)[:42].strip("_.-") or "tool"
                public_name = f"mcp_{server['id'][:8]}_{safe_remote}"[:64]
                active_names.append(remote_name)
                description = str(tool.get("description") or f"MCP tool {remote_name}")[:1000]
                schema = tool.get("inputSchema") or {"type": "object", "properties": {}}
                if not isinstance(schema, dict):
                    schema = {"type": "object", "properties": {}}
                stamp = now_iso()
                connection.execute(
                    "INSERT INTO tool_registry(id,name,server,description,input_schema,enabled,created_at,updated_at,mcp_tool_name) VALUES(?,?,?,?,?,0,?,?,?) "
                    "ON CONFLICT(name) DO UPDATE SET server=excluded.server,description=excluded.description,input_schema=excluded.input_schema,mcp_tool_name=excluded.mcp_tool_name,updated_at=excluded.updated_at",
                    (uuid.uuid4().hex, public_name, server["id"], description, json.dumps(schema, ensure_ascii=False), stamp, stamp, remote_name),
                )
                output.append({"name": public_name, "remote_name": remote_name, "description": description, "input_schema": schema})
            if active_names:
                placeholders = ",".join("?" for _ in active_names)
                connection.execute(f"DELETE FROM tool_registry WHERE server=? AND mcp_tool_name IS NOT NULL AND mcp_tool_name NOT IN ({placeholders})", (server["id"], *active_names))
            else:
                connection.execute("DELETE FROM tool_registry WHERE server=? AND mcp_tool_name IS NOT NULL", (server["id"],))
        return output

    def _connect_mcp_server(self, body: dict) -> None:
        name = str(body.get("name", "")).strip()[:80]
        try:
            url = validate_mcp_url(body.get("url", ""))
            if not name:
                name = urlsplit(url).hostname or "MCP server"
            server_id = uuid.uuid4().hex
            with db() as connection:
                existing = connection.execute("SELECT * FROM mcp_servers WHERE url=?", (url,)).fetchone()
                if existing:
                    server_id = existing["id"]
            server = {"id": server_id, "name": name, "url": url, "session_id": None, "protocol_version": MCP_PROTOCOL_VERSION}
            server_info = None
            try:
                listing = mcp_list_all_tools(server)
            except Exception as modern_error:
                # Compatibility with Streamable HTTP servers using the 2025-11-25 initialize/session handshake.
                try:
                    initialize, session, version = mcp_initialize(url)
                    server["session_id"] = session or None
                    server["protocol_version"] = version
                    server_info = initialize.get("serverInfo")
                    listing = mcp_list_all_tools(server)
                except Exception:
                    raise modern_error
            version = server["protocol_version"]
            session = server.get("session_id")
            with db() as connection:
                connection.execute(
                    "INSERT INTO mcp_servers(id,name,url,session_id,protocol_version,created_at,updated_at) VALUES(?,?,?,?,?,?,?) "
                    "ON CONFLICT(id) DO UPDATE SET name=excluded.name,session_id=excluded.session_id,protocol_version=excluded.protocol_version,updated_at=excluded.updated_at",
                    (server_id, name, url, session or None, version, now_iso(), now_iso()),
                )
            if server.get("session_id"):
                with db() as connection:
                    connection.execute("UPDATE mcp_servers SET session_id=? WHERE id=?", (server["session_id"], server_id))
            tools = self._discover_tools(server, listing)
            self._send_json(HTTPStatus.OK, {"ok": True, "server": {"id": server_id, "name": name, "url": url, "protocol_version": version}, "tools": tools, "server_info": server_info})
        except Exception as exc:
            self._send_json(HTTPStatus.BAD_GATEWAY, {"error": f"Không kết nối/dò được MCP: {str(exc)[:500]}"})

    def _refresh_mcp_server(self, server_id: str) -> None:
        with db() as connection:
            row = connection.execute("SELECT * FROM mcp_servers WHERE id=?", (server_id,)).fetchone()
        if not row:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Không tìm thấy MCP server."})
            return
        try:
            server = dict(row)
            listing = mcp_list_all_tools(server)
            if server.get("session_id"):
                with db() as connection:
                    connection.execute("UPDATE mcp_servers SET session_id=?,updated_at=? WHERE id=?", (server["session_id"], now_iso(), server_id))
            tools = self._discover_tools(server, listing)
            self._send_json(HTTPStatus.OK, {"ok": True, "tools": tools})
        except Exception as exc:
            self._send_json(HTTPStatus.BAD_GATEWAY, {"error": f"Không làm mới được danh sách MCP: {str(exc)[:500]}"})

    def _admin_get(self, path: str, parsed) -> None:
        with db() as connection:
            if path == "/api/admin/users":
                rows = connection.execute("SELECT * FROM users ORDER BY created_at DESC").fetchall()
                self._send_json(HTTPStatus.OK, {"users": [user_public(row) for row in rows]})
                return
            if path == "/api/admin/memory":
                rows = connection.execute(
                    "SELECT u.id user_id,u.username,u.display_name,COALESCE(m.content,'[]') content,m.updated_at "
                    "FROM users u LEFT JOIN user_memories m ON m.user_id=u.id ORDER BY u.username COLLATE NOCASE"
                ).fetchall()
                accounts = []
                for row in rows:
                    item = dict(row)
                    item["memories"] = json.loads(item.pop("content"))
                    accounts.append(item)
                self._send_json(HTTPStatus.OK, {"accounts": accounts})
                return
            if path == "/api/admin/settings":
                settings = dict(connection.execute("SELECT key,value FROM settings WHERE key IN ('llama_base_url','app_name','accent_color','ai_display_name','ai_language','ai_system_prompt','ai_tools_enabled','github_app_client_id')").fetchall())
                counts = connection.execute("SELECT (SELECT COUNT(*) FROM users) users, (SELECT COUNT(*) FROM conversations) conversations, (SELECT COUNT(*) FROM messages) messages").fetchone()
                _, models = read_upstream_json(base_url() + "/v1/models", timeout=5)
                model_id = None
                if isinstance(models, dict):
                    listed = models.get("data") or models.get("models") or []
                    if listed and isinstance(listed[0], dict):
                        model_id = listed[0].get("id") or listed[0].get("name")
                self._send_json(HTTPStatus.OK, {
                    "llama_base_url": settings.get("llama_base_url", DEFAULT_LLM_BASE),
                    "app_name": settings.get("app_name", "LocalMind"),
                    "accent_color": settings.get("accent_color", "#b3a0ff"),
                    "github_app_client_id": settings.get("github_app_client_id", ""),
                    "model_id": model_id,
                    "ai_display_name": settings.get("ai_display_name", "Gemma 4 26B"),
                    "ai_language": settings.get("ai_language", "auto"),
                    "ai_system_prompt": settings.get("ai_system_prompt", ""),
                    "ai_tools_enabled": settings.get("ai_tools_enabled", "0") == "1",
                    "stats": dict(counts),
                })
                return
            if path == "/api/admin/tools":
                rows = connection.execute("SELECT t.*,s.name server_name FROM tool_registry t LEFT JOIN mcp_servers s ON s.id=t.server WHERE t.mcp_tool_name IS NOT NULL ORDER BY s.name COLLATE NOCASE,t.name COLLATE NOCASE").fetchall()
                servers = connection.execute("SELECT id,name,url,protocol_version,updated_at FROM mcp_servers ORDER BY name COLLATE NOCASE").fetchall()
                activity = connection.execute("SELECT username,tool_name,server_name,status,duration_ms,error_kind,created_at FROM tool_activity ORDER BY created_at DESC LIMIT 40").fetchall()
                self._send_json(HTTPStatus.OK, {"tools": [dict(row) for row in rows], "servers": [dict(row) for row in servers], "tools_enabled": get_tools_enabled(), "activity": [dict(row) for row in activity]})
                return
            if path == "/api/admin/mcp-servers":
                servers = connection.execute("SELECT id,name,url,protocol_version,updated_at FROM mcp_servers ORDER BY name COLLATE NOCASE").fetchall()
                self._send_json(HTTPStatus.OK, {"servers": [dict(row) for row in servers]})
                return
            if path == "/api/admin/conversations":
                query = (parse_qs(parsed.query).get("q") or [""])[0].strip()
                if query:
                    rows = connection.execute(
                        "SELECT c.*,u.username,u.display_name FROM conversations c JOIN users u ON u.id=c.user_id "
                        "WHERE c.title LIKE ? OR u.username LIKE ? ORDER BY c.updated_at DESC LIMIT 200",
                        (f"%{query}%", f"%{query}%"),
                    ).fetchall()
                else:
                    rows = connection.execute(
                        "SELECT c.*,u.username,u.display_name FROM conversations c JOIN users u ON u.id=c.user_id "
                        "ORDER BY c.updated_at DESC LIMIT 200"
                    ).fetchall()
                self._send_json(HTTPStatus.OK, {"conversations": [dict(row) for row in rows]})
                return
            match = re.fullmatch(r"/api/admin/conversations/([a-f0-9]{32})", path)
            if match:
                conversation = connection.execute(
                    "SELECT c.*,u.username,u.display_name FROM conversations c JOIN users u ON u.id=c.user_id WHERE c.id=?",
                    (match.group(1),),
                ).fetchone()
                if not conversation:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "Không tìm thấy cuộc trò chuyện."})
                    return
                messages = connection.execute(
                    "SELECT id,role,content,attachments,mode,status,feedback,created_at FROM messages WHERE conversation_id=? ORDER BY rowid",
                    (conversation["id"],),
                ).fetchall()
                result = [dict(row) for row in messages]
                for item in result:
                    item["attachments"] = json.loads(item["attachments"] or "[]")
                self._send_json(HTTPStatus.OK, {"conversation": dict(conversation), "messages": result})
                return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Không tìm thấy API quản trị."})

    def do_POST(self) -> None:
        path = unquote(urlsplit(self.path).path)
        body = self._body_or_error()
        if body is None:
            return
        if path == "/api/setup/admin":
            self._setup_admin(body)
            return
        if path == "/api/login":
            self._login(body)
            return
        if path == "/api/logout":
            self._logout()
            return
        if path == "/api/github/device/start":
            user = self._require_user()
            if user:
                self._github_start_device(user)
            return
        if path == "/api/github/device/poll":
            user = self._require_user()
            if user:
                self._github_poll_device(user, body)
            return
        github_project_match = re.fullmatch(r"/api/projects/([a-f0-9]{32})/github", path)
        if github_project_match:
            user = self._require_user()
            if user:
                self._github_create_project_repo(user, github_project_match.group(1), body)
            return
        github_sync_match = re.fullmatch(r"/api/projects/([a-f0-9]{32})/github/sync", path)
        if github_sync_match:
            user = self._require_user()
            if not user:
                return
            queued = schedule_github_sync(github_sync_match.group(1), user["id"], force=True)
            if not queued:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Dự án chưa liên kết repo GitHub."})
                return
            self._send_json(HTTPStatus.ACCEPTED, {"queued": True})
            return
        if path == "/api/projects":
            user=self._require_user()
            if not user: return
            name=str(body.get("name","Dự án mới")).strip()[:80] or "Dự án mới"
            project_id=uuid.uuid4().hex; stamp=now_iso()
            with db() as connection:
                connection.execute("INSERT INTO projects(id,user_id,name,created_at,updated_at) VALUES(?,?,?,?,?)",(project_id,user["id"],name,stamp,stamp))
            self._send_json(HTTPStatus.CREATED,{"project":{"id":project_id,"name":name,"created_at":stamp,"updated_at":stamp}})
            return
        restore_match=re.fullmatch(r"/api/projects/([a-f0-9]{32})/restore",path)
        if restore_match:
            user=self._require_user()
            if not user: return
            project_id=restore_match.group(1)
            try:
                file_path=safe_project_path(body.get("path",""))
                version_id=int(body.get("version_id",0))
            except (ValueError,TypeError):
                self._send_json(HTTPStatus.BAD_REQUEST,{"error":"Đường dẫn hoặc phiên bản không hợp lệ."}); return
            with db() as connection:
                owned=connection.execute("SELECT 1 FROM projects WHERE id=? AND user_id=?",(project_id,user["id"])).fetchone()
                version=connection.execute("SELECT content FROM project_versions WHERE id=? AND project_id=? AND path=?",(version_id,project_id,file_path)).fetchone() if owned else None
                if not version:
                    self._send_json(HTTPStatus.NOT_FOUND,{"error":"Không tìm thấy phiên bản của dự án."}); return
                current=connection.execute("SELECT id FROM project_files WHERE project_id=? AND path=?",(project_id,file_path)).fetchone()
                stamp=now_iso(); file_id=current["id"] if current else uuid.uuid4().hex
                connection.execute("INSERT INTO project_files(id,project_id,path,content,updated_at) VALUES(?,?,?,?,?) ON CONFLICT(project_id,path) DO UPDATE SET content=excluded.content,updated_at=excluded.updated_at",(file_id,project_id,file_path,version["content"],stamp))
                connection.execute("INSERT INTO project_versions(project_id,path,content,created_at,source) VALUES(?,?,?,?,?)",(project_id,file_path,version["content"],stamp,"restore"))
                trim_project_versions(connection,project_id,file_path)
                connection.execute("UPDATE projects SET updated_at=? WHERE id=?",(stamp,project_id))
            schedule_github_sync(project_id, user["id"])
            self._send_json(HTTPStatus.OK,{"ok":True,"path":file_path,"content":version["content"],"updated_at":stamp}); return
        if path == "/api/conversations":
            user = self._require_user()
            if not user:
                return
            conversation_id = uuid.uuid4().hex
            stamp = now_iso()
            with db() as connection:
                connection.execute("INSERT INTO conversations(id,user_id,title,created_at,updated_at) VALUES(?,?,?,?,?)",
                                   (conversation_id, user["id"], "Cuộc trò chuyện mới", stamp, stamp))
            self._send_json(HTTPStatus.CREATED, {"conversation": {"id": conversation_id, "title": "Cuộc trò chuyện mới", "created_at": stamp, "updated_at": stamp}})
            return
        match = re.fullmatch(r"/api/conversations/([a-f0-9]{32})/stop", path)
        if match:
            user = self._require_user()
            if user:
                self._stop_generation(match.group(1), user)
            return
        if path == "/api/admin/users":
            admin = self._require_admin()
            if admin:
                self._create_user(body)
            return
        if path == "/api/admin/tools":
            if self._require_admin():
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Thêm MCP bằng URL server ở tab này; tool sẽ tự được dò từ server."})
            return
        if path == "/api/admin/mcp-servers":
            if self._require_admin():
                self._connect_mcp_server(body)
            return
        match = re.fullmatch(r"/api/admin/mcp-servers/([a-f0-9]{32})/refresh", path)
        if match:
            if self._require_admin():
                self._refresh_mcp_server(match.group(1))
            return
        if path == "/api/memory":
            user = self._require_user()
            if user:
                self._save_user_memory(user["id"], body.get("memories"))
            return
        match = re.fullmatch(r"/api/conversations/([a-f0-9]{32})/messages", path)
        if match:
            user = self._require_user()
            if user:
                self._stream_chat(match.group(1), user, body, regenerate=False)
            return
        match = re.fullmatch(r"/api/conversations/([a-f0-9]{32})/regenerate", path)
        if match:
            user = self._require_user()
            if user:
                self._stream_chat(match.group(1), user, body, regenerate=True)
            return
        match = re.fullmatch(r"/api/messages/([a-f0-9]{32})/feedback", path)
        if match:
            user = self._require_user()
            if user:
                self._feedback(match.group(1), user, body)
            return
        if path == "/api/admin/conversations/delete":
            if not self._require_admin():
                return
            conversation_id = body.get("conversation_id", "")
            with db() as connection:
                cursor = connection.execute("DELETE FROM conversations WHERE id=?", (conversation_id,))
            self._send_json(HTTPStatus.OK, {"deleted": cursor.rowcount > 0})
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Không tìm thấy API."})

    def _setup_admin(self, body: dict) -> None:
        username = str(body.get("username", "")).strip()
        display_name = str(body.get("display_name", "")).strip() or username
        password = str(body.get("password", ""))
        setup_token = str(body.get("setup_token", ""))
        if not hmac.compare_digest(setup_token, SETUP_TOKEN):
            self._send_json(HTTPStatus.FORBIDDEN, {"error": "Mã thiết lập không đúng. Lấy mã từ cửa sổ đang chạy LocalMind."})
            return
        if not re.fullmatch(r"[A-Za-z0-9_.-]{3,32}", username):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Tên đăng nhập cần 3–32 ký tự: chữ, số, dấu chấm, gạch dưới hoặc gạch ngang."})
            return
        if len(password) < 12:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Mật khẩu cần ít nhất 12 ký tự."})
            return
        with SETUP_LOCK, db() as connection:
            if connection.execute("SELECT 1 FROM users LIMIT 1").fetchone():
                self._send_json(HTTPStatus.CONFLICT, {"error": "Thiết lập ban đầu đã hoàn tất."})
                return
            user_id = uuid.uuid4().hex
            connection.execute("INSERT INTO users(id,username,display_name,password_hash,role,active,created_at) VALUES(?,?,?,?,?,?,?)",
                               (user_id, username, display_name[:64], password_hash(password), "admin", 1, now_iso()))
        token = self._start_session(user_id)
        with db() as connection:
            user = connection.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
        self.send_response(HTTPStatus.CREATED)
        self._cookie(token)
        data = json.dumps({"user": user_public(user)}, ensure_ascii=False).encode()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _login(self, body: dict) -> None:
        username = str(body.get("username", "")).strip()
        password = str(body.get("password", ""))
        with db() as connection:
            user = connection.execute("SELECT * FROM users WHERE username=? COLLATE NOCASE", (username,)).fetchone()
        if not user or not user["active"] or not password_matches(password, user["password_hash"]):
            self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "Tên đăng nhập hoặc mật khẩu không đúng."})
            return
        token = self._start_session(user["id"])
        data = json.dumps({"user": user_public(user)}, ensure_ascii=False).encode()
        self.send_response(HTTPStatus.OK)
        self._cookie(token)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _logout(self) -> None:
        cookie = SimpleCookie()
        try:
            cookie.load(self.headers.get("Cookie", ""))
            raw_token = cookie.get("localmind_session").value if cookie.get("localmind_session") else ""
        except Exception:
            raw_token = ""
        if raw_token:
            token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
            with db() as connection:
                connection.execute("DELETE FROM sessions WHERE token_hash=?", (token_hash,))
        data = b'{"ok":true}'
        self.send_response(HTTPStatus.OK)
        self._cookie("", max_age=0)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _create_user(self, body: dict) -> None:
        username = str(body.get("username", "")).strip()
        display_name = str(body.get("display_name", "")).strip() or username
        password = str(body.get("password", ""))
        role = body.get("role", "user")
        if not re.fullmatch(r"[A-Za-z0-9_.-]{3,32}", username):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Tên đăng nhập không hợp lệ."})
            return
        if len(password) < 12:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Mật khẩu cần ít nhất 12 ký tự."})
            return
        if role not in ("admin", "user"):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Vai trò không hợp lệ."})
            return
        user_id = uuid.uuid4().hex
        try:
            with db() as connection:
                connection.execute("INSERT INTO users(id,username,display_name,password_hash,role,active,created_at) VALUES(?,?,?,?,?,?,?)",
                                   (user_id, username, display_name[:64], password_hash(password), role, 1, now_iso()))
                row = connection.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
        except sqlite3.IntegrityError:
            self._send_json(HTTPStatus.CONFLICT, {"error": "Tên đăng nhập đã tồn tại."})
            return
        self._send_json(HTTPStatus.CREATED, {"user": user_public(row)})

    def _feedback(self, message_id: str, user: sqlite3.Row, body: dict) -> None:
        score = body.get("score")
        if score not in (-1, 1):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Đánh giá không hợp lệ."})
            return
        with db() as connection:
            message = connection.execute(
                "SELECT m.id FROM messages m JOIN conversations c ON c.id=m.conversation_id "
                "WHERE m.id=? AND m.role='assistant' AND c.user_id=?", (message_id, user["id"])
            ).fetchone()
            if not message:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Không tìm thấy câu trả lời."})
                return
            connection.execute("UPDATE messages SET feedback=? WHERE id=?", (score, message_id))
        self._send_json(HTTPStatus.OK, {"ok": True, "score": score})

    def _stream_chat(self, conversation_id: str, user: sqlite3.Row, body: dict, regenerate: bool) -> None:
        conversation = self._owned_conversation(conversation_id, user["id"])
        if not conversation:
            return
        with ACTIVE_GENERATIONS_LOCK:
            running = ACTIVE_GENERATIONS.get(conversation_id)
            if running and not running["stop"].is_set():
                self._send_json(HTTPStatus.CONFLICT, {"error": "Cuộc trò chuyện này đang có phản hồi chạy."})
                return
        mode = str(body.get("mode", "medium"))
        if mode not in ("off", "medium", "ultra"):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Chế độ suy luận không hợp lệ."})
            return
        workspace_enabled = body.get("workspace_enabled") is True
        project_id = str(body.get("project_id", "")) if workspace_enabled else ""
        if project_id and not re.fullmatch(r"[a-f0-9]{32}", project_id):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Dự án workspace không hợp lệ."})
            return
        if project_id:
            with db() as connection:
                owned_project = connection.execute("SELECT 1 FROM projects WHERE id=? AND user_id=?", (project_id, user["id"])).fetchone()
            if not owned_project:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Dự án không thuộc tài khoản hiện tại."})
                return
        attachments = body.get("attachments", [])
        if not isinstance(attachments, list) or len(attachments) > 4:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Chỉ gửi tối đa 4 ảnh mỗi lượt."})
            return
        clean_attachments = []
        total_image_bytes = 0
        for item in attachments:
            if not isinstance(item, dict):
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Tệp đính kèm không hợp lệ."})
                return
            name = str(item.get("name", "image"))[:120]
            mime = str(item.get("type", ""))
            data = str(item.get("data", ""))
            if mime not in ("image/png", "image/jpeg", "image/webp", "image/gif"):
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"Định dạng ảnh không được hỗ trợ: {name}"})
                return
            try:
                decoded = base64.b64decode(data, validate=True)
            except Exception:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Dữ liệu ảnh không hợp lệ."})
                return
            total_image_bytes += len(decoded)
            if len(decoded) > MAX_IMAGE_BYTES or total_image_bytes > MAX_IMAGE_BYTES:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Tổng dung lượng ảnh tối đa là 10 MB mỗi lượt."})
                return
            clean_attachments.append({"name": name, "type": mime, "data": data})

        text = str(body.get("message", "")).strip()
        if not regenerate and not text and not clean_attachments:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Nhập nội dung hoặc đính kèm ảnh trước khi gửi."})
            return
        try:
            api_base = base_url()
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        stamp = now_iso()
        assistant_id = uuid.uuid4().hex
        user_message_id = None
        replace_message_id = None
        with db() as connection:
            existing = connection.execute("SELECT COUNT(*) n FROM messages WHERE conversation_id=?", (conversation_id,)).fetchone()["n"]
            if regenerate:
                last = connection.execute("SELECT id,role FROM messages WHERE conversation_id=? ORDER BY rowid DESC LIMIT 1", (conversation_id,)).fetchone()
                if not last or last["role"] != "assistant":
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Chưa có câu trả lời để tạo lại."})
                    return
                replace_message_id = last["id"]
                connection.execute("DELETE FROM messages WHERE id=?", (last["id"],))
            else:
                user_message_id = uuid.uuid4().hex
                title = conversation["title"]
                if existing == 0:
                    title = (text or "Hình ảnh đính kèm").replace("\n", " ")[:52].strip() or "Cuộc trò chuyện mới"
                    connection.execute("UPDATE conversations SET title=? WHERE id=?", (title, conversation_id))
                connection.execute("INSERT INTO messages(id,conversation_id,role,content,attachments,mode,status,created_at) VALUES(?,?,?,?,?,?,?,?)",
                                   (user_message_id, conversation_id, "user", text, json.dumps(clean_attachments, ensure_ascii=False), mode, "complete", stamp))
            history_rows = connection.execute(
                "SELECT role,content,attachments FROM messages WHERE conversation_id=? ORDER BY rowid", (conversation_id,)
            ).fetchall()
            memory_row = connection.execute("SELECT content FROM user_memories WHERE user_id=?", (user["id"],)).fetchone()
            global_settings = dict(connection.execute("SELECT key,value FROM settings WHERE key IN ('ai_language','ai_system_prompt')").fetchall())
            connection.execute("INSERT INTO messages(id,conversation_id,role,content,attachments,mode,status,created_at) VALUES(?,?,?,?,?,?,?,?)",
                               (assistant_id, conversation_id, "assistant", "", "[]", mode, "pending", stamp))
            connection.execute("UPDATE conversations SET updated_at=? WHERE id=?", (stamp, conversation_id))

        active_state = {"stop": threading.Event(), "response": None, "tool_response": None, "text": "", "assistant_id": assistant_id, "user_message_id": user_message_id, "replace_message_id": replace_message_id, "mode": mode, "conversation_id": conversation_id, "username": user["username"], "user_id": user["id"], "project_id": project_id or None}
        with ACTIVE_GENERATIONS_LOCK:
            ACTIVE_GENERATIONS[conversation_id] = active_state

        memory_items = json.loads(memory_row["content"]) if memory_row else []
        messages = build_context_messages(history_rows, memory_items, global_settings)

        full_text = ""
        thinking_started = False
        headers_sent = False
        extract_memory = False
        emit_event = None
        try:
            _, model_payload = read_upstream_json(api_base + "/v1/models", timeout=8)
            if active_state["stop"].is_set():
                self._mark_stream_cancelled(assistant_id, full_text)
                return
            model_id = "local-model"
            if isinstance(model_payload, dict):
                model_list = model_payload.get("data") or model_payload.get("models") or []
                if model_list and isinstance(model_list[0], dict):
                    model_id = model_list[0].get("id") or model_list[0].get("name") or model_id
            tools_for_model = workspace_tool_definitions(
                include_create=workspace_enabled and not project_id,
                include_github=workspace_enabled and bool(project_id),
            ) if workspace_enabled else []
            tool_rows = []
            if get_tools_enabled():
                with db() as connection:
                    tool_rows = connection.execute("SELECT t.name,t.description,t.input_schema,s.name server_name FROM tool_registry t JOIN mcp_servers s ON s.id=t.server WHERE t.mcp_tool_name IS NOT NULL ORDER BY t.name COLLATE NOCASE").fetchall()
                tools_for_model.extend({"type": "function", "function": {"name": row["name"], "description": row["description"], "parameters": json.loads(row["input_schema"] or "{}")}} for row in tool_rows)
            if tools_for_model:
                if workspace_enabled:
                    messages.insert(0, {"role": "system", "content": "LocalMind có workspace riêng cho tài khoản. Khi người dùng yêu cầu tạo hoặc sửa sản phẩm/code, hãy thao tác trên tệp workspace bằng tool thay vì chỉ dán mã vào chat. Nếu chưa có dự án, tạo dự án trước; nếu đã có, hãy liệt kê/đọc tệp liên quan rồi mới sửa. Với yêu cầu giao diện web, GUI tương tác hoặc prototype trực quan, mặc định tạo ứng dụng HTML/CSS/JavaScript trong index.html để LocalMind hiển thị preview tự động; chỉ dùng Tkinter/GUI desktop native khi người dùng yêu cầu rõ, vì preview web không chạy ứng dụng desktop. Viết từng tệp hoàn chỉnh bằng workspace_write_file; giao diện sẽ lưu phiên bản và cập nhật preview. Chỉ báo đã lưu khi tool xác nhận thành công. Workspace chỉ truy cập được bởi đúng tài khoản đang đăng nhập. Nếu có công cụ workspace_github_status và workspace_github_sync, dùng status trước khi nói về trạng thái GitHub; file được AI ghi sẽ tự đồng bộ nếu dự án đã liên kết repo và bật auto-sync. Chỉ gọi sync cho yêu cầu đồng bộ của user; không tạo/xóa repo bằng tool và không tuyên bố đã đồng bộ xong cho tới khi trạng thái xác nhận."})
                has_display = any(str(row["server_name"]).casefold() == "display" for row in tool_rows)
                desktop_rule = " Với thao tác GUI hiển thị, chỉ dùng tool của MCP server DISPLAY; không dùng CMD/shell để mở GUI vì tiến trình shell có thể không gắn vào desktop đang được quan sát. Chụp màn hình trước, thao tác bằng click/keyboard, rồi chụp lại để xác nhận." if has_display else ""
                if tool_rows:
                    messages.insert(0, {"role": "system", "content": "LocalMind có thể chuyển tool calls qua MCP ở mạng nội bộ. Nếu user yêu cầu thao tác thật, hãy gọi tool phù hợp thay vì chỉ kể kế hoạch. Chỉ nói đã làm xong khi tool call thực sự được chạy và kết quả trả về xác nhận; nếu tool không được gọi, bị chặn hoặc lỗi thì nói rõ thao tác chưa hoàn tất. Khi tool trả ảnh, phải quan sát ảnh đó trước khi kết luận; không suy đoán nội dung màn hình từ tên tool hay hệ điều hành. Ảnh MCP có thể được chuyển thành thông điệp riêng để tương thích với chat template; thông điệp bắt đầu bằng [Ảnh do tool MCP trả về:] là output của tool gần nhất, không phải yêu cầu mới của user." + desktop_rule})
            context_limit = model_context_limit(api_base)

            def emit(event: dict) -> None:
                self.wfile.write(("data: " + json.dumps(event, ensure_ascii=False) + "\n\n").encode("utf-8"))
                self.wfile.flush()

            while True:
                if active_state["stop"].is_set():
                    break
                payload = {
                    "model": model_id,
                    "messages": messages,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "chat_template_kwargs": {"enable_thinking": mode != "off"},
                    "reasoning_budget": 0 if mode == "off" else (5000 if mode == "medium" else -1),
                }
                if tools_for_model:
                    payload["tools"] = tools_for_model
                    payload["tool_choice"] = "auto"
                prompt_tokens = count_model_input_tokens(api_base, model_id, messages, payload.get("tools"))
                streamed_pieces = 0
                generation_started = time.monotonic()
                if not headers_sent:
                    # Open the first upstream stream before committing LocalMind's stream response.
                    stream = iter_model_stream(api_base, payload, active_state)
                    first_item = next(stream, None)
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache, no-transform")
                    self.send_header("X-Accel-Buffering", "no")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    headers_sent = True
                    def emit_event(event: dict) -> None:
                        emit(event)
                    emit({"type": "start", "message_id": assistant_id, "user_message_id": user_message_id, "replace_message_id": replace_message_id})
                    emit({"type": "context", "used": prompt_tokens, "limit": context_limit,
                          "remaining": max(0, context_limit - prompt_tokens) if context_limit and prompt_tokens is not None else None,
                          "speed": None, "exact": prompt_tokens is not None})
                    # Keep the upstream iterator lazy. Extending a list here drains the
                    # whole llama-server stream before forwarding any generated text.
                    stream_items = chain((first_item,), stream) if first_item is not None else stream
                else:
                    stream_items = iter_model_stream(api_base, payload, active_state)
                tool_calls = []
                for kind, value in stream_items:
                    if kind == "ping":
                        emit({"type": "ping"})
                    elif kind == "thinking":
                        thinking_started = True
                        streamed_pieces += 1
                        live_used = prompt_tokens + streamed_pieces if prompt_tokens is not None else None
                        elapsed = max(0.001, time.monotonic() - generation_started)
                        emit({"type": "reasoning_progress", "tokens": streamed_pieces,
                              "used": live_used, "limit": context_limit,
                              "remaining": max(0, context_limit - live_used) if context_limit and live_used is not None else None,
                              "speed": round(streamed_pieces / elapsed, 1), "exact": False})
                    elif kind == "token":
                        full_text += value
                        active_state["text"] = full_text
                        streamed_pieces += 1
                        live_used = prompt_tokens + streamed_pieces if prompt_tokens is not None else None
                        elapsed = max(0.001, time.monotonic() - generation_started)
                        emit({"type": "token", "text": value, "used": live_used, "limit": context_limit,
                              "remaining": max(0, context_limit - live_used) if context_limit and live_used is not None else None,
                              "speed": round(streamed_pieces / elapsed, 1), "exact": False})
                    elif kind == "usage":
                        usage = value.get("usage") or {}
                        timings = value.get("timings") or {}
                        used = None
                        try:
                            used = int(usage.get("prompt_tokens", 0)) + int(usage.get("completion_tokens", 0))
                        except (TypeError, ValueError):
                            pass
                        speed = timings.get("predicted_per_second")
                        try:
                            speed = round(float(speed), 1) if speed is not None else None
                        except (TypeError, ValueError):
                            speed = None
                        emit({"type": "context", "used": used, "limit": context_limit,
                              "remaining": max(0, context_limit - used) if context_limit and used is not None else None,
                              "speed": speed, "exact": used is not None})
                    elif kind == "complete":
                        tool_calls = value
                if active_state["stop"].is_set():
                    break
                if not tool_calls:
                    break
                messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls})
                image_handoffs = []
                for call in tool_calls:
                    function = call.get("function") or {}
                    tool_name = str(function.get("name", ""))
                    with db() as connection:
                        tool_label = connection.execute("SELECT mcp_tool_name FROM tool_registry WHERE name=?", (tool_name,)).fetchone()
                    display_tool_name = tool_label["mcp_tool_name"] if tool_label else tool_name
                    emit({"type": "tool", "id": call.get("id", ""), "name": display_tool_name,
                          "arguments": redact_tool_arguments(function.get("arguments", "{}"))})
                    if tool_name.startswith("workspace_"):
                        result_content, tool_status, project_update = execute_workspace_tool(call, active_state)
                        if project_update and project_update.get("name"):
                            emit({"type": "workspace_project", "project": project_update})
                        if project_update and project_update.get("path"):
                            emit({"type": "workspace_changed", "project_id": active_state.get("project_id"), "path": project_update["path"]})
                    else:
                        result_content, tool_status = execute_mcp_tool(call, active_state)
                    result_preview, preview_image = summarize_tool_result(result_content)
                    emit({"type": "tool_result", "id": call.get("id", ""), "name": display_tool_name,
                          "status": tool_status, "preview": result_preview, "preview_image": preview_image})
                    if isinstance(result_content, list):
                        text_parts = [part["text"] for part in result_content if part.get("type") == "text"]
                        image_parts = [part for part in result_content if part.get("type") == "image_url"]
                        tool_text = "\n".join(text_parts).strip() or ("MCP trả về ảnh; ảnh được đính kèm ở thông điệp kế tiếp." if image_parts else "Tool hoàn tất.")
                        messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": tool_text})
                        if image_parts:
                            handoff = [{"type": "text", "text": f"[Ảnh do tool MCP trả về: {tool_label['mcp_tool_name'] if tool_label else tool_name}. Hãy quan sát ảnh này như kết quả của tool gần nhất.]"}]
                            handoff.extend(image_parts)
                            image_handoffs.append({"role": "user", "content": handoff})
                    else:
                        messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": result_content})
                    if active_state["stop"].is_set():
                        break
                else:
                    messages.extend(image_handoffs)
                    continue
                break

            status = "partial" if active_state["stop"].is_set() else ("complete" if full_text else "error")
            with db() as connection:
                connection.execute("UPDATE messages SET content=?,status=? WHERE id=?", (full_text, status, assistant_id))
                connection.execute("UPDATE conversations SET updated_at=? WHERE id=?", (now_iso(), conversation_id))
            extract_memory = status == "complete" and bool(full_text)
            emit({"type": "done", "message_id": assistant_id, "status": status})
        except HTTPError as exc:
            detail = ""
            try:
                detail = exc.read(4096).decode("utf-8", "replace")
            except Exception:
                pass
            self._mark_stream_error(assistant_id, full_text)
            if headers_sent and emit_event:
                try:
                    emit_event({"type": "error", "error": detail or "Máy chủ AI từ chối yêu cầu."})
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self._send_json(HTTPStatus.BAD_GATEWAY, {"error": detail or "Máy chủ AI từ chối yêu cầu."})
        except (URLError, TimeoutError, OSError, BrokenPipeError, ConnectionResetError) as exc:
            cancelled = active_state["stop"].is_set() or isinstance(exc, (BrokenPipeError, ConnectionResetError))
            if cancelled:
                self._mark_stream_cancelled(assistant_id, full_text)
            else:
                print(f"LocalMind upstream stream interrupted: {type(exc).__name__}: {exc}")
                self._mark_stream_error(assistant_id, full_text)
                if full_text:
                    with db() as connection:
                        connection.execute("UPDATE messages SET status='partial' WHERE id=?", (assistant_id,))
            if headers_sent and emit_event and not cancelled:
                try:
                    emit_event({"type": "error", "error": "Kết nối với máy chủ AI bị gián đoạn."})
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
            elif not headers_sent:
                self._send_json(HTTPStatus.BAD_GATEWAY, {"error": "Không kết nối được máy chủ AI. Kiểm tra trạng thái máy chủ và địa chỉ API."})
        except Exception as exc:
            if active_state["stop"].is_set():
                self._mark_stream_cancelled(assistant_id, full_text)
            else:
                self._mark_stream_error(assistant_id, full_text)
            print(f"LocalMind stream failure: {type(exc).__name__}: {exc}")
            if headers_sent and emit_event and not active_state["stop"].is_set():
                try:
                    emit_event({"type": "error", "error": "Lỗi trong lúc nhận phản hồi từ AI."})
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
            else:
                if not active_state["stop"].is_set():
                    self._send_json(HTTPStatus.BAD_GATEWAY, {"error": "Không thể bắt đầu yêu cầu tới AI."})
        finally:
            with ACTIVE_GENERATIONS_LOCK:
                if ACTIVE_GENERATIONS.get(conversation_id) is active_state:
                    ACTIVE_GENERATIONS.pop(conversation_id, None)
            if extract_memory:
                memory_user_text = text or next((row["content"] for row in reversed(history_rows) if row["role"] == "user"), "")
                threading.Thread(target=self._extract_user_memory, args=(user["id"], memory_user_text, full_text), daemon=True).start()

    def _stop_generation(self, conversation_id: str, user: sqlite3.Row) -> None:
        if not self._owned_conversation(conversation_id, user["id"]):
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Không tìm thấy cuộc trò chuyện."})
            return
        with ACTIVE_GENERATIONS_LOCK:
            state = ACTIVE_GENERATIONS.get(conversation_id)
            if state:
                state["stop"].set()
                upstream = state.get("response")
                if upstream:
                    try:
                        upstream.close()
                    except OSError:
                        pass
                self._mark_stream_cancelled(state["assistant_id"], state.get("text", ""))
        self._send_json(HTTPStatus.OK, {"ok": True, "stopped": bool(state)})

    def _save_user_memory(self, user_id: str, memories) -> None:
        cleaned = clean_memory_list(memories)
        if cleaned is None:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Memory phải là danh sách tối đa 30 mục, mỗi mục tối đa 300 ký tự."})
            return
        with db() as connection:
            connection.execute(
                "INSERT INTO user_memories(user_id,content,updated_at) VALUES(?,?,?) "
                "ON CONFLICT(user_id) DO UPDATE SET content=excluded.content,updated_at=excluded.updated_at",
                (user_id, json.dumps(cleaned, ensure_ascii=False), now_iso()),
            )
        self._send_json(HTTPStatus.OK, {"memories": cleaned})

    def _create_tool(self, body: dict) -> None:
        name = str(body.get("name", "")).strip()
        server = str(body.get("server", "")).strip()[:100]
        description = str(body.get("description", "")).strip()[:1000]
        schema_text = str(body.get("input_schema", "{}")).strip() or "{}"
        if not re.fullmatch(r"[A-Za-z0-9_.-]{2,64}", name) or not server or not description:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Cần tên tool hợp lệ, tên MCP server và mô tả."})
            return
        try:
            schema = json.loads(schema_text)
            if not isinstance(schema, dict):
                raise ValueError()
        except (json.JSONDecodeError, ValueError):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Input schema phải là JSON object hợp lệ."})
            return
        stamp = now_iso()
        try:
            with db() as connection:
                connection.execute(
                    "INSERT INTO tool_registry(id,name,server,description,input_schema,enabled,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)",
                    (uuid.uuid4().hex, name, server, description, json.dumps(schema, ensure_ascii=False), 0, stamp, stamp),
                )
        except sqlite3.IntegrityError:
            self._send_json(HTTPStatus.CONFLICT, {"error": "Tên tool đã tồn tại."})
            return
        self._send_json(HTTPStatus.CREATED, {"ok": True})

    def _patch_tool(self, tool_id: str, body: dict) -> None:
        with db() as connection:
            row = connection.execute("SELECT * FROM tool_registry WHERE id=?", (tool_id,)).fetchone()
            if not row:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Không tìm thấy tool."})
                return
            name = str(body.get("name", row["name"])).strip()
            server = str(body.get("server", row["server"])).strip()[:100]
            description = str(body.get("description", row["description"])).strip()[:1000]
            schema_text = body.get("input_schema", row["input_schema"])
            if not re.fullmatch(r"[A-Za-z0-9_.-]{2,64}", name) or not server or not description:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Tên, MCP server hoặc mô tả tool không hợp lệ."})
                return
            try:
                schema = json.loads(schema_text) if isinstance(schema_text, str) else schema_text
                if not isinstance(schema, dict):
                    raise ValueError()
            except (json.JSONDecodeError, ValueError, TypeError):
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Input schema phải là JSON object hợp lệ."})
                return
            try:
                connection.execute(
                    "UPDATE tool_registry SET name=?,server=?,description=?,input_schema=?,updated_at=? WHERE id=?",
                    (name, server, description, json.dumps(schema, ensure_ascii=False), now_iso(), tool_id),
                )
            except sqlite3.IntegrityError:
                self._send_json(HTTPStatus.CONFLICT, {"error": "Tên tool đã tồn tại."})
                return
        self._send_json(HTTPStatus.OK, {"ok": True})

    def _extract_user_memory(self, user_id: str, user_text: str, assistant_text: str) -> None:
        """Use the local model to extract durable facts after a completed turn."""
        try:
            with db() as connection:
                existing_row = connection.execute("SELECT content FROM user_memories WHERE user_id=?", (user_id,)).fetchone()
                user_row = connection.execute("SELECT active FROM users WHERE id=?", (user_id,)).fetchone()
            if not user_row or not user_row["active"]:
                return
            existing = json.loads(existing_row["content"]) if existing_row else []
            explicit = bool(re.search(r"(?:nhớ|ghi nhớ|lưu lại|đừng quên|remember|save this|keep in mind)", user_text, re.I))
            base = base_url()
            _, model_payload = read_upstream_json(base + "/v1/models", timeout=8)
            models = model_payload.get("data") or model_payload.get("models") or [] if isinstance(model_payload, dict) else []
            model_id = (models[0].get("id") or models[0].get("name")) if models and isinstance(models[0], dict) else "local-model"
            prompt = {
                "existing_memories": existing,
                "latest_user_message": user_text[-5000:],
                "latest_assistant_answer": assistant_text[-3000:],
                "explicit_remember_request": explicit,
            }
            payload = {
                "model": model_id,
                "stream": False,
                "temperature": 0.1,
                "max_tokens": 700,
                "messages": [
                    {"role": "system", "content": "Bạn quản lý memory dài hạn cho MỘT tài khoản LocalMind. Cập nhật danh sách thông tin ổn định, hữu ích cho hội thoại tương lai: sở thích, mục tiêu, dự án, bối cảnh cá nhân/công việc, cách trả lời mong muốn. Chỉ lưu điều user trực tiếp nói; không coi câu trả lời của assistant là bằng chứng. Ưu tiên tuyệt đối yêu cầu rõ ràng kiểu 'hãy nhớ/ghi nhớ/lưu lại'. Không ghi nội dung nhạy cảm như mật khẩu, khóa bí mật, mã xác thực, dữ liệu tài chính/y tế riêng tư; không suy đoán. Loại bỏ thông tin lỗi thời/mâu thuẫn khi có bằng chứng mới, gộp mục trùng. Nếu không có gì đáng lưu, giữ nguyên. Chỉ trả JSON array gồm tối đa 30 chuỗi ngắn, mỗi chuỗi tối đa 300 ký tự; không markdown."},
                    {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
                ],
            }
            request = Request(base + "/v1/chat/completions", data=json.dumps(payload, ensure_ascii=False).encode(), headers={"Content-Type": "application/json"}, method="POST")
            with urlopen(request, timeout=90) as response:
                result = json.loads(response.read(1024 * 1024).decode("utf-8", "replace"))
            text = result["choices"][0]["message"].get("content", "")
            match = re.search(r"\[[\s\S]*\]", text)
            if not match:
                return
            parsed = clean_memory_list(json.loads(match.group(0)))
            if parsed is None:
                return
            with db() as connection:
                current = connection.execute("SELECT active FROM users WHERE id=?", (user_id,)).fetchone()
                if current and current["active"]:
                    connection.execute(
                        "INSERT INTO user_memories(user_id,content,updated_at) VALUES(?,?,?) "
                        "ON CONFLICT(user_id) DO UPDATE SET content=excluded.content,updated_at=excluded.updated_at",
                        (user_id, json.dumps(parsed, ensure_ascii=False), now_iso()),
                    )
        except Exception as exc:
            print(f"LocalMind memory extraction skipped: {type(exc).__name__}")

    def _mark_stream_error(self, assistant_id: str, text: str) -> None:
        status = "partial" if text else "error"
        try:
            with db() as connection:
                connection.execute("UPDATE messages SET content=?,status=? WHERE id=?", (text, status, assistant_id))
        except sqlite3.Error:
            pass

    def _mark_stream_cancelled(self, assistant_id: str, text: str) -> None:
        try:
            with db() as connection:
                connection.execute("UPDATE messages SET content=?,status='partial' WHERE id=?", (text, assistant_id))
        except sqlite3.Error:
            pass

    def do_PATCH(self) -> None:
        path = unquote(urlsplit(self.path).path)
        body = self._body_or_error()
        if body is None:
            return
        if path.startswith("/api/admin/users/"):
            admin = self._require_admin()
            if admin:
                self._patch_user(path.rsplit("/", 1)[-1], admin, body)
            return
        if path.startswith("/api/admin/tools/"):
            if self._require_admin():
                self._patch_tool(path.rsplit("/", 1)[-1], body)
            return
        user = self._require_user()
        if not user:
            return
        github_project_match = re.fullmatch(r"/api/projects/([a-f0-9]{32})/github", path)
        if github_project_match:
            enabled = body.get("auto_sync")
            if not isinstance(enabled, bool):
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "auto_sync phải là true hoặc false."})
                return
            project_id = github_project_match.group(1)
            with db() as connection:
                cursor = connection.execute("UPDATE github_projects SET auto_sync=?,sync_status=CASE WHEN ?=0 THEN 'paused' ELSE 'idle' END,sync_error='',updated_at=? WHERE project_id=? AND user_id=?", (int(enabled), int(enabled), now_iso(), project_id, user["id"]))
            if not cursor.rowcount:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Dự án chưa liên kết repo GitHub."})
                return
            if enabled:
                schedule_github_sync(project_id, user["id"], force=True)
            self._send_json(HTTPStatus.OK, {"auto_sync": enabled})
            return
        match = re.fullmatch(r"/api/conversations/([a-f0-9]{32})", path)
        if match:
            conversation = self._owned_conversation(match.group(1), user["id"])
            if not conversation:
                return
            title = str(body.get("title", "")).strip()[:80]
            if not title:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Tên cuộc trò chuyện không được để trống."})
                return
            with db() as connection:
                connection.execute("UPDATE conversations SET title=?,updated_at=? WHERE id=?", (title, now_iso(), conversation["id"]))
            self._send_json(HTTPStatus.OK, {"ok": True, "title": title})
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Không tìm thấy API."})

    def _patch_user(self, user_id: str, admin: sqlite3.Row, body: dict) -> None:
        role = body.get("role")
        active = body.get("active")
        display_name = body.get("display_name")
        password = body.get("password")
        with db() as connection:
            row = connection.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
            if not row:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Không tìm thấy tài khoản."})
                return
            if user_id == admin["id"] and (role and role != "admin" or active is False):
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Không thể tự hạ quyền hoặc vô hiệu hóa tài khoản đang dùng."})
                return
            if role is not None and role not in ("admin", "user"):
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Vai trò không hợp lệ."})
                return
            if password is not None and len(str(password)) < 12:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Mật khẩu cần ít nhất 12 ký tự."})
                return
            connection.execute(
                "UPDATE users SET role=?,active=?,display_name=?,password_hash=? WHERE id=?",
                (role or row["role"], int(bool(active)) if active is not None else row["active"],
                 (str(display_name).strip()[:64] or row["display_name"]) if display_name is not None else row["display_name"],
                 password_hash(str(password)) if password is not None else row["password_hash"], user_id),
            )
            updated = connection.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
            if active is False or password is not None:
                connection.execute("DELETE FROM sessions WHERE user_id=?", (user_id,))
        self._send_json(HTTPStatus.OK, {"user": user_public(updated)})

    def do_PUT(self) -> None:
        path = unquote(urlsplit(self.path).path)
        body = self._body_or_error()
        if body is None:
            return
        file_match=re.fullmatch(r"/api/projects/([a-f0-9]{32})/file",path)
        if file_match:
            user=self._require_user()
            if not user: return
            project_id=file_match.group(1)
            try:
                file_path=safe_project_path(body.get("path",""))
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST,{"error":str(exc)}); return
            content=body.get("content")
            if not isinstance(content,str) or len(content.encode("utf-8"))>1_000_000:
                self._send_json(HTTPStatus.BAD_REQUEST,{"error":"Tệp phải là văn bản và tối đa 1 MB."}); return
            with db() as connection:
                owned=connection.execute("SELECT 1 FROM projects WHERE id=? AND user_id=?",(project_id,user["id"])).fetchone()
                if not owned:
                    self._send_json(HTTPStatus.NOT_FOUND,{"error":"Không tìm thấy dự án."}); return
                used=connection.execute("SELECT COALESCE(SUM(length(CAST(content AS BLOB))),0) FROM project_files WHERE project_id=?",(project_id,)).fetchone()[0]
                old=connection.execute("SELECT id,content,length(CAST(content AS BLOB)) size FROM project_files WHERE project_id=? AND path=?",(project_id,file_path)).fetchone()
                file_count=connection.execute("SELECT COUNT(*) FROM project_files WHERE project_id=?",(project_id,)).fetchone()[0]
                if not old and file_count>=500:
                    self._send_json(HTTPStatus.INSUFFICIENT_STORAGE,{"error":"Dự án đạt giới hạn 500 tệp."}); return
                if used-(old["size"] if old else 0)+len(content.encode("utf-8"))>20_000_000:
                    self._send_json(HTTPStatus.INSUFFICIENT_STORAGE,{"error":"Dự án tối đa 20 MB."}); return
                if old and old["content"] == content:
                    self._send_json(HTTPStatus.OK,{"ok":True,"unchanged":True,"path":file_path}); return
                stamp=now_iso(); file_id=old["id"] if old else uuid.uuid4().hex
                connection.execute("INSERT INTO project_files(id,project_id,path,content,updated_at) VALUES(?,?,?,?,?) ON CONFLICT(project_id,path) DO UPDATE SET content=excluded.content,updated_at=excluded.updated_at",(file_id,project_id,file_path,content,stamp))
                autosave=body.get("autosave") is True
                recent=connection.execute("SELECT id FROM project_versions WHERE project_id=? AND path=? AND julianday('now')-julianday(created_at)<15.0/86400 ORDER BY id DESC LIMIT 1",(project_id,file_path)).fetchone() if autosave else None
                if recent:
                    connection.execute("UPDATE project_versions SET content=?,created_at=?,source='autosave' WHERE id=?",(content,stamp,recent["id"]))
                else:
                    connection.execute("INSERT INTO project_versions(project_id,path,content,created_at,source) VALUES(?,?,?,?,?)",(project_id,file_path,content,stamp,"autosave" if autosave else "manual"))
                trim_project_versions(connection,project_id,file_path)
                connection.execute("UPDATE projects SET updated_at=? WHERE id=?",(stamp,project_id))
            schedule_github_sync(project_id, user["id"])
            self._send_json(HTTPStatus.OK,{"ok":True,"path":file_path,"updated_at":stamp}); return
        if path == "/api/admin/settings":
            if not self._require_admin():
                return
            try:
                value = safe_base_url(str(body.get("llama_base_url", "")))
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            app_name = str(body.get("app_name", "LocalMind")).strip()[:32] or "LocalMind"
            accent = str(body.get("accent_color", "#b3a0ff")).strip()
            github_app_id = str(body.get("github_app_client_id", "")).strip()
            if len(github_app_id) > 100 or (github_app_id and not re.fullmatch(r"[A-Za-z0-9._-]+", github_app_id)):
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "GitHub App Client ID không hợp lệ."})
                return
            if not re.fullmatch(r"#[0-9A-Fa-f]{6}", accent):
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Màu nhấn cần theo dạng mã hex, ví dụ #b3a0ff."})
                return
            with db() as connection:
                connection.execute("INSERT INTO settings(key,value) VALUES('llama_base_url',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (value,))
                connection.execute("INSERT INTO settings(key,value) VALUES('app_name',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (app_name,))
                connection.execute("INSERT INTO settings(key,value) VALUES('accent_color',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (accent,))
                connection.execute("INSERT INTO settings(key,value) VALUES('github_app_client_id',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (github_app_id,))
            self._send_json(HTTPStatus.OK, {"llama_base_url": value, "app_name": app_name, "accent_color": accent, "github_app_client_id": github_app_id})
            return
        if path == "/api/admin/ai-settings":
            if not self._require_admin():
                return
            with db() as connection:
                current = dict(connection.execute("SELECT key,value FROM settings WHERE key IN ('ai_display_name','ai_language','ai_system_prompt','ai_tools_enabled')").fetchall())
            display_name = str(body.get("ai_display_name", current.get("ai_display_name", "AI nội bộ"))).strip()[:80] or "AI nội bộ"
            language = str(body.get("ai_language", current.get("ai_language", "auto")))
            system_prompt = str(body.get("ai_system_prompt", current.get("ai_system_prompt", "")))
            tools_enabled = body.get("ai_tools_enabled", current.get("ai_tools_enabled", "0") == "1")
            if not isinstance(tools_enabled, bool):
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Công tắc Tools không hợp lệ."})
                return
            if language not in LANGUAGE_NAMES:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Ngôn ngữ mặc định không hợp lệ."})
                return
            if len(system_prompt) > 12000:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "System prompt tối đa 12.000 ký tự."})
                return
            with db() as connection:
                for key, value in (("ai_display_name", display_name), ("ai_language", language), ("ai_system_prompt", system_prompt), ("ai_tools_enabled", "1" if tools_enabled else "0")):
                    connection.execute("INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
            self._send_json(HTTPStatus.OK, {"ai_display_name": display_name, "ai_language": language, "ai_tools_enabled": tools_enabled})
            return
        if path == "/api/admin/tools-enabled":
            if not self._require_admin():
                return
            enabled = body.get("enabled")
            if not isinstance(enabled, bool):
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Trạng thái Tools phải là true hoặc false."})
                return
            with db() as connection:
                connection.execute("INSERT INTO settings(key,value) VALUES('ai_tools_enabled',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", ("1" if enabled else "0",))
            if not enabled:
                with ACTIVE_GENERATIONS_LOCK:
                    active = list(ACTIVE_GENERATIONS.values())
                for state in active:
                    response = state.get("tool_response")
                    if response:
                        try:
                            response.close()
                        except OSError:
                            pass
            self._send_json(HTTPStatus.OK, {"enabled": enabled})
            return
        if path == "/api/memory":
            user = self._require_user()
            if user:
                self._save_user_memory(user["id"], body.get("memories"))
            return
        match = re.fullmatch(r"/api/admin/memory/([a-f0-9]{32})", path)
        if match:
            if not self._require_admin():
                return
            with db() as connection:
                exists = connection.execute("SELECT 1 FROM users WHERE id=?", (match.group(1),)).fetchone()
            if not exists:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Không tìm thấy tài khoản."})
                return
            self._save_user_memory(match.group(1), body.get("memories"))
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Không tìm thấy API."})

    def do_DELETE(self) -> None:
        path = unquote(urlsplit(self.path).path)
        if path == "/api/github/account":
            user = self._require_user()
            if not user:
                return
            with db() as connection:
                project_rows = connection.execute("SELECT project_id FROM github_projects WHERE user_id=?", (user["id"],)).fetchall()
                connection.execute("DELETE FROM github_accounts WHERE user_id=?", (user["id"],))
                connection.execute("UPDATE github_projects SET auto_sync=0,sync_status='paused',sync_error='Tài khoản GitHub đã ngắt kết nối.',updated_at=? WHERE user_id=?", (now_iso(), user["id"]))
            try:
                github_token_path(user["id"]).unlink(missing_ok=True)
            except OSError:
                pass
            with GITHUB_SYNC_LOCK:
                for row in project_rows:
                    timer = GITHUB_SYNC_TIMERS.pop(row["project_id"], None)
                    if timer:
                        timer.cancel()
            self._send_json(HTTPStatus.OK, {"ok": True, "remote_repositories_preserved": True})
            return
        project_github_match = re.fullmatch(r"/api/projects/([a-f0-9]{32})/github", path)
        if project_github_match:
            user = self._require_user()
            if not user:
                return
            project_id = project_github_match.group(1)
            with db() as connection:
                cursor = connection.execute("DELETE FROM github_projects WHERE project_id=? AND user_id=? AND EXISTS (SELECT 1 FROM projects WHERE id=? AND user_id=?)", (project_id, user["id"], project_id, user["id"]))
            if not cursor.rowcount:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Dự án chưa liên kết repo GitHub."})
                return
            with GITHUB_SYNC_LOCK:
                timer = GITHUB_SYNC_TIMERS.pop(project_id, None)
                if timer:
                    timer.cancel()
            self._send_json(HTTPStatus.OK, {"ok": True, "remote_repository_preserved": True})
            return
        project_match = re.fullmatch(r"/api/projects/([a-f0-9]{32})", path)
        if project_match:
            user = self._require_user()
            if not user:
                return
            with db() as connection:
                cursor = connection.execute("DELETE FROM projects WHERE id=? AND user_id=?", (project_match.group(1), user["id"]))
            if not cursor.rowcount:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Không tìm thấy dự án."})
                return
            self._send_json(HTTPStatus.OK, {"ok": True})
            return
        match = re.fullmatch(r"/api/admin/mcp-servers/([a-f0-9]{32})", path)
        if match:
            if not self._require_admin():
                return
            with db() as connection:
                connection.execute("DELETE FROM tool_registry WHERE server=?", (match.group(1),))
                cursor = connection.execute("DELETE FROM mcp_servers WHERE id=?", (match.group(1),))
            self._send_json(HTTPStatus.OK, {"ok": True, "deleted": cursor.rowcount > 0})
            return
        match = re.fullmatch(r"/api/admin/tools/([a-f0-9]{32})", path)
        if match:
            if not self._require_admin():
                return
            with db() as connection:
                connection.execute("DELETE FROM tool_registry WHERE id=?", (match.group(1),))
            self._send_json(HTTPStatus.OK, {"ok": True})
            return
        if path == "/api/memory":
            user = self._require_user()
            if not user:
                return
            with db() as connection:
                connection.execute("DELETE FROM user_memories WHERE user_id=?", (user["id"],))
            self._send_json(HTTPStatus.OK, {"ok": True})
            return
        match = re.fullmatch(r"/api/admin/memory/([a-f0-9]{32})", path)
        if match:
            if not self._require_admin():
                return
            with db() as connection:
                connection.execute("DELETE FROM user_memories WHERE user_id=?", (match.group(1),))
            self._send_json(HTTPStatus.OK, {"ok": True})
            return
        match = re.fullmatch(r"/api/conversations/([a-f0-9]{32})", path)
        if match:
            user = self._require_user()
            if not user:
                return
            conversation = self._owned_conversation(match.group(1), user["id"])
            if not conversation:
                return
            with db() as connection:
                connection.execute("DELETE FROM conversations WHERE id=?", (conversation["id"],))
            self._send_json(HTTPStatus.OK, {"ok": True})
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Không tìm thấy API."})


def main() -> None:
    init_db()
    resume_pending_github_syncs()
    server = ThreadingHTTPServer((HOST, PORT), LocalMindHandler)
    print(f"LocalMind listening on http://{HOST}:{PORT}")
    print(f"SQLite data: {DB_PATH}")
    print(f"llama.cpp API: {base_url()}")
    with db() as connection:
        needs_setup = connection.execute("SELECT 1 FROM users LIMIT 1").fetchone() is None
    if needs_setup:
        print(f"First admin setup code: {SETUP_TOKEN}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping LocalMind")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
