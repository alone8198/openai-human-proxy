"""
OpenAI Human Proxy — 模拟 OpenAI 标准 API 接口，支持人工回复。

工作流程：
1. 客户端调用 /v1/chat/completions 发送请求
2. 请求进入"待处理"队列，在管理后台显示
3. 人类在后台查看并输入回复
4. 等待中的 API 调用收到回复，格式完全兼容 OpenAI 响应

修复的 Bug：
- ❌ ~~客户端断开后请求从后台消失，来不及回复~~
  ✅ 客户端断开后请求仍保留在后台，可继续回复
- ❌ ~~流式响应长时间无数据，客户端超时断开~~
  ✅ 每 15 秒发送 SSE keepalive 心跳保活
- ❌ ~~已完成的回复无法查看~~
  ✅ 已完成的回复在"历史记录"中可查询
"""

import asyncio
import json
import time
import uuid
from datetime import datetime
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Form, Query
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# ============================================================
# 应用初始化
# ============================================================

app = FastAPI(
    title="OpenAI Human Proxy",
    description="模拟 OpenAI 标准 API，支持人工回复的代理服务",
    version="1.1.0",
)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ============================================================
# 数据结构
# ============================================================

# 待处理的请求队列 {request_id: PendingRequest}
pending_requests: dict[str, "PendingRequest"] = {}

# 已完成的请求缓存 {request_id: CompletedRequest}
completed_requests: dict[str, "CompletedRequest"] = {}

# SSE / 等待事件 —— 用于通知等待中的 API 调用
pending_events: dict[str, asyncio.Event] = {}

# 已处理消息计数（用于消息 ID 生成）
_message_counter = 0

# SSE 心跳间隔（秒）
SSE_HEARTBEAT_INTERVAL = 15
# 最长等待人工回复（秒）
HUMAN_REPLY_TIMEOUT = 3600


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "human-gpt"
    messages: list[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stop: Optional[list[str]] = None
    seed: Optional[int] = None


class PendingRequest:
    """待人工处理的请求"""

    def __init__(self, request_id: str, model: str, messages: list[dict],
                 original_request: ChatCompletionRequest):
        self.request_id = request_id
        self.model = model
        self.messages = messages
        self.original_request = original_request
        self.created_at = datetime.now().isoformat()
        self.reply: Optional[str] = None
        self.is_done = False
        self.is_stream = original_request.stream
        # 标记客户端是否还连着
        self.client_connected = True


class CompletedRequest:
    """已完成（人工已回复）的请求"""

    def __init__(self, pending: PendingRequest, reply_content: str,
                 client_was_connected: bool = False):
        global _message_counter
        _message_counter += 1
        self.request_id = pending.request_id
        self.model = pending.model
        self.messages = pending.messages
        self.reply_content = reply_content
        self.created_at = pending.created_at
        self.replied_at = datetime.now().isoformat()
        self.completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        self.client_was_connected = client_was_connected
        self.usage = {
            "prompt_tokens": self._count_tokens(
                [m["content"] for m in pending.messages]
            ),
            "completion_tokens": self._count_tokens([reply_content]),
            "total_tokens": self._count_tokens(
                [m["content"] for m in pending.messages] + [reply_content]
            ),
        }

    @staticmethod
    def _count_tokens(texts: list[str]) -> int:
        """粗略估算 token 数（约 4 字符 = 1 token）"""
        return sum(len(t) // 4 + 1 for t in texts)


def make_openai_chunk(request_id: str, model: str, delta_content: str,
                      finish_reason: Optional[str] = None,
                      created: Optional[int] = None) -> str:
    """构造 SSE 格式的 OpenAI 流式响应块"""
    if created is None:
        created = int(time.time())
    choice = {
        "index": 0,
        "delta": {"content": delta_content} if delta_content else {},
        "finish_reason": finish_reason,
    }
    if finish_reason:
        choice["delta"] = {}
    chunk = {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [choice],
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


def make_openai_response(request_id: str, model: str, content: str,
                         completion_id: str,
                         usage: dict, created: Optional[int] = None
                         ) -> dict:
    """构造完整的 OpenAI 非流式响应"""
    if created is None:
        created = int(time.time())
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
        "usage": usage,
        "system_fingerprint": f"fp_{uuid.uuid4().hex[:12]}",
    }


def _cleanup_request(request_id: str, move_to_completed: bool = False):
    """
    清理请求资源。

    核心规则：
    - 如果人类已回复 → 移到 completed_requests 存档
    - 否则 → 保留在 pending_requests 中，人类仍可在后台回复
    - 只清除 pending_events（事件句柄）
    """
    pending = pending_requests.get(request_id)
    event = pending_events.get(request_id)

    # 清除等待事件（这个不再需要了）
    if event:
        pending_events.pop(request_id, None)

    if not pending:
        return

    # 如果人类已回复 → 归档到 completed_requests 并移除 pending
    if pending.is_done and pending.reply is not None:
        completed = CompletedRequest(
            pending, pending.reply,
            client_was_connected=pending.client_connected,
        )
        completed_requests[request_id] = completed
        pending_requests.pop(request_id, None)
    # 如果人类还没有回复 → 保留在 pending（客户端可能断了，但人还可以继续回）
    # 但标记客户端已断开
    else:
        pending.client_connected = False


# ============================================================
# OpenAI 兼容接口
# ============================================================


@app.get("/v1/models")
async def list_models():
    """返回可用模型列表（OpenAI 兼容）"""
    return {
        "object": "list",
        "data": [
            {
                "id": "human-gpt",
                "object": "model",
                "created": 1700000000,
                "owned_by": "human-proxy",
            },
            {
                "id": "human-gpt-4o",
                "object": "model",
                "created": 1700000000,
                "owned_by": "human-proxy",
            },
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest, request: Request):
    """OpenAI 兼容的聊天补全接口——请求会进入人工审核队列"""
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    # 构建待处理请求
    messages_dict = [m.model_dump() for m in body.messages]
    pending = PendingRequest(
        request_id=request_id,
        model=body.model,
        messages=messages_dict,
        original_request=body,
    )
    pending_requests[request_id] = pending

    # 创建等待事件
    event = asyncio.Event()
    pending_events[request_id] = event

    if body.stream:
        return await _handle_streaming(request_id, model=body.model,
                                       created=created, event=event)
    else:
        return await _handle_nonstreaming(request_id, model=body.model,
                                          created=created, event=event)


async def _handle_streaming(request_id: str, model: str,
                            created: int, event: asyncio.Event):
    """处理流式请求——等待人工回复后以 SSE 流式输出"""

    async def event_generator():
        try:
            # 先发一个角色标识块（空的 delta）
            yield make_openai_chunk(
                request_id, model, "",
                created=created,
            )

            # 等待人工回复 —— 同时发送 SSE keepalive 心跳
            reply_content: Optional[str] = None
            while True:
                try:
                    await asyncio.wait_for(event.wait(),
                                           timeout=SSE_HEARTBEAT_INTERVAL)
                    # Event 被 set → 有人回复了
                    pending = pending_requests.get(request_id)
                    if pending and pending.reply is not None:
                        reply_content = pending.reply
                    break
                except asyncio.TimeoutError:
                    # 超时是正常的——发一个 SSE keepalive 注释
                    # SSE 协议中 : 开头的行是注释，客户端会忽略
                    yield ": keepalive\n\n"

            if reply_content is None:
                yield make_openai_chunk(
                    request_id, model, "⚠️ 回复为空或请求已过期",
                    finish_reason="stop", created=created,
                )
                yield "data: [DONE]\n\n"
                return

            # 发送回复内容
            yield make_openai_chunk(
                request_id, model, reply_content,
                finish_reason=None, created=created,
            )
            # 结束标记
            yield make_openai_chunk(
                request_id, model, "",
                finish_reason="stop", created=created,
            )
            yield "data: [DONE]\n\n"

        except asyncio.CancelledError:
            # 客户端断开连接 → 触发 CancelledError
            # 保留 pending_requests（人还能后台回复）
            _cleanup_request(request_id)
            return

        except Exception:
            _cleanup_request(request_id)
            raise

        finally:
            # finally 会在正常 return 或异常退出时执行
            # 注意：如果上面 except CancelledError 里已处理，这里还会再执行一次
            # 但 _cleanup_request 是幂等的，所以没问题
            _cleanup_request(request_id)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _handle_nonstreaming(request_id: str, model: str,
                               created: int, event: asyncio.Event):
    """处理非流式请求——等待人工回复后返回完整响应"""
    try:
        await asyncio.wait_for(event.wait(), timeout=HUMAN_REPLY_TIMEOUT)
    except asyncio.TimeoutError:
        # 超时 → 清理（保留给后台回复）
        _cleanup_request(request_id)
        raise HTTPException(
            status_code=408,
            detail="请求超时：人工回复等待超时",
        )
    except asyncio.CancelledError:
        # 客户端断开连接
        _cleanup_request(request_id)
        raise

    pending = pending_requests.get(request_id)
    if not pending or not pending.reply:
        _cleanup_request(request_id)
        raise HTTPException(
            status_code=404,
            detail="请求已过期或回复为空",
        )

    # 构建完整响应
    completed = CompletedRequest(pending, pending.reply,
                                 client_was_connected=True)
    completed_requests[request_id] = completed
    pending_requests.pop(request_id, None)
    pending_events.pop(request_id, None)

    return make_openai_response(
        request_id=request_id,
        model=model,
        content=pending.reply,
        completion_id=completed.completion_id,
        usage=completed.usage,
        created=created,
    )


# ============================================================
# 管理后台 —— 查看 & 回复
# ============================================================


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """管理后台首页——查看待处理和已完成的请求"""
    # 待处理
    pending_items = []
    for rid, p in pending_requests.items():
        last_msg = p.messages[-1]["content"] if p.messages else ""
        pending_items.append({
            "id": rid,
            "model": p.model,
            "messages_count": len(p.messages),
            "last_message_preview": last_msg[:200] + ("..." if len(last_msg) > 200 else ""),
            "created_at": p.created_at,
            "is_stream": p.is_stream,
            "client_connected": p.client_connected,
        })
    pending_items.sort(key=lambda x: x["created_at"], reverse=True)

    # 已完成的历史记录
    completed_items = []
    for rid, c in completed_requests.items():
        last_msg = c.messages[-1]["content"] if c.messages else ""
        completed_items.append({
            "id": rid,
            "model": c.model,
            "last_message_preview": last_msg[:200] + ("..." if len(last_msg) > 200 else ""),
            "reply_preview": c.reply_content[:200] + ("..." if len(c.reply_content) > 200 else ""),
            "created_at": c.created_at,
            "replied_at": c.replied_at,
            "client_was_connected": c.client_was_connected,
        })
    completed_items.sort(key=lambda x: x["replied_at"], reverse=True)

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "pending_requests": pending_items,
            "completed_requests": completed_items,
        },
    )


@app.get("/request/{request_id}", response_class=HTMLResponse)
async def view_request(request: Request, request_id: str):
    """查看单个请求的完整详情"""
    # 先查 pending
    pending = pending_requests.get(request_id)
    if pending:
        return templates.TemplateResponse(
            "request_detail.html",
            {
                "request": request,
                "req": {
                    "id": pending.request_id,
                    "model": pending.model,
                    "messages": pending.messages,
                    "created_at": pending.created_at,
                    "reply": pending.reply,
                    "is_done": pending.is_done,
                    "is_stream": pending.is_stream,
                    "client_connected": pending.client_connected,
                },
            },
        )

    # 再查 completed
    completed = completed_requests.get(request_id)
    if completed:
        return templates.TemplateResponse(
            "request_detail.html",
            {
                "request": request,
                "req": {
                    "id": completed.request_id,
                    "model": completed.model,
                    "messages": completed.messages,
                    "created_at": completed.created_at,
                    "replied_at": completed.replied_at,
                    "reply": completed.reply_content,
                    "is_done": True,
                    "is_stream": False,
                    "client_connected": completed.client_was_connected,
                },
            },
        )

    raise HTTPException(status_code=404, detail="请求不存在或已过期")


@app.post("/request/{request_id}/reply")
async def reply_request(request_id: str, content: str = Form(...)):
    """人工回复指定请求"""
    pending = pending_requests.get(request_id)
    if not pending:
        raise HTTPException(status_code=404, detail="请求不存在或已过期")

    pending.reply = content
    pending.is_done = True

    # 如果有客户端还在等 → 触发事件
    event = pending_events.get(request_id)
    if event:
        event.set()
    else:
        # 客户端已断开 → 直接归档
        _cleanup_request(request_id)

    return RedirectResponse(url="/", status_code=303)


@app.post("/request/{request_id}/cancel")
async def cancel_request(request_id: str):
    """取消待处理的请求"""
    pending = pending_requests.get(request_id)
    if not pending:
        raise HTTPException(status_code=404, detail="请求不存在")

    pending_requests.pop(request_id, None)
    pending_events.pop(request_id, None)

    return RedirectResponse(url="/", status_code=303)


@app.get("/api/pending", response_class=JSONResponse)
async def api_pending_requests():
    """API：获取待处理请求列表（JSON 格式）"""
    items = []
    for rid, p in pending_requests.items():
        items.append({
            "id": rid,
            "model": p.model,
            "messages": p.messages,
            "created_at": p.created_at,
            "is_stream": p.is_stream,
            "is_done": p.is_done,
            "has_reply": p.reply is not None,
            "client_connected": p.client_connected,
        })
    items.sort(key=lambda x: x["created_at"], reverse=True)
    return {"pending_requests": items}


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    import socket

    def find_free_port(start=8000, end=9000):
        for port in range(start, end):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(("127.0.0.1", port)) != 0:
                    return port
        return 8000

    port = find_free_port()
    print(f"🚀 OpenAI Human Proxy v1.1.0")
    print(f"📋 管理后台: http://127.0.0.1:{port}/")
    print(f"🤖 OpenAI API: http://127.0.0.1:{port}/v1/")
    print(f"📦 示例调用:")
    print(f'    curl http://127.0.0.1:{port}/v1/chat/completions \\')
    print(f'      -H "Content-Type: application/json" \\')
    print(f'      -d \'{{"model":"human-gpt","messages":[{{"role":"user","content":"你好"}}]}}\'')
    print()
    uvicorn.run(app, host="0.0.0.0", port=port)
