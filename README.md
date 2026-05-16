# OpenAI Human Proxy 🤖

**模拟 OpenAI 标准 API 接口，但回复由真人输入。**

![license](https://img.shields.io/badge/license-MIT-green)

## 这是什么？

这是一个完全兼容 OpenAI API 格式的代理服务。客户端可以使用 OpenAI 的 SDK 或直接调用 REST API 发请求，
但这些请求**不会发给 GPT**，而是进入一个人工审核队列。人类在管理后台看到请求后手动输入回复，
客户端的 API 调用便会收到响应。

### 适用场景

- 🧪 **测试 & 开发** — 用真人回复模拟 AI 输出，方便调试对话逻辑
- 🎭 **演示 & 评审** — 让产品经理/客户手动"扮演"AI，验证交互体验
- 🔐 **人工审核** — 在 AI 回复前需要人工确认的场景
- 📚 **教学 & 培训** — 教学员如何与 AI API 交互，用真人控制输出内容

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动服务
python main.py

# 3. 在浏览器打开管理后台
#    http://localhost:8000/
```

## API 接口

### 兼容 OpenAI Chat Completions

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "human-gpt",
    "messages": [
      {"role": "user", "content": "你好，请问今天天气怎么样？"}
    ]
  }'
```

请求会进入待处理队列，在管理后台中显示。你在后台输入回复后，API 调用会收到完全兼容 OpenAI 格式的响应：

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "human-gpt",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "今天天气很好，阳光明媚！"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 8,
    "total_tokens": 18
  }
}
```

### 流式响应 (Streaming)

支持 SSE 流式输出：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "human-gpt",
    "stream": true,
    "messages": [
      {"role": "user", "content": "请写一首诗"}
    ]
  }'
```

### 列出可用模型

```bash
curl http://localhost:8000/v1/models
```

## 管理后台

访问 `http://localhost:8000/` 即可打开管理后台：

1. **请求列表** — 所有待处理的 API 请求一目了然
2. **查看详情** — 点击某个请求查看完整对话上下文
3. **输入回复** — 在文本框中输入回复内容，点击发送
4. **客户端即时收到** — 回复发送后，等待中的 API 调用立即收到响应

## 使用 Python OpenAI SDK 调用

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",  # 此代理不需要 API Key
)

response = client.chat.completions.create(
    model="human-gpt",
    messages=[
        {"role": "user", "content": "用 Python 写一个斐波那契函数"}
    ],
)

print(response.choices[0].message.content)
```

## 项目结构

```
.
├── main.py              # FastAPI 应用主文件
├── requirements.txt     # Python 依赖
├── templates/
│   ├── dashboard.html   # 管理后台首页
│   └── request_detail.html  # 请求详情页
├── static/
│   └── style.css        # 后台样式
└── README.md
```

## 自定义

- 修改 `timeout` 值（在 `_handle_streaming` 和 `_handle_nonstreaming` 中）可调整最长等待时间
- 修改 `/v1/models` 端点可自定义返回的模型列表
- 可通过 `api_key` 验证增强安全性

## License

MIT
