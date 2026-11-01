
import os
import re
import asyncio
import time
from urllib.parse import parse_qs

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from groq import Groq

app = FastAPI(title="Binary Groq AI")

MODEL = "openai/gpt-oss-20b"
END_CODE = "11111111"
MAX_MESSAGE_LENGTH = 2000
CHARACTER_DELAY = 0.1

ALPHABET = {
    "00000000": " ",
    "00000001": "A",
    "00000010": "B",
    "00000011": "C",
    "00000100": "D",
    "00000101": "E",
    "00000110": "F",
    "00000111": "G",
    "00001000": "H",
    "00001001": "I",
    "00001010": "J",
    "00001011": "K",
    "00001100": "L",
    "00001101": "M",
    "00001110": "N",
    "00001111": "O",
    "00010000": "P",
    "00010001": "Q",
    "00010010": "R",
    "00010011": "S",
    "00010100": "T",
    "00010101": "U",
    "00010110": "V",
    "00010111": "W",
    "00011000": "X",
    "00011001": "Y",
    "00011010": "Z",
}

REVERSE_ALPHABET = {v: k for k, v in ALPHABET.items()}

message_buffer = []
latest_answer_codes = []

output_index = 0
response_ready = False
last_output_time = 0.0

conversation = [
    {
        "role": "system",
        "content": (
            "You are a helpful AI assistant inside a game. "
            "Reply clearly, naturally, and briefly. "
            "Use simple language."
        )
    }
]


def encode_answer(answer):
    clean = re.sub(r"[^A-Z ]", " ", answer.upper())
    clean = " ".join(clean.split())

    return [
        REVERSE_ALPHABET[character]
        for character in clean
    ]


def ask_groq(message):
    global latest_answer_codes
    global output_index
    global response_ready
    global last_output_time

    try:
        client = Groq(
            api_key=os.environ["GROQ_API_KEY"]
        )

        messages = conversation + [
            {"role": "user", "content": message}
        ]

        result = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,
            max_completion_tokens=150
        )

        answer = (
            result.choices[0].message.content
            or "NO ANSWER"
        )

        conversation.append({
            "role": "user",
            "content": message
        })

        conversation.append({
            "role": "assistant",
            "content": answer
        })

        if len(conversation) > 21:
            del conversation[1:3]

        latest_answer_codes = encode_answer(answer)

    except Exception as error:
        print("GROQ ERROR:", error)
        latest_answer_codes = encode_answer("AI ERROR")

    output_index = 0
    response_ready = True
    last_output_time = 0.0


async def read_value(request):
    body = (await request.body()).decode(
        "utf-8", errors="ignore"
    )

    data = parse_qs(body)

    if "value" in data:
        return data["value"][0]

    try:
        data = await request.json()
        return str(data.get("value", ""))
    except Exception:
        return request.query_params.get("value", "")


@app.post("/receive/1")
async def receive(request: Request):
    global message_buffer
    global response_ready
    global output_index

    value = (await read_value(request)).strip()

    if not re.fullmatch(r"[01]{8}", value):
        return JSONResponse({
            "value": END_CODE
        })

    if value == END_CODE:
        if not message_buffer:
            return JSONResponse({
                "value": END_CODE
            })

        message = "".join(message_buffer)
        message_buffer = []

        response_ready = False
        output_index = 0

        await asyncio.to_thread(
            ask_groq, message
        )

        return JSONResponse({
            "value": "00000000"
        })

    character = ALPHABET.get(value)

    if character is None:
        return JSONResponse({
            "value": END_CODE
        })

    if len(message_buffer) < MAX_MESSAGE_LENGTH:
        message_buffer.append(character)

    return JSONResponse({
        "value": value
    })


@app.get("/result/codes")
async def result_codes():
    global output_index
    global last_output_time

    if not response_ready:
        return JSONResponse({
            "value": "00000000"
        })

    now = time.monotonic()

    if last_output_time and now - last_output_time < CHARACTER_DELAY:
        return JSONResponse({
            "value": "00000000"
        })

    if output_index < len(latest_answer_codes):
        code = latest_answer_codes[output_index]
        output_index += 1
        last_output_time = now

        return JSONResponse({
            "value": code
        })

    return JSONResponse({
        "value": END_CODE
    })


@app.get("/result/stream")
async def result_stream():
    async def generate():
        for code in latest_answer_codes:
            yield code + "\n"
            await asyncio.sleep(CHARACTER_DELAY)

        yield END_CODE + "\n"

    return StreamingResponse(
        generate(),
        media_type="text/plain"
    )


@app.post("/reset")
def reset():
    global message_buffer
    global latest_answer_codes
    global output_index
    global response_ready

    message_buffer = []
    latest_answer_codes = []
    output_index = 0
    response_ready = False

    return JSONResponse({
        "value": "00000000"
    })


@app.get("/health")
def health():
    return {
        "status": "ok",
        "groq_key_configured": bool(
            os.getenv("GROQ_API_KEY")
        ),
        "model": MODEL
    }


@app.get("/")
def home():
    return {"status": "online"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000))
    )
