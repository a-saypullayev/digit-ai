
import os
import re
import asyncio
from urllib.parse import parse_qs

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from groq import Groq

app = FastAPI(title="Build a Logic Groq AI")

# -----------------------------
# SETTINGS
# -----------------------------

MODEL = "openai/gpt-oss-20b"
END_CODE = "11111111"
MAX_MESSAGE_LENGTH = 2000

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

# -----------------------------
# MEMORY
# -----------------------------

message_buffer = []
conversation = [
    {
        "role": "system",
        "content": (
            "You are a helpful AI assistant inside a game. "
            "Reply clearly and naturally. "
            "Use simple language."
        )
    }
]

latest_answer = ""
latest_answer_codes = []
last_received_message = ""

# -----------------------------
# GROQ
# -----------------------------

def ask_groq(message):
    global latest_answer
    global latest_answer_codes
    global last_received_message

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return "ERROR GROQ API KEY IS MISSING"

    client = Groq(api_key=api_key)

    conversation.append({
        "role": "user",
        "content": message
    })

    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=conversation,
            temperature=0.7,
            max_completion_tokens=300
        )

        answer = completion.choices[0].message.content or ""

        conversation.append({
            "role": "assistant",
            "content": answer
        })

        # Keep conversation memory from growing forever.
        if len(conversation) > 21:
            del conversation[1:3]

        latest_answer = answer

        # Our alphabet only supports A-Z and spaces.
        clean_answer = answer.upper()
        clean_answer = re.sub(r"[^A-Z ]", " ", clean_answer)
        clean_answer = " ".join(clean_answer.split())

        latest_answer_codes = [
            REVERSE_ALPHABET[character]
            for character in clean_answer
        ]

        last_received_message = message

        return answer

    except Exception as error:
        conversation.pop()
        return "GROQ ERROR: " + str(error)


# -----------------------------
# READ TRANSMITTER DATA
# -----------------------------

async def read_value(request: Request):
    body = (await request.body()).decode("utf-8", errors="ignore")

    # Build a Logic uses application/x-www-form-urlencoded.
    form_data = parse_qs(body)
    if "value" in form_data:
        return form_data["value"][0]

    # Also accept JSON, if supported by a different transmitter.
    try:
        data = await request.json()
        return str(data.get("value", ""))
    except Exception:
        pass

    # Also accept a query parameter.
    return request.query_params.get("value", "")


# -----------------------------
# RECEIVE ONE CHARACTER
# -----------------------------

@app.post("/receive/{transmitter_id}")
async def receive_character(
    transmitter_id: int,
    request: Request
):
    global latest_answer

    value = (await read_value(request)).strip()

    if not re.fullmatch(r"[01]{8}", value):
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": "Expected exactly 8 binary digits",
                "received": value
            }
        )

    # End of message!
    if value == END_CODE:
        if not message_buffer:
            return {
                "status": "error",
                "message": "Message is empty"
            }

        message = "".join(message_buffer)
        message_buffer.clear()

        print("USER MESSAGE:", message)

        # Ask Groq without blocking the async server.
        answer = await asyncio.to_thread(ask_groq, message)

        print("GROQ ANSWER:", answer)

        return {
            "status": "complete",
            "message": message,
            "answer": answer,
            "answer_codes": latest_answer_codes
        }

    # Decode character.
    character = ALPHABET.get(value)

    if character is None:
        return {
            "status": "error",
            "message": "Unknown character code"
        }

    if len(message_buffer) >= MAX_MESSAGE_LENGTH:
        message_buffer.clear()
        return {
            "status": "error",
            "message": "Message too long. Buffer cleared."
        }

    message_buffer.append(character)

    print(
        f"Transmitter {transmitter_id}: "
        f"{value} -> {character}"
    )

    return {
        "status": "receiving",
        "character": character,
        "characters_received": len(message_buffer)
    }


# -----------------------------
# GET AI ANSWER
# -----------------------------

@app.get("/result")
def get_result():
    return {
        "message": last_received_message,
        "answer": latest_answer
    }


@app.get("/result/codes")
def get_result_codes():
    return {
        "answer": latest_answer,
        "codes": latest_answer_codes,
        "end_code": END_CODE
    }


# -----------------------------
# RESET MESSAGE BUFFER
# -----------------------------

@app.post("/reset")
def reset_message():
    message_buffer.clear()

    return {
        "status": "reset"
    }


# -----------------------------
# HEALTH CHECK
# -----------------------------

@app.get("/")
def home():
    return {
        "status": "online",
        "name": "Build a Logic Groq AI"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "groq_key_configured": bool(os.getenv("GROQ_API_KEY")),
        "model": MODEL,
        "characters": len(ALPHABET),
        "buffer_length": len(message_buffer)
    }


# -----------------------------
# START SERVER
# -----------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
