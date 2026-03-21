"""
Telegram Chat Summarizer Bot  –  Option A: Bot-as-Admin
────────────────────────────────────────────────────────
Stack:
  • aiogram  3.x    – command handling & replies
  • Pyrogram  2.x   – Client-API history fetching (same bot token, no user session)
  • huggingface_hub – Inference API for summarisation (no GPU needed)
  • python-dotenv   – config
"""

from __future__ import annotations
from aiogram.types import ChatMemberUpdated
import asyncio
import html
import logging
import os

from aiogram import Bot, Dispatcher, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import Message
from dotenv import load_dotenv
from huggingface_hub import AsyncInferenceClient
from huggingface_hub.errors import HfHubHTTPError
from pyrogram import Client
from pyrogram.errors import (
    ChatAdminRequired,
    FloodWait,
    MsgIdInvalid,
    PeerIdInvalid,
    UserNotParticipant,
)
from pyrogram.types import Message as PyroMessage

# ──────────────────────────────────────────────────────────────────────────────
# Bootstrap
# ──────────────────────────────────────────────────────────────────────────────

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Config  (all values from .env)
# ──────────────────────────────────────────────────────────────────────────────

# ── Telegram ──────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN: str = os.environ["TELEGRAM_BOT_TOKEN"]
PYROGRAM_API_ID:    int = int(os.environ["PYROGRAM_API_ID"])
PYROGRAM_API_HASH:  str = os.environ["PYROGRAM_API_HASH"]

# ── Hugging Face ──────────────────────────────────────────────────────────────
# FIXED: Updated to match 'HF_TOKEN' from your .env file
HF_API_TOKEN:  str = os.environ["HF_TOKEN"]
HF_MODEL:      str = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
HF_MAX_TOKENS: int = int(os.getenv("HF_MAX_TOKENS", "1024"))
HF_TIMEOUT:    int = int(os.getenv("HF_TIMEOUT_SEC", "120"))

# ── Safety caps ───────────────────────────────────────────────────────────────
MAX_CONTEXT_CHARS:     int = int(os.getenv("MAX_CONTEXT_CHARS",     "120000"))
MAX_MESSAGES_TO_FETCH: int = int(os.getenv("MAX_MESSAGES_TO_FETCH", "2000"))

# ──────────────────────────────────────────────────────────────────────────────
# Hugging Face client  (module-level singleton, re-used across all requests)
# ──────────────────────────────────────────────────────────────────────────────

hf_client = AsyncInferenceClient(
    model=HF_MODEL,
    token=HF_API_TOKEN,
    timeout=HF_TIMEOUT,
)

SYSTEM_PROMPT = """\
Ти — експерт-асистент, який створює підсумки для Telegram-чатів.
Ти отримуєш хронологічну історію повідомлень, де кожне повідомлення починається з імені відправника.

Твоє завдання:
• Написати зв'язний, суцільний текст-підсумок, який детально описує загальний хід розмови.
• Чітко вказати, хто з учасників що сказав, запропонував або яку думку висловив (наприклад: "Костя запропонував..., на що Данило відповів...").
• Об'єднувати дрібні повідомлення в одну логічну думку.
• НІКОЛИ не вигадувати інформацію, якої немає в тексті.
• Відповідати ТІЛЬКИ суцільним текстом — без маркованих списків, привітань, вступних слів чи прощань. Усі відповіді мають бути українською мовою.
"""

# ──────────────────────────────────────────────────────────────────────────────
# AI summarisation
# ──────────────────────────────────────────────────────────────────────────────

async def get_ai_summary(text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": text},
    ]

    try:
        response = await hf_client.chat_completion(
            messages=messages,
            max_tokens=HF_MAX_TOKENS,
            temperature=0.3,
        )
    except HfHubHTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "?"
        if status == 429:
            raise RuntimeError(
                "Hugging Face rate limit reached. "
                "Upgrade to PRO or wait before retrying."
            ) from exc
        if status == 503:
            raise RuntimeError(
                f"Model <code>{HF_MODEL}</code> is loading on HF servers "
                f"(cold start). Retry in ~30 s."
            ) from exc
        raise RuntimeError(
            f"Hugging Face API error {status}: {exc}"
        ) from exc
    except TimeoutError as exc:
        raise RuntimeError(
            f"Hugging Face request timed out after {HF_TIMEOUT}s. "
            "The model may be cold — try again or increase HF_TIMEOUT_SEC."
        ) from exc

    content = response.choices[0].message.content
    return content.strip() if content else "(model returned an empty response)"


# ──────────────────────────────────────────────────────────────────────────────
# Message collection helpers  (Pyrogram)
# ──────────────────────────────────────────────────────────────────────────────

def _is_noise(msg: PyroMessage) -> bool:
    if msg.service:
        return True
    text = (msg.text or msg.caption or "").strip()
    return text.startswith("/")


def _format_line(msg: PyroMessage) -> str:
    if msg.from_user:
        name = msg.from_user.first_name or "User"
        if msg.from_user.last_name:
            name += f" {msg.from_user.last_name}"
    elif msg.sender_chat:
        name = msg.sender_chat.title or "Channel"
    else:
        name = "Unknown"

    body = (msg.text or msg.caption or "").strip()
    if not body:
        parts: list[str] = []
        if msg.photo:    parts.append("📷 photo")
        if msg.video:    parts.append("🎥 video")
        if msg.document: parts.append("📎 document")
        if msg.sticker:  parts.append(f"😀 sticker {msg.sticker.emoji or ''}")
        if msg.voice:    parts.append("🎤 voice message")
        if msg.poll:     parts.append(f"📊 poll: «{msg.poll.question}»")
        body = f"[{', '.join(parts) or 'media'}]"

    return f"{name}: {body}"


async def collect_messages(
    pyro: Client,
    chat_id: int,
    start_id: int,
    end_id: int,
) -> list[str]:
    all_ids = list(range(start_id, end_id + 1))

    if len(all_ids) > MAX_MESSAGES_TO_FETCH:
        log.warning(
            "Range contains %d messages — keeping last %d.",
            len(all_ids), MAX_MESSAGES_TO_FETCH,
        )
        all_ids = all_ids[-MAX_MESSAGES_TO_FETCH:]

    BATCH = 200
    lines: list[str] = []

    for i in range(0, len(all_ids), BATCH):
        chunk = all_ids[i : i + BATCH]
        try:
            messages = await pyro.get_messages(chat_id, chunk)
        except FloodWait as exc:
            log.warning("Telegram FloodWait: sleeping %ds…", exc.value)
            await asyncio.sleep(exc.value)
            messages = await pyro.get_messages(chat_id, chunk)
        except ChatAdminRequired:
            raise RuntimeError(
                "Bot must be an admin with <b>Read Messages</b> permission in this chat."
            )
        except (PeerIdInvalid, UserNotParticipant):
            raise RuntimeError("Bot is not a member / admin of this chat.")
        except MsgIdInvalid:
            raise RuntimeError(
                "Invalid message ID range — the start message may have been deleted."
            )

        for msg in messages:
            if msg is None or msg.empty or _is_noise(msg):
                continue

            if msg.chat and msg.chat.id != chat_id:
                continue

            lines.append(_format_line(msg))

        await asyncio.sleep(0.05)

    return lines


# ──────────────────────────────────────────────────────────────────────────────
# aiogram router & /summarize command
# ──────────────────────────────────────────────────────────────────────────────


router = Router()

@router.message(Command("summarize"))
async def cmd_summarize(message: Message, pyro_client: Client) -> None:
    if not message.reply_to_message:
        await message.reply(
            "↩️  <b>How to use:</b>\n"
            "Reply to the message you want to start the summary from, "
            "then send /summarize.",
        )
        return

    start_id: int = message.reply_to_message.message_id
    end_id:   int = message.message_id - 1
    chat_id:  int = message.chat.id

    if end_id < start_id:
        await message.reply("⚠️ There are no messages between that point and now.")
        return

    status = await message.reply("⏳ Collecting messages…")

    try:
        lines = await collect_messages(pyro_client, chat_id, start_id, end_id)
    except RuntimeError as exc:
        await status.edit_text(
            f"❌ <b>History access error:</b>\n\n{exc}",
        )
        return

    if not lines:
        await status.edit_text(
            "ℹ️ <b>No readable messages found.</b>\n"
            "If there are messages here, I cannot see them. Please make sure I am promoted to <b>Admin</b>!"
        )
        return

    msg_count = len(lines)
    context   = "\n".join(lines)

    truncated = len(context) > MAX_CONTEXT_CHARS
    if truncated:
        context = "…[earlier messages omitted due to length]…\n" + context[-MAX_CONTEXT_CHARS:]
        log.warning("Context truncated to %d chars for chat %d.", MAX_CONTEXT_CHARS, chat_id)

    await status.edit_text(
        f"⏳ {msg_count} messages collected"
        + (" — context window truncated to fit model limits" if truncated else "")
        + ". Sending to Hugging Face…"
    )

    try:
        summary = await get_ai_summary(context)
    except RuntimeError as exc:
        await status.edit_text(
            f"❌ <b>AI error:</b> {html.escape(str(exc))}",
        )
        return

    chat_id_str = str(chat_id)
    pure_id = chat_id_str[4:] if chat_id_str.startswith("-100") else chat_id_str
    start_link = f"https://t.me/c/{pure_id}/{start_id}"

    header = (
        f"📋 <b>Summary</b> — "
        f"<a href='{start_link}'>{msg_count} messages</a>"
        + (" ⚠️ <i>(context truncated)</i>" if truncated else "")
        + "\n\n"
    )
    full_text = header + summary

    if len(full_text) <= 4_000:
        await status.edit_text(
            full_text,
            disable_web_page_preview=True,
        )
    else:
        chunks = [full_text[i : i + 4_000] for i in range(0, len(full_text), 4_000)]
        await status.edit_text(chunks[0], disable_web_page_preview=True)
        for chunk in chunks[1:]:
            await message.reply(chunk, disable_web_page_preview=True)


# ──────────────────────────────────────────────────────────────────────────────
# Startup & entry point
# ──────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    bot = Bot(
        token=TELEGRAM_BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dp = Dispatcher()
    dp.include_router(router)

    pyro = Client(
        name="summarizer_bot",
        api_id=PYROGRAM_API_ID,
        api_hash=PYROGRAM_API_HASH,
        bot_token=TELEGRAM_BOT_TOKEN,
    )

    log.info("Starting Pyrogram client…")
    await pyro.start()
    me = await pyro.get_me()
    log.info("Pyrogram ready → @%s (id=%d)", me.username, me.id)

    dp["pyro_client"] = pyro

    log.info("Starting aiogram polling…")
    try:
        await dp.start_polling(bot, allowed_updates=["message", "my_chat_member"])
    finally:
        log.info("Shutdown requested — stopping clients…")
        await pyro.stop()
        await bot.session.close()
        log.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())