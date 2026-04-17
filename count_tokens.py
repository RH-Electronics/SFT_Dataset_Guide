"""
🔬 Dataset Token Analyzer
Считает токены в каждом диалоге dataset.jsonl
используя реальный токенизатор модели.

Использование:
  python count_tokens.py dataset.jsonl

Если нет доступа к модели — будет fallback на tiktoken.
"""

import json
import sys
import statistics

path = "dataset.jsonl"


def load_conversations(path):
    """Загружает диалоги из JSONL файла."""
    convos = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                messages = obj.get("messages", obj.get("conversations", []))
                if messages:
                    convos.append({"index": i, "messages": messages})
            except json.JSONDecodeError as e:
                print(f"⚠️  Строка {i}: ошибка JSON — {e}")
    return convos


def try_qwen_tokenizer():
    """Пытается загрузить токенизатор."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            "unsloth/gemma-4-31B-it",
            trust_remote_code=True
        )
        print("✅ Используем токенизатор модели (точный подсчёт)")
        return tok
    except Exception as e:
        print(f"⚠️  токенизатор недоступен: {e}")
        return None


def try_tiktoken_fallback():
    """Fallback на tiktoken (приблизительный подсчёт)."""
    try:
        import tiktoken
        tok = tiktoken.get_encoding("cl100k_base")
        print("⚠️  Используем tiktoken cl100k_base (приблизительный подсчёт)")
        return tok
    except ImportError:
        return None


def count_tokens_transformers(tokenizer, messages):
    """Считает токены через apply_chat_template."""
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        tokens = tokenizer.encode(text)
        return len(tokens), text
    except Exception:
        # fallback — просто конкатенация
        text = " ".join(m.get("content", "") for m in messages)
        tokens = tokenizer.encode(text)
        return len(tokens), text


def count_tokens_tiktoken(tokenizer, messages):
    """Считает токены через tiktoken."""
    text = ""
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    tokens = tokenizer.encode(text)
    return len(tokens), text


def analyze(data_path):
    convos = load_conversations(data_path)
    if not convos:
        print("❌ Не найдено диалогов!")
        return

    print(f"📂 Файл: {data_path}")
    print(f"📊 Всего диалогов: {len(convos)}\n")

    # Выбираем токенизатор
    tokenizer = try_qwen_tokenizer()
    use_transformers = tokenizer is not None

    if not use_transformers:
        tokenizer = try_tiktoken_fallback()

    if tokenizer is None:
        print("❌ Нет доступного токенизатора!")
        print("   Установи: pip install tiktoken")
        return

    # Считаем
    token_counts = []
    long_examples = []  # диалоги > 2048

    for convo in convos:
        if use_transformers:
            n_tokens, _ = count_tokens_transformers(tokenizer, convo["messages"])
        else:
            n_tokens, _ = count_tokens_tiktoken(tokenizer, convo["messages"])

        token_counts.append(n_tokens)
        turns = len(convo["messages"])

        if n_tokens > 2048:
            long_examples.append({
                "line": convo["index"],
                "tokens": n_tokens,
                "turns": turns,
                "preview": convo["messages"][0].get("content", "")[:80]
            })

    # Статистика
    print("=" * 55)
    print("📈 СТАТИСТИКА ТОКЕНОВ")
    print("=" * 55)
    print(f"  Всего диалогов:     {len(token_counts)}")
    print(f"  Минимум:            {min(token_counts)} токенов")
    print(f"  Максимум:           {max(token_counts)} токенов")
    print(f"  Среднее:            {statistics.mean(token_counts):.0f} токенов")
    print(f"  Медиана:            {statistics.median(token_counts):.0f} токенов")

    if len(token_counts) > 1:
        print(f"  Стд. отклонение:    {statistics.stdev(token_counts):.0f} токенов")

    # Распределение по бакетам
    buckets = [128, 256, 512, 1024, 2048, 4096, 8192]
    print(f"\n📊 РАСПРЕДЕЛЕНИЕ:")
    prev = 0
    for b in buckets:
        count = sum(1 for t in token_counts if prev < t <= b)
        pct = count / len(token_counts) * 100
        bar = "█" * int(pct / 2)
        print(f"  {prev:>5}-{b:>5}: {count:>4} ({pct:5.1f}%) {bar}")
        prev = b
    overflow = sum(1 for t in token_counts if t > buckets[-1])
    if overflow:
        print(f"  {buckets[-1]:>5}+    : {overflow:>4} ({overflow/len(token_counts)*100:5.1f}%)")

    # Рекомендация по MAX_SEQ_LENGTH
    p95 = sorted(token_counts)[int(len(token_counts) * 0.95)]
    p99 = sorted(token_counts)[int(len(token_counts) * 0.99)]
    print(f"\n🎯 РЕКОМЕНДАЦИИ ПО MAX_SEQ_LENGTH:")
    print(f"  95-й перцентиль:    {p95} токенов")
    print(f"  99-й перцентиль:    {p99} токенов")

    if p95 <= 512:
        rec = 512
    elif p95 <= 1024:
        rec = 1024
    elif p95 <= 2048:
        rec = 2048
    else:
        rec = 4096

    print(f"  ➡️  Рекомендую:      MAX_SEQ_LENGTH = {rec}")
    if rec < 2048:
        saved = (2048 - rec) / 2048 * 100
        print(f"  💾 Экономия VRAM:   ~{saved:.0f}% на активациях vs 2048")

    # Длинные примеры
    if long_examples:
        print(f"\n⚠️  ДИАЛОГИ ДЛИННЕЕ 2048 ТОКЕНОВ ({len(long_examples)} шт.):")
        for ex in long_examples[:10]:
            print(f"  Строка {ex['line']:>4}: {ex['tokens']:>5} tok, "
                  f"{ex['turns']} turns — «{ex['preview']}...»")

    # Суммарные токены
    total = sum(token_counts)
    print(f"\n📦 Всего токенов в датасете: {total:,}")
    print(f"   При 3 эпохах модель увидит: ~{total * 3:,} токенов")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "dataset.jsonl"
    analyze(path)
