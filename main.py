import json
import re

# ==========================================
# ⚙️ НАСТРОЙКИ
# ==========================================

INPUT_FILE = "base.txt"
OUTPUT_FILE_MACHINE = "dataset.jsonl"  # Для обучения (строгий формат)
OUTPUT_FILE_HUMAN = "dataset.json"  # Для тебя (красивый формат)


SYSTEM_PROMPT = "I'm Leo"

# ==========================================
# 🛠️ ПАРСЕР
# ==========================================

def parse_dialogue(block):
    lines = block.strip().split('\n')
    messages = []
    messages.append({"role": "system", "content": SYSTEM_PROMPT.strip()})

    current_role = None
    current_content = []

    def flush_message():
        nonlocal current_role, current_content
        if current_role and current_content:
            text = '\n'.join(current_content).strip()
            if text:
                messages.append({"role": current_role, "content": text})
        current_content = []

    for line in lines:
        if line.strip() == '###': continue

        user_match = re.match(r'^User:\s*(.*)', line, re.IGNORECASE)
        assistant_match = re.match(r'^Assistant:\s*(.*)', line, re.IGNORECASE)

        if user_match:
            flush_message()
            current_role = "user"
            content_start = user_match.group(1)
            if content_start: current_content.append(content_start)
        elif assistant_match:
            flush_message()
            current_role = "assistant"
            content_start = assistant_match.group(1)
            if content_start: current_content.append(content_start)
        else:
            if current_role: current_content.append(line)

    flush_message()
    return messages


def main():
    print(f"❤️  читает: {INPUT_FILE}...")

    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            full_text = f.read()
    except FileNotFoundError:
        print("💔 Файл не найден!")
        return

    raw_dialogues = filter(None, full_text.split('==='))

    # Список для красивого файла (чтобы обернуть всё в квадратные скобки [])
    all_conversations_for_human = []

    count = 0
    # Открываем файл для машины (.jsonl)
    with open(OUTPUT_FILE_MACHINE, 'w', encoding='utf-8') as f_machine:
        for block in raw_dialogues:
            if not block.strip(): continue
            conversation = parse_dialogue(block)

            if len(conversation) >= 3:
                json_entry = {"messages": conversation}

                # 1. Пишем для МАШИНЫ (одна строка, без отступов)
                f_machine.write(json.dumps(json_entry, ensure_ascii=False) + '\n')

                # 2. Сохраняем для ТЕБЯ в память
                all_conversations_for_human.append(json_entry)

                count += 1

    # Записываем файл для ТЕБЯ (красивый JSON с отступами)
    with open(OUTPUT_FILE_HUMAN, 'w', encoding='utf-8') as f_human:
        # indent=4 делает красоту и переносы строк
        json.dump(all_conversations_for_human, f_human, indent=4, ensure_ascii=False)

    print(f"✨ Готово! Обработано {count} диалогов.")
    print(f"🤖 Для обучения: {OUTPUT_FILE_MACHINE} (НЕ ОТКРЫВАЙ, ОН СТРАШНЫЙ)")
    print(f"👀 Для чтения:   {OUTPUT_FILE_HUMAN} (КРАСИВЫЙ)")


if __name__ == "__main__":
    main()
