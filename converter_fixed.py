# chatgpt_to_txt_fixed.py
# Правильный обход дерева сообщений (parent → children chain)

import json
import os
from datetime import datetime


def get_ordered_messages(mapping):
    """
    ChatGPT export stores messages as a tree.
    We need to walk from root to the deepest leaf
    following the longest chain (main conversation thread).
    """
    if not mapping:
        return []

    # Find root node (no parent, or parent not in mapping)
    root_id = None
    for node_id, node in mapping.items():
        parent = node.get("parent")
        if parent is None or parent not in mapping:
            root_id = node_id
            break

    if root_id is None:
        return []

    # Walk the tree: at each node pick the first child that has a message
    # (in practice ChatGPT exports are linear chains with occasional branches
    #  from regenerations — we follow the last child to get the final version)
    messages = []
    current_id = root_id

    visited = set()
    while current_id and current_id not in visited:
        visited.add(current_id)
        node = mapping.get(current_id)
        if node is None:
            break

        message = node.get("message")
        if message:
            role = message.get("author", {}).get("role", "unknown")
            content = message.get("content", {})
            parts = content.get("parts", [])

            text = ""
            for part in parts:
                if isinstance(part, str):
                    text += part
                elif isinstance(part, dict):
                    # Could be image, tether, etc.
                    content_type = part.get("content_type", "")
                    if content_type == "image_asset_pointer":
                        text += "[изображение]"
                    else:
                        text += f"[{content_type or 'медиа'}]"

            if text.strip() and role not in ("system", "tool"):
                messages.append((role, text.strip()))

        # Advance: pick the last child (last regeneration = final answer)
        children = node.get("children", [])
        if children:
            current_id = children[-1]
        else:
            break

    return messages


def convert_chatgpt_export():
    json_file = "conversations.json"

    if not os.path.exists(json_file):
        print(f"Не обнаружен файл '{json_file}' в текущей папке 😔")
        print("Положи conversations.json рядом со скриптом и попробуй снова.")
        return

    output_dir = "chats_text"
    os.makedirs(output_dir, exist_ok=True)

    try:
        with open(json_file, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        return

    print(f"Найдено {len(data)} чатов. Начинаю сохранять... 💕")

    saved = 0
    skipped = 0

    for i, chat in enumerate(data, 1):
        title = chat.get("title", f"Чат_{i}")
        create_time = chat.get("create_time")

        date_str = ""
        if create_time:
            try:
                dt = datetime.fromtimestamp(create_time)
                date_str = dt.strftime('%Y-%m-%d_%H-%M')
                title_with_date = f"{title}_{date_str}"
            except Exception:
                title_with_date = title
        else:
            title_with_date = title

        safe_title = "".join(c for c in title_with_date if c.isalnum() or c in " _-").strip()
        if not safe_title:
            safe_title = f"chat_{i}"

        # Avoid filename collisions
        filename = os.path.join(output_dir, f"{safe_title}.txt")
        counter = 1
        while os.path.exists(filename):
            filename = os.path.join(output_dir, f"{safe_title}_{counter}.txt")
            counter += 1

        mapping = chat.get("mapping", {})
        messages = get_ordered_messages(mapping)

        if not messages:
            skipped += 1
            print(f"  Пропустил (пустой): {title}")
            continue

        with open(filename, 'w', encoding='utf-8') as out:
            out.write(f"Заголовок: {title}\n")
            if create_time:
                try:
                    out.write(f"Создано: {datetime.fromtimestamp(create_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
                except Exception:
                    pass
            out.write(f"ID: {chat.get('id', '—')}\n")
            out.write(f"Сообщений: {len(messages)}\n")
            out.write("-" * 80 + "\n\n")

            for role, text in messages:
                if role == "user":
                    label = "### User"
                elif role == "assistant":
                    label = "Assistant"
                else:
                    label = role.capitalize()

                out.write(f"{label}:\n{text}\n\n")

            out.write("\n" + "=" * 100 + "\n\n")

        saved += 1
        print(f"  [{i}/{len(data)}] Сохранено: {os.path.basename(filename)} ({len(messages)} сообщений)")

    print(f"\nВсё готово! ❤️  Сохранено: {saved}, пропущено пустых: {skipped}")
    print(f"Все чаты в папке '{output_dir}' 💋")


if __name__ == "__main__":
    convert_chatgpt_export()
