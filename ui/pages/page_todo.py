"""
ui/pages/page_todo.py  –  Tools › TODO list (persists across sessions)

Storage: a JSON file (default: <project_root>/data/todo.json).
Override the location with the env var EQUITYDESK_TODO_FILE.
"""

import json
import os
import re
import uuid
from datetime import date, datetime
from pathlib import Path

import streamlit as st

try:
    from streamlit_quill import st_quill   # pip install streamlit-quill
except ImportError:                        # fallback: plain text area
    st_quill = None

DATA_FILE = Path(
    os.environ.get(
        "EQUITYDESK_TODO_FILE",
        Path(__file__).resolve().parents[2] / "data" / "todo.json",
        )
)

PRIORITIES = ["High", "Medium", "Low"]
PRIO_ICON = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
PRIO_ORDER = {p: i for i, p in enumerate(PRIORITIES)}

QUILL_TOOLBAR = [
    ["bold", "italic", "underline", "strike"],
    [{"header": [1, 2, 3, False]}],
    [{"list": "ordered"}, {"list": "bullet"}],
    ["blockquote", "code-block", "link"],
    [{"color": []}, {"background": []}],
    ["clean"],
]


def _is_empty_html(html: str | None) -> bool:
    """Quill returns '<p><br></p>' for an empty editor."""
    if not html:
        return True
    if "<img" in html:
        return False
    text = re.sub(r"<[^>]+>", "", html).replace("&nbsp;", "").strip()
    return text == ""


# ── Storage ───────────────────────────────────────────────────────────────────
def _load() -> list[dict]:
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save(items: list[dict]) -> None:
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = DATA_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)
    os.replace(tmp, DATA_FILE)  # atomic: no half-written file on crash


# ── Callbacks (run before the rerun, so UI is always fresh) ──────────────────
def _add(text: str, priority: str, due: date | None, note: str) -> None:
    items = _load()
    items.append({
        "id": uuid.uuid4().hex[:8],
        "text": text.strip(),
        "priority": priority,
        "due": due.isoformat() if due else None,
        "note": note.strip(),
        "done": False,
        "created": datetime.now().isoformat(timespec="seconds"),
        "completed": None,
    })
    _save(items)


def _toggle(item_id: str) -> None:
    items = _load()
    for it in items:
        if it["id"] == item_id:
            it["done"] = not it["done"]
            it["completed"] = (
                datetime.now().isoformat(timespec="seconds") if it["done"] else None
            )
    _save(items)


def _delete(item_id: str) -> None:
    _save([it for it in _load() if it["id"] != item_id])


def _clear_done() -> None:
    _save([it for it in _load() if not it["done"]])


def _sort_key(it: dict):
    return (
        it["done"],
        it.get("due") or "9999-12-31",
        PRIO_ORDER.get(it.get("priority"), 1),
    )


# ── Page ──────────────────────────────────────────────────────────────────────
def render():
    st.header("✅ TODO List")
    items = _load()

    open_n = sum(not i["done"] for i in items)
    done_n = len(items) - open_n
    today = date.today().isoformat()
    overdue_n = sum(
        1 for i in items
        if not i["done"] and i.get("due") and i["due"] < today
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Open", open_n)
    m2.metric("Done", done_n)
    m3.metric("Overdue", overdue_n)

    # ── Add form ──────────────────────────────────────────────────────────────
    # Custom components aren't reset by clear_on_submit, so we bump a counter
    # to give the editor a fresh key (= empty editor) after each successful add.
    st.session_state.setdefault("todo_note_ver", 0)

    with st.expander("➕ Add task", expanded=False):
        with st.form("add_todo_form", clear_on_submit=True):
            text = st.text_input("New task", placeholder="e.g. Review INFY DCF assumptions")
            c1, c2 = st.columns([1, 1])
            priority = c1.selectbox("Priority", PRIORITIES, index=1)
            has_due = c2.checkbox("Set due date")
            due = c2.date_input("Due", value=date.today(), label_visibility="collapsed")

            st.caption("Note (optional)")
            note_key = f"todo_note_{st.session_state['todo_note_ver']}"
            if st_quill is not None:
                note = st_quill(
                    placeholder="Add details, links, checklists…",
                    html=True,
                    toolbar=QUILL_TOOLBAR,
                    key=note_key,
                )
            else:
                note = st.text_area(
                    "Note", label_visibility="collapsed", key=note_key,
                    placeholder="Markdown supported. (pip install streamlit-quill for a rich editor)",
                )

            if st.form_submit_button("➕ Add", type="primary"):
                if text.strip():
                    _add(text, priority, due if has_due else None,
                         "" if _is_empty_html(note) else note)
                    st.session_state["todo_note_ver"] += 1
                    st.rerun()
                else:
                    st.warning("Enter a task description.")

    st.divider()

    # ── Filters ───────────────────────────────────────────────────────────────
    f1, f2 = st.columns([1, 1])
    view = f1.radio("Show", ["Open", "Done", "All"], horizontal=True)
    prio_filter = f2.multiselect("Priority filter", PRIORITIES, default=PRIORITIES)

    shown = [
        i for i in sorted(items, key=_sort_key)
        if i.get("priority", "Medium") in prio_filter
           and (view == "All" or (view == "Done") == i["done"])
    ]

    if not shown:
        st.info("Nothing here. 🎉" if view == "Open" else "No matching tasks.")

    # ── List ──────────────────────────────────────────────────────────────────
    for it in shown:
        c_chk, c_txt, c_del = st.columns([0.5, 8, 1])
        c_chk.checkbox(
            "done", value=it["done"], key=f"chk_{it['id']}",
            label_visibility="collapsed",
            on_change=_toggle, args=(it["id"],),
        )

        icon = PRIO_ICON.get(it.get("priority"), "🟡")
        label = f"~~{it['text']}~~" if it["done"] else f"**{it['text']}**"
        meta = []
        if it.get("due"):
            is_overdue = (not it["done"]) and it["due"] < today
            meta.append(f"{'⚠️ overdue · ' if is_overdue else '📅 '}{it['due']}")
        c_txt.markdown(
            f"{icon} {label}" + (f"  \n<small>{' · '.join(meta)}</small>" if meta else ""),
            unsafe_allow_html=True,
            )

        c_del.button(
            "🗑", key=f"del_{it['id']}", help="Delete",
            on_click=_delete, args=(it["id"],),
        )

        if not _is_empty_html(it.get("note")):
            with st.expander("📝 Note", expanded=False):
                # Notes are HTML from Quill (or plain/markdown text from older
                # entries). This is your own local data, so rendering is fine.
                st.markdown(it["note"], unsafe_allow_html=True)

    # ── Footer actions ────────────────────────────────────────────────────────
    if items:
        st.divider()
        b1, b2, _ = st.columns([1.5, 1.5, 4])
        if done_n:
            b1.button(f"🧹 Clear {done_n} done", on_click=_clear_done)
        b2.download_button(
            "⬇ Export JSON",
            data=json.dumps(items, indent=2, ensure_ascii=False),
            file_name="todo.json",
            mime="application/json",
        )
    st.caption(f"Saved to `{DATA_FILE}`")