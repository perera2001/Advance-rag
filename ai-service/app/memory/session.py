from langchain_core.chat_history import (
    InMemoryChatMessageHistory,
)


session_store: dict[
    str,
    InMemoryChatMessageHistory,
] = {}


def get_session_history(
    session_id: str,
) -> InMemoryChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = (
            InMemoryChatMessageHistory()
        )

    return session_store[session_id]


def add_conversation(
    session_id: str,
    question: str,
    answer: str,
):
    history = get_session_history(session_id)

    history.add_user_message(question)
    history.add_ai_message(answer)

    if len(history.messages) > 20:
        history.messages = history.messages[-20:]


def clear_session(session_id: str) -> bool:
    if session_id not in session_store:
        return False

    del session_store[session_id]
    return True
