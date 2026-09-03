from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)


contextualize_question_prompt = (
    ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
Rewrite the latest user question as a standalone question
using the conversation history.

Do not answer the question.
Preserve the original meaning.
If the question is already standalone, return it unchanged.
""",
            ),
            MessagesPlaceholder("chat_history"),
            (
                "human",
                "{question}",
            ),
        ]
    )
)


answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a research-paper question-answering assistant.

Answer using only the supplied research-paper context.

Rules:
1. Do not use outside knowledge.
2. Do not invent facts.
3. If the context does not contain enough information, say:
   "I could not find enough information in the selected documents."
4. Give a clear and concise answer.
5. Do not mention retrieved chunk numbers.
6. The application will display source citations separately.

Context:
{context}
""",
        ),
        MessagesPlaceholder("chat_history"),
        (
            "human",
            "{question}",
        ),
    ]
)
