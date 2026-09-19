from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        if not question or not question.strip():
            return "Vui lòng cung cấp câu hỏi."

        if self.store.get_collection_size() == 0:
            return "Không tìm thấy tài liệu phù hợp trong cơ sở tri thức."

        results = self.store.search(question, top_k=top_k)
        if not results:
            return "Không tìm thấy thông tin liên quan đến câu hỏi trong cơ sở tri thức."

        context_blocks = []
        for i, r in enumerate(results, start=1):
            source_info = r.get("metadata", {}).get("source", r.get("id", f"doc_{i}"))
            context_blocks.append(f"[{i}] Nguồn: {source_info}\n{r['content']}")
        context_str = "\n\n".join(context_blocks)

        prompt = (
            "Bạn là trợ lý AI trả lời câu hỏi dựa trên thông tin được cung cấp từ cơ sở tri thức.\n"
            "Chỉ sử dụng thông tin trong các đoạn trích dẫn dưới đây để trả lời. "
            "Nếu thông tin không đủ để trả lời câu hỏi, hãy nói rõ 'Tôi không tìm thấy thông tin này trong tài liệu'.\n"
            "Khi trả lời, hãy trích dẫn số thứ tự nguồn [1], [2] tương ứng.\n\n"
            f"Ngữ cảnh:\n{context_str}\n\n"
            f"Câu hỏi: {question}\n\n"
            "Trả lời:"
        )

        return self.llm_fn(prompt)
