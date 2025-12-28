"""Multi-hop question generation using GPT-5.2 Responses API."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, asdict
from typing import List, Optional

from openai import OpenAI

from .chunk_pair_finder import ChunkPair


@dataclass
class MultiHopQuestion:
    """A multi-hop question requiring two chunks to answer."""

    question: str
    answer: str
    doc_id: str
    chunk_a_pos: int
    chunk_b_pos: int
    chunk_a_text: str
    chunk_b_text: str
    reasoning: str = ""
    chunk_a_contribution: str = ""
    chunk_b_contribution: str = ""
    difficulty: str = "medium"
    validation_passed: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "MultiHopQuestion":
        return cls(**data)

    @property
    def target_chunks(self) -> List[int]:
        """Return the target chunk positions."""
        return [self.chunk_a_pos, self.chunk_b_pos]


def _extract_keywords(text: str) -> List[str]:
    """Extract significant keywords to avoid in questions."""
    # Extract capitalized words and significant terms
    words = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)

    # Also extract numbers with context
    numbers = re.findall(r'\b(\d{4})\b', text)  # Years
    words.extend(numbers)

    # Filter and deduplicate
    stopwords = {'The', 'This', 'That', 'However', 'Although', 'Because'}
    return list(set(w for w in words if w not in stopwords and len(str(w)) > 2))


def generate_question_for_pair(
    pair: ChunkPair,
    model: str = "gpt-5.2",
    validate: bool = True,
) -> Optional[MultiHopQuestion]:
    """Generate a multi-hop question for a chunk pair.

    Parameters
    ----------
    pair : ChunkPair
        The chunk pair to generate a question for
    model : str
        Model to use for generation
    validate : bool
        Whether to validate the question

    Returns
    -------
    MultiHopQuestion or None
        Generated question, or None if generation/validation failed
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")

    client = OpenAI()

    # Extract keywords to avoid
    keywords_a = _extract_keywords(pair.chunk_a_text)
    keywords_b = _extract_keywords(pair.chunk_b_text)
    all_keywords = list(set(keywords_a + keywords_b))[:20]

    generation_prompt = f"""You are creating a multi-hop reasoning question for RAG evaluation.

DOCUMENT: {pair.doc_id}
CONNECTING CONCEPT: {pair.connecting_entity}

CHUNK A (position {pair.chunk_a_pos}):
{pair.chunk_a_text}

CHUNK B (position {pair.chunk_b_pos}):
{pair.chunk_b_text}

Generate a question that:
1. REQUIRES information from BOTH chunks to answer correctly
2. Uses SYNONYMS and PARAPHRASING - AVOID these exact words: {', '.join(all_keywords[:15])}
3. Cannot be answered by either chunk alone
4. Has a clear, factual answer (not opinion)
5. Is challenging but not impossible

Return ONLY valid JSON (no markdown, no explanation):
{{
  "question": "<paraphrased question using synonyms>",
  "answer": "<factual answer>",
  "reasoning": "<why both chunks are needed>",
  "chunk_a_contribution": "<what specific info comes from chunk A>",
  "chunk_b_contribution": "<what specific info comes from chunk B>",
  "difficulty": "easy|medium|hard"
}}"""

    try:
        # Retry up to 3 times for API failures
        result_text = ""
        for attempt in range(3):
            response = client.responses.create(
                model=model,
                input=generation_prompt,
                reasoning={"effort": "high"},
                max_output_tokens=1024,
            )
            result_text = response.output_text.strip()
            if result_text:
                break
            import time
            time.sleep(1)  # Brief pause before retry

        if not result_text:
            print(f"Empty response after 3 attempts")
            return None

        # Extract JSON
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0]

        # Try to find JSON object
        json_match = re.search(r'\{[^{}]*\}', result_text, re.DOTALL)
        if json_match:
            result_text = json_match.group()

        data = json.loads(result_text)

        question = MultiHopQuestion(
            question=data.get("question", ""),
            answer=data.get("answer", ""),
            doc_id=pair.doc_id,
            chunk_a_pos=pair.chunk_a_pos,
            chunk_b_pos=pair.chunk_b_pos,
            chunk_a_text=pair.chunk_a_text,
            chunk_b_text=pair.chunk_b_text,
            reasoning=data.get("reasoning", ""),
            chunk_a_contribution=data.get("chunk_a_contribution", ""),
            chunk_b_contribution=data.get("chunk_b_contribution", ""),
            difficulty=data.get("difficulty", "medium"),
        )

        # Validate if requested
        if validate:
            is_valid = validate_question(question, model=model)
            question.validation_passed = is_valid
            if not is_valid:
                return None

        return question

    except json.JSONDecodeError as e:
        print(f"JSON parse error in question generation: {e}")
        return None
    except Exception as e:
        print(f"Question generation error: {e}")
        return None


def validate_question(
    question: MultiHopQuestion,
    model: str = "gpt-5.2",
) -> bool:
    """Validate that a question truly requires both chunks.

    Parameters
    ----------
    question : MultiHopQuestion
        Question to validate
    model : str
        Model to use for validation

    Returns
    -------
    bool
        True if question passes validation
    """
    if not os.getenv("OPENAI_API_KEY"):
        return True  # Skip validation if no API key

    client = OpenAI()

    validation_prompt = f"""Validate this multi-hop question for RAG evaluation.

QUESTION: {question.question}
EXPECTED ANSWER: {question.answer}

CHUNK A (position {question.chunk_a_pos}):
{question.chunk_a_text}

CHUNK B (position {question.chunk_b_pos}):
{question.chunk_b_text}

Evaluate strictly:
1. Can this question be FULLY answered using ONLY Chunk A? (yes/no)
2. Can this question be FULLY answered using ONLY Chunk B? (yes/no)
3. Can this question be answered using BOTH chunks together? (yes/no)
4. Is the expected answer factually supported by the chunks? (yes/no)

Return ONLY valid JSON:
{{
  "answerable_from_a_only": true/false,
  "answerable_from_b_only": true/false,
  "answerable_from_both": true/false,
  "answer_is_supported": true/false,
  "validation_passed": true/false,
  "reason": "<brief explanation if failed>"
}}"""

    try:
        response = client.responses.create(
            model=model,
            input=validation_prompt,
            reasoning={"effort": "medium"},
            max_output_tokens=512,
        )

        result_text = response.output_text.strip()

        # Extract JSON
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0]

        json_match = re.search(r'\{[^{}]*\}', result_text, re.DOTALL)
        if json_match:
            result_text = json_match.group()

        data = json.loads(result_text)

        # Question is valid if:
        # - NOT answerable from A only
        # - NOT answerable from B only
        # - IS answerable from both
        # - Answer is supported
        is_valid = (
            not data.get("answerable_from_a_only", True) and
            not data.get("answerable_from_b_only", True) and
            data.get("answerable_from_both", False) and
            data.get("answer_is_supported", False)
        )

        if not is_valid:
            reason = data.get("reason", "Validation criteria not met")
            print(f"  Validation failed: {reason}")

        return is_valid

    except Exception as e:
        print(f"Validation error: {e}")
        return True  # Pass by default on error


def generate_multihop_questions(
    chunk_pairs: List[ChunkPair],
    model: str = "gpt-5.2",
    validate: bool = True,
    max_questions: Optional[int] = None,
) -> List[MultiHopQuestion]:
    """Generate multi-hop questions for a list of chunk pairs.

    Parameters
    ----------
    chunk_pairs : List[ChunkPair]
        Chunk pairs to generate questions for
    model : str
        Model to use for generation
    validate : bool
        Whether to validate questions
    max_questions : int, optional
        Maximum number of questions to generate

    Returns
    -------
    List[MultiHopQuestion]
        Generated (and optionally validated) questions
    """
    questions = []
    attempts = 0
    max_attempts = len(chunk_pairs)

    if max_questions:
        max_attempts = min(max_attempts, max_questions * 2)  # Allow for failures

    for pair in chunk_pairs[:max_attempts]:
        print(f"Generating question for chunks [{pair.chunk_a_pos}] <-> [{pair.chunk_b_pos}]...")

        question = generate_question_for_pair(pair, model=model, validate=validate)

        if question:
            questions.append(question)
            print(f"  ✓ Generated: {question.question[:60]}...")

            if max_questions and len(questions) >= max_questions:
                break
        else:
            print(f"  ✗ Failed or invalid")

        attempts += 1

    print(f"\nGenerated {len(questions)} valid questions from {attempts} attempts")
    return questions


def save_questions(questions: List[MultiHopQuestion], filepath: str) -> None:
    """Save questions to JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump([q.to_dict() for q in questions], f, indent=2)
    print(f"Saved {len(questions)} questions to {filepath}")


def load_questions(filepath: str) -> List[MultiHopQuestion]:
    """Load questions from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [MultiHopQuestion.from_dict(q) for q in data]


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    # Test with sample pair
    from .chunk_pair_finder import ChunkPair

    sample_pair = ChunkPair(
        doc_id="Einstein",
        chunk_a_pos=0,
        chunk_b_pos=4,
        chunk_a_text="Albert Einstein was born in Ulm, Germany in 1879. He showed early aptitude for mathematics.",
        chunk_b_text="In 1905, Einstein published four groundbreaking papers while working at the Swiss Patent Office. This year became known as his 'miracle year' or Annus Mirabilis.",
        connecting_entity="Einstein",
        relationship="early_life_to_achievement",
    )

    print("Testing question generation...")
    question = generate_question_for_pair(sample_pair, validate=True)

    if question:
        print(f"\nQuestion: {question.question}")
        print(f"Answer: {question.answer}")
        print(f"Difficulty: {question.difficulty}")
        print(f"Validation: {'Passed' if question.validation_passed else 'Failed'}")
