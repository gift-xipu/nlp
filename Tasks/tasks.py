import re
from typing import Any, Dict, List, Optional, Callable, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class LLMTask:
  
    DEFAULT_MAX_BATCHES = 10

    def __init__(self, llm_client: Any, prompts_dir: str | Path):
        
        self.llm_client = llm_client
        self.prompts_dir = Path(prompts_dir)

    def run(
        self,
        task_folder: str,
        prompt_type: str,
        placeholders: Dict[str, Any],
        target_count: int = 100,
        temperature: float = 1.0,
        max_tokens: int = 500,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        max_batches: int = DEFAULT_MAX_BATCHES,
    ) -> List[Dict[str, Any]]:
        
        prompt = self._load_and_fill_prompt(task_folder, prompt_type, placeholders)

        # loop batches until target_count or max_batches reached
        collected: List[Dict[str, Any]] = []
        batch_count = 0

        while len(collected) < target_count and batch_count < max_batches:
            batch_count += 1
            try:
                response_text = self.llm_client.generate(
                    prompt=prompt, temperature=temperature, max_tokens=max_tokens
                )
            except Exception as e:
                logger.exception("LLM generation error in batch %s: %s", batch_count, e)
                break

            batch_items = self._parse_response_by_task(response_text, task_folder, placeholders)
            if not batch_items:
                # Avoid infinite loops if model returns nothing parsable
                logger.warning("No items parsed from batch %d; stopping.", batch_count)
                break

            collected.extend(batch_items)

            if progress_callback:
                try:
                    progress_callback(min(len(collected), target_count), target_count)
                except Exception:
                    logger.exception("progress_callback raised an exception; ignoring it.")

        # trim to requested size
        return collected[:target_count]


    def _load_and_fill_prompt(self, task_folder: str, prompt_type: str, placeholders: Dict[str, Any]) -> str:
        prompt_path = self.prompts_dir / task_folder / prompt_type / "prompt.txt"
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

        raw = prompt_path.read_text(encoding="utf-8")
        placeholder_names = self._find_placeholders(raw)

        missing = [p for p in placeholder_names if p not in placeholders]
        if missing:
            raise ValueError(f"Missing placeholders for prompt: {missing}. Provided placeholders: {list(placeholders.keys())}")

        def _repl(match: re.Match) -> str:
            name = match.group(1)
            val = placeholders.get(name, "")
            return str(val)

        filled = re.sub(r"\{(\w+)\}", _repl, raw)
        return filled

    @staticmethod
    def _find_placeholders(text: str) -> List[str]:
        return list(dict.fromkeys(re.findall(r"\{(\w+)\}", text)))  # unique, preserve order

    def _parse_response_by_task(self, response: str, task_folder: str, placeholders: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Decide which parser to use based on task_folder name.
        """
        key = task_folder.lower()
        if "lexicon_generating" in key or "lexicon-generating" in key:
            return self._parse_lexicon_generating(response, placeholders)
        if "lexicon_bootstrapping" in key or "lexicon-bootstrapping" in key:
            return self._parse_bootstrapping(response, placeholders)
        if "sentiment_bearings" in key or "sentiment-bearings" in key:
            return self._parse_sentiment_bearings(response, placeholders)

        return self._parse_generic_key_value(response, placeholders)

    def _standard_line_clean(self, line: str) -> str:
        # strip bullets, numbers, whitespace
        line = line.strip()
        if not line:
            return ""
        line = re.sub(r'^\s*[-*•\u2022]\s*', '', line)  # bullets
        line = re.sub(r'^\d+\.?\s*', '', line)  # leading numbers like "1. "
        return line.strip()

    def _split_key_val(self, line: str) -> Optional[Tuple[str, str]]:
        # Try several separators, prefer ":" then " - " then tab then " — " etc.
        separators = [":", " - ", " — ", "–", "\t"]
        for sep in separators:
            if sep in line:
                left, right = line.split(sep, 1)
                return left.strip().strip('*_`"\' '), right.strip().strip('*_`"\' ')
        # if nothing found but line contains single word, treat as key with empty val
        if line:
            return line.strip().strip('*_`"\' '), ""
        return None

    def _parse_lexicon_generating(self, response: str, placeholders: Dict[str, Any]) -> List[Dict[str, Any]]:
    
        items = []
        for raw_line in response.splitlines():
            line = self._standard_line_clean(raw_line)
            if not line or line.upper() == "DONE" or line.startswith("#"):
                continue

            kv = self._split_key_val(line)
            if not kv:
                continue
            word, translation = kv
            if not translation:
                # sometimes model returns "word - translation / extra" or just "word"; skip if no translation
                # but still accept if translation missing (user might want single-token lexicons)
                pass

            items.append({
                "word": word,
                "translation": translation,
                "language": placeholders.get("language"),
                "sentiment": placeholders.get("sentiment") or placeholders.get("sentiment_type"),
            })
        return items

    def _parse_bootstrapping(self, response: str, placeholders: Dict[str, Any]) -> List[Dict[str, Any]]:
      
        items = []
        for raw_line in response.splitlines():
            line = self._standard_line_clean(raw_line)
            if not line or line.upper() == "DONE" or line.startswith("#"):
                continue
            kv = self._split_key_val(line)
            if not kv:
                continue
            item_key, desc = kv
            items.append({
                "item": item_key,
                "description": desc,
                "language": placeholders.get("language"),
                "target_type": placeholders.get("target_type"),
            })
        return items

    def _parse_sentiment_bearings(self, response: str, placeholders: Dict[str, Any]) -> List[Dict[str, Any]]:
       
        items = []
        for raw_line in response.splitlines():
            line = self._standard_line_clean(raw_line)
            if not line or line.upper() == "DONE" or line.startswith("#"):
                continue
            kv = self._split_key_val(line)
            if not kv:
                continue
            phrase, bearing = kv
            items.append({
                "phrase": phrase,
                "bearing": bearing,
                "language": placeholders.get("language"),
                "sentiment_type": placeholders.get("sentiment_type") or placeholders.get("sentiment"),
            })
        return items

    def _parse_generic_key_value(self, response: str, placeholders: Dict[str, Any]) -> List[Dict[str, Any]]:
       
        items = []
        for raw_line in response.splitlines():
            line = self._standard_line_clean(raw_line)
            if not line or line.upper() == "DONE" or line.startswith("#"):
                continue
            kv = self._split_key_val(line)
            if not kv:
                continue
            left, right = kv
            items.append({
                "key": left,
                "value": right,
                **{k: v for k, v in placeholders.items() if k in ("language", "target_type", "sentiment", "sentiment_type")}
            })
        return items


