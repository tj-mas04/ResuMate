from __future__ import annotations
import os
from pathlib import Path
import streamlit as st
import language_tool_python
from . import nlp_service  # ensure Streamlit caches are initialized
from config import get_languagetool_cache_dir


@st.cache_resource
def load_grammar_tool():
    cache_dir: Path = get_languagetool_cache_dir()
    os.environ["LANGUAGE_TOOL_CACHE_DIR"] = str(cache_dir)

    try:
        tool = language_tool_python.LanguageTool("en-US")
        tool._available = True  # type: ignore[attr-defined]
        return tool
    except SystemError as se:  # likely Java version issue
        msg = f"LanguageTool init failed: {se}. Grammar checks disabled."
        print(msg)

        class DummyGrammarTool:
            def __init__(self, message: str):
                self._message = message
                self._available = False

            def check(self, text: str):
                return []

        return DummyGrammarTool(msg)
    except Exception as e:
        msg = f"LanguageTool error: {e}. Grammar checks disabled."
        print(msg)

        class DummyGrammarTool2:
            def __init__(self, message: str):
                self._message = message
                self._available = False

            def check(self, text: str):
                return []

        return DummyGrammarTool2(msg)


grammar_tool = load_grammar_tool()


def check_grammar(text: str):
    matches = grammar_tool.check(text)
    return len(matches), [{"Error": m.message, "Sentence": m.context} for m in matches]
