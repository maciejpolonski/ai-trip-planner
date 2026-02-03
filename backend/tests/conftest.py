"""
Pytest configuration. Set TEST_MODE before main is imported so the app uses the fake LLM.
"""
import os
import sys

# Ensure backend root is on path when running tests
_backend = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _backend not in sys.path:
    sys.path.insert(0, _backend)

# Must set before any import from main so _init_llm() uses fake LLM
os.environ["TEST_MODE"] = "1"
# Enable RAG so LocalGuideRetriever runs (keyword fallback when TEST_MODE, no embeddings)
os.environ["ENABLE_RAG"] = "1"
