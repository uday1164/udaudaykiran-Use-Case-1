"""Utility functions for handling API rate limits and retries."""

import re
import time
from langchain_core.runnables import RunnableConfig


def is_rate_limit_error(error: Exception) -> bool:
    """Return True when an exception represents provider rate limiting."""
    error_msg = str(error).lower()
    error_str = str(error)
    return (
        "rate limit" in error_msg or      # "Rate limit reached"
        "rate_limit" in error_msg or      # "rate_limit_exceeded"
        "429" in error_str or             # HTTP 429 status
        "tokens per" in error_msg         # "tokens per day (TPD)" / "tokens per minute"
    )


def is_context_length_error(error: Exception) -> bool:
    """Return True when an exception indicates request context is too large."""
    error_msg = str(error).lower()
    return (
        "context_length_exceeded" in error_msg or
        "reduce the length of the messages" in error_msg or
        "reduce your message size" in error_msg or
        "request too large" in error_msg
    )


def is_request_too_large_error(error: Exception) -> bool:
    """Return True when the provider rejected the payload size before execution."""
    error_msg = str(error).lower()
    error_str = str(error)
    return (
        "request too large" in error_msg or
        "payload too large" in error_msg or
        ("413" in error_str and "tokens" in error_msg)
    )


def is_tool_schema_error(error: Exception) -> bool:
    """Return True when the model attempted an undeclared or invalid tool call."""
    error_msg = str(error).lower()
    return (
        "tool_use_failed" in error_msg or
        "tool call validation failed" in error_msg or
        "attempted to call tool" in error_msg or
        "failed_generation" in error_msg
    )


def estimate_text_tokens(text: str) -> int:
    """Approximate token count conservatively using a simple chars-per-token heuristic."""
    return max(1, len(text) // 4) if text else 0


def estimate_input_tokens(input_dict) -> int:
    """Approximate token count for the primary agent input payload."""
    if not isinstance(input_dict, dict):
        return estimate_text_tokens(str(input_dict))

    total = 0
    messages = input_dict.get("messages", [])
    for message in messages:
        content = getattr(message, "content", "")
        if isinstance(content, list):
            content = " ".join(str(part) for part in content)
        total += estimate_text_tokens(str(content))

    if total == 0:
        total = estimate_text_tokens(str(input_dict))
    return total


def classify_error_type(error: Exception) -> str:
    """Return a stable error class label for observability and retry policy."""
    if is_tpd_quota_error(error):
        return "tpd_hard_block"
    if is_request_too_large_error(error):
        return "request_too_large_shrink"
    if is_tpm_window_error(error):
        return "tpm_window_retry"
    if is_tool_schema_error(error):
        return "tool_schema_hard_block"
    if is_context_length_error(error):
        return "context_length_hard_block"
    if is_rate_limit_error(error):
        return "rate_limit_other"
    return "other_transient"


def extract_failed_generation(error: Exception) -> str:
    """Extract a short failed-generation snippet from provider error text when present."""
    error_str = str(error)
    match = re.search(r"failed_generation['\"]?\s*[:=]\s*['\"](.+?)['\"](?:[}\n]|$)", error_str, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1)[:500]
    marker = "failed_generation"
    if marker in error_str.lower():
        idx = error_str.lower().find(marker)
        return error_str[idx: idx + 500]
    return ""


def is_tpm_window_error(error: Exception) -> bool:
    """Return True for short-window tokens-per-minute throttling errors."""
    error_msg = str(error).lower()
    return "tokens per minute" in error_msg or "(tpm)" in error_msg


def is_tpd_quota_error(error: Exception) -> bool:
    """Return True for tokens-per-day / quota-exhausted errors that should fail fast."""
    error_msg = str(error).lower()
    return "tokens per day" in error_msg or "(tpd)" in error_msg or "quota exhausted" in error_msg


def extract_retry_after_seconds(error: Exception) -> float | None:
    """Extract retry duration from provider messages (e.g. 6.755s, 14m29.184s)."""
    message = str(error).lower()
    # Prefer explicit "try again in ..." phrases.
    m = re.search(r"try again in\s+([0-9]+(?:\.[0-9]+)?)s", message)
    if m:
        return float(m.group(1))
    m = re.search(r"try again in\s+([0-9]+)m([0-9]+(?:\.[0-9]+)?)s", message)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    return None


def is_llm_blocker_error(error: Exception) -> bool:
    """Return True when an exception indicates LLM/provider/tool-call blocking."""
    if is_rate_limit_error(error):
        return True
    error_msg = str(error).lower()
    blocker_signals = [
        "invalid_request_error",
        "badrequesterror",
        "ratelimiterror",
        "apierror",
        "chat.completions",
        "create_react_agent",
        "tool call",
        "tool_calls",
        "failed generation",
        "model produced",
        "groq",
        "gemini",
        "llm",
    ]
    return any(signal in error_msg for signal in blocker_signals)


def invoke_agent_with_retry(agent, input_dict, max_retries=1, agent_name="Agent", recursion_limit=25, max_input_tokens: int | None = None):
    """Invoke an agent with exponential backoff retry for TRANSIENT errors only.
    
    CRITICAL: Rate limit errors (quota exhausted) are NOT retried—retrying when quota is 
    exhausted wastes tokens on guaranteed-to-fail calls. Caller must catch rate limits 
    and fall back to deterministic methods immediately.
    
    Only network/timeout errors (transient) are retried.
    
    Args:
        agent: The agent to invoke
        input_dict: Input dictionary for agent.invoke()
        max_retries: Maximum number of retry attempts for transient errors (default 1)
        agent_name: Name of agent for logging
        recursion_limit: Max iterations for ReAct agent graph (one complete reasoning cycle). Default 25.
        max_input_tokens: Optional conservative ceiling for input tokens. When exceeded,
            fail fast so the caller can shrink context or use deterministic fallback.
        
    Returns:
        Agent result
        
    Raises:
        Rate limit error immediately (no retry) for caller to handle via fallback
        Other exception after max_retries attempts exhausted
    """
    try:
        import groq
        rate_limit_errors = (groq.RateLimitError,)  # Explicit: only this type
    except ImportError:
        rate_limit_errors = ()
    
    # Build config with recursion_limit to prevent runaway ReAct loops (one-shot execution)
    config = RunnableConfig(recursion_limit=recursion_limit)

    approx_input_tokens = estimate_input_tokens(input_dict)
    if max_input_tokens is not None:
        print(f"[{agent_name}] Approx input tokens={approx_input_tokens} budget={max_input_tokens}")
        if approx_input_tokens > max_input_tokens:
            raise RuntimeError(
                f"preflight_input_budget_exceeded: approx_tokens={approx_input_tokens} budget={max_input_tokens}"
            )
    
    tpm_retry_used = False
    transient_attempt = 0

    while True:
        try:
            return agent.invoke(input_dict, config=config)
        except rate_limit_errors as e:
            # Allow one bounded retry for TPM window throttling only.
            if is_tpm_window_error(e) and not is_tpd_quota_error(e) and not tpm_retry_used:
                wait_for = extract_retry_after_seconds(e)
                if wait_for is not None and wait_for <= 15:
                    tpm_retry_used = True
                    sleep_for = wait_for + 1
                    print(f"[{agent_name}] TPM window limit hit. Waiting {sleep_for:.1f}s and retrying once.")
                    time.sleep(sleep_for)
                    continue

            print(f"[{agent_name}] Rate limit hit (quota exhausted). Failing fast to preserve tokens.")
            raise
        except Exception as e:
            if is_tool_schema_error(e):
                print(f"[{agent_name}] Tool-schema blocker detected. Failing fast (no retry): {e}")
                raise

            if is_request_too_large_error(e):
                print(f"[{agent_name}] Request-too-large blocker detected. Failing fast for caller-side shrink/retry: {e}")
                raise

            if is_context_length_error(e):
                print(f"[{agent_name}] Context-length blocker detected. Failing fast (no retry): {e}")
                raise

            if is_tpm_window_error(e) and not is_tpd_quota_error(e) and not tpm_retry_used:
                wait_for = extract_retry_after_seconds(e)
                if wait_for is not None and wait_for <= 15:
                    tpm_retry_used = True
                    sleep_for = wait_for + 1
                    print(f"[{agent_name}] TPM window limit hit. Waiting {sleep_for:.1f}s and retrying once.")
                    time.sleep(sleep_for)
                    continue

            # Transient error (network, timeout, etc). Retry with exponential backoff.
            if transient_attempt < max_retries - 1:
                wait_time = (2 ** transient_attempt) + 1  # exponential backoff: 2s, 4s, 8s
                print(f"[{agent_name}] Transient error, retrying in {wait_time}s... (attempt {transient_attempt + 1}/{max_retries}): {type(e).__name__}")
                time.sleep(wait_time)
                transient_attempt += 1
            else:
                print(f"[{agent_name}] Transient error - all {max_retries} retries exhausted. Raising: {e}")
                raise
