import asyncio, sys
sys.path.insert(0, ".")
from services.groq_client import get_groq_client

async def test():
    print("Testing MemoryService summarization path via get_groq_client()...")
    try:
        result = await get_groq_client().complete(
            messages=[{
                "role": "user",
                "content": (
                    "Summarize this conversation as bullet points.\n\n"
                    "User: Hi, my name is Arjun.\n"
                    "Assistant: Nice to meet you, Arjun!\n"
                    "User: I work in AI.\n"
                    "Assistant: That's fascinating!"
                )
            }],
            temperature=0.3,
            max_tokens=200,
        )
        print(f"\n✅ PASSED — Got summary ({len(result)} chars):")
        print(result)
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        sys.exit(1)

asyncio.run(test())