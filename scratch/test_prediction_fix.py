import asyncio
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from rag.predictor import predict_ipl_winner

async def test_prediction():
    print("Testing IPL winner prediction for 2026...")
    try:
        result = await predict_ipl_winner(2026)
        print(f"SUCCESS: Predicted winner: {result.predicted_winner}")
        print(f"Confidence: {result.confidence:.0%}")
        print(f"Reasoning snippet: {result.reasoning[:100]}...")
    except Exception as e:
        print(f"FAILURE: {e}")

if __name__ == "__main__":
    asyncio.run(test_prediction())
