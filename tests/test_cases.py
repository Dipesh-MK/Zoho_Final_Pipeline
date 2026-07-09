"""
Two unambiguous sanity-check cases, shared across all tool audit scripts.
A correct tool should get BOTH right. If it flips either one, that's a real bug.
"""

DOCUMENT = (
    "The Eiffel Tower was completed in 1889 and stands 330 meters tall. "
    "It was designed by engineer Gustave Eiffel for the World's Fair in Paris."
)
QUERY = "When was the Eiffel Tower completed and how tall is it?"

# Case 1: clearly FAITHFUL — directly supported by the document
FAITHFUL_RESPONSE = "The Eiffel Tower was completed in 1889 and is 330 meters tall."

# Case 2: clearly HALLUCINATED — contradicts the document (wrong year, wrong height)
HALLUCINATED_RESPONSE = "The Eiffel Tower was completed in 1920 and is 500 meters tall."

CASES = [
    {"id": "SANITY_FAITHFUL", "query": QUERY, "document": DOCUMENT,
     "response": FAITHFUL_RESPONSE, "expected": "faithful"},
    {"id": "SANITY_HALLUCINATED", "query": QUERY, "document": DOCUMENT,
     "response": HALLUCINATED_RESPONSE, "expected": "hallucinated"},
]
