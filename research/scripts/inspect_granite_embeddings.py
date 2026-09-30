"""Diagnostic only: load ibm-granite/granite-embedding-small-english-r2 and sanity-check it.

Not wired into chunks.json, retrieval, evaluation, or any production code.
Loads the model by its Hugging Face model ID (letting HF download/cache it
normally, into the default HF cache - NOT into this repo), embeds a few
employee-policy test sentences, and prints embedding shape + a cosine
similarity matrix so semantically related sentences can be checked against
unrelated ones.

Run from the project root:
    venv/bin/python scripts/inspect_granite_embeddings.py
"""

from sentence_transformers import SentenceTransformer  # official recommended usage for this model card

_MODEL_ID = "ibm-granite/granite-embedding-small-english-r2"

# Employee-policy sentences: 0-1 and 2-3 are each a semantically related
# pair (same topic, different phrasing); 4 is unrelated to all of them.
_SENTENCES = [
    "How do I submit an expense claim?",
    "What is the process for claiming travel expenses?",
    "How many days of annual leave am I entitled to?",
    "What is the policy for carrying over unused holiday?",
    "How do I report a lost or stolen company laptop?",
]


def main() -> None:
    model = SentenceTransformer(_MODEL_ID)  # HF model ID -> normal download/cache flow, no local file path

    embeddings = model.encode(_SENTENCES)  # default: unnormalized vectors, per the model card
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Max sequence length: {model.max_seq_length}")
    print()

    similarities = model.similarity(embeddings, embeddings)  # sentence-transformers cosine similarity helper

    print("Cosine similarity matrix:")
    header = "".join(f"  s{i}" for i in range(len(_SENTENCES)))
    print("      " + header)
    for i, row in enumerate(similarities):
        row_str = "".join(f" {value:.3f}" for value in row.tolist())
        print(f"  s{i}:{row_str}")
    print()

    for i, sentence in enumerate(_SENTENCES):
        print(f"s{i}: {sentence}")
    print()

    print("Related-pair vs unrelated-pair check:")
    print(f"  s0 vs s1 (both about expenses):        {similarities[0][1]:.3f}")
    print(f"  s2 vs s3 (both about annual leave):     {similarities[2][3]:.3f}")
    print(f"  s0 vs s4 (expenses vs lost device):     {similarities[0][4]:.3f}")
    print(f"  s2 vs s4 (leave vs lost device):        {similarities[2][4]:.3f}")
    print(f"  s0 vs s2 (expenses vs leave, unrelated): {similarities[0][2]:.3f}")


if __name__ == "__main__":
    main()
