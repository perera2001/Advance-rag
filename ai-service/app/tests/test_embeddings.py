from app.rag.embeddings import embedding_model


def main():
    vector = embedding_model.embed_query(
        "Advanced retrieval-augmented generation"
    )

    print("Embedding created successfully.")
    print(f"Vector length: {len(vector)}")


if __name__ == "__main__":
    main()