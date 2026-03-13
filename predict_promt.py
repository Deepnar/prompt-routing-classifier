import joblib

encoder = joblib.load("models/encoder.pkl")

topic_model = joblib.load("models/topic_model.pkl")
intent_model = joblib.load("models/intent_model.pkl")

topic_binarizer = joblib.load("models/topic_binarizer.pkl")
intent_binarizer = joblib.load("models/intent_binarizer.pkl")

def predict_prompt(prompt):

    if len(prompt.split()) <= 2 and len(prompt) < 15:
        return [("general", 1.0)], [("information_request", 1.0)]

    embedding = encoder.encode([prompt])

    topic_probs = topic_model.predict_proba(embedding)[0]
    intent_probs = intent_model.predict_proba(embedding)[0]

    topic_labels = []
    intent_labels = []

    for i, p in enumerate(topic_probs):
        if p > 0.4:
            topic_labels.append((topic_binarizer.classes_[i], float(p)))

    for i, p in enumerate(intent_probs):
        if p > 0.4:
            intent_labels.append((intent_binarizer.classes_[i], float(p)))

    if not topic_labels:
        best = topic_probs.argmax()
        topic_labels = [(topic_binarizer.classes_[best], float(topic_probs[best]))]

    if not intent_labels:
        best = intent_probs.argmax()
        intent_labels = [(intent_binarizer.classes_[best], float(intent_probs[best]))]

    return topic_labels, intent_labels

while True:

    user_input = input("Enter a prompt (or 'exit' to quit): ")

    if user_input.lower() == "exit":
        break

    topics, intents = predict_prompt(user_input)

    print("\nTopics:")
    for t, p in topics:
        print(f"{t} ({p:.2f})")

    print("\nIntents:")
    for i, p in intents:
        print(f"{i} ({p:.2f})")

    print()