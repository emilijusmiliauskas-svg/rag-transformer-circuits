EVALUATION_DATA: list[dict[str, str]] = [
    {
        "question": "What are induction heads and what role do they play "
                    "in in-context learning?",

        "ground_truth": "Induction heads are a type of attention head found "
                        "in transformer models that implement a simple "
                        "pattern-matching algorithm: they look back over "
                        "the context for previous instances of the current "
                        "token and attend to the token that followed it, "
                        "copying that token as the prediction. They are "
                        "believed to be a key mechanism driving in-context "
                        "learning in large language models."
    },
    {
        "question": "What is superposition in neural networks and why does "
                    "it matter for interpretability?",

        "ground_truth": "Superposition is the phenomenon where a neural "
                        "network represents more features than it has "
                        "dimensions by encoding features as near-orthogonal "
                        "directions in activation space. This means a "
                        "single neuron can participate in representing "
                        "multiple unrelated features (polysemanticity), "
                        "which makes individual neurons hard to interpret "
                        "and motivates techniques like sparse autoencoders "
                        "to decompose activations back into monosemantic "
                        "features."
    },
    {
        "question": "How do sparse autoencoders help decompose polysemantic "
                    "neurons into monosemantic features?",

        "ground_truth": "Sparse autoencoders are trained to reconstruct a "
                        "model's activations using an overcomplete "
                        "dictionary of features with a sparsity penalty "
                        "encouraging only a few features to activate for "
                        "any given input. This pulls apart the superposed "
                        "features entangled in polysemantic neurons and "
                        "yields a larger set of monosemantic features — "
                        "each corresponding to a single, human-interpretable "
                        "concept — which researchers use to understand what "
                        "a model has learned."
    },
    # -----
    # It's best to test your code with only a question or two
    # until you're confident that it's working.
    # This saves a lot of your free API calls until you need them.
    # -----
]
