# Utility functions
def format_predictions(probs, encoder):
    top3 = probs.argsort()[-3:][::-1]
    crops = encoder.inverse_transform(top3)
    return [(crops[i], round(probs[top3[i]] * 100, 2)) for i in range(3)]
