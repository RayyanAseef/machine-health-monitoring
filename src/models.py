from sklearn.linear_model import LogisticRegression

# Baseline Model
def create_logistic_regression():
    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    return model
