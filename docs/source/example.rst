Example
=======

In this section we show how to use the library. We will use the Titanic dataset, but you can use any other dataset.

.. code-block:: python
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    from titanic_model_processor.main import make_pipeline, fit, export

    X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    pipeline = make_pipeline(
        num_vars=("age", "fare"), 
        cat_vars=("pclass", "sex", "embarked"),
        classifier=RandomForestClassifier(n_estimators=100),
        scaler=StandardScaler(),
        num_imp_strategy="median"
    )

    pipeline = fit(pipeline=pipeline, x_train=X_train, y_train=y_train)

    print(f"Train score: {pipeline.score(X_train, y_train)}")
    print(f"Test score: {pipeline.score(X_test, y_test)}")

    export(pipeline=pipeline, file="titanic.pkl")