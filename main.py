"""Entry point: run both Iris and Titanic pipelines."""
from src.iris_classifier import run_iris_pipeline
from src.titanic_classifier import run_titanic_pipeline


def main():
    print("=" * 60)
    print("IRIS CLASSIFICATION")
    print("=" * 60)
    iris_results = run_iris_pipeline()

    print("\n" + "=" * 60)
    print("TITANIC SURVIVAL CLASSIFICATION")
    print("=" * 60)
    titanic_results = run_titanic_pipeline()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in iris_results:
        print(f"[Iris]    {r['model']:<20} acc={r['accuracy']:.3f} cv={r['cv_mean']:.3f}")
    for r in titanic_results:
        print(f"[Titanic] {r['model']:<20} acc={r['accuracy']:.3f} cv={r['cv_mean']:.3f}")


if __name__ == "__main__":
    main()