def pytest_addoption(parser) -> None:
    parser.addoption("--submission-path", action="store", default="submission.csv")
