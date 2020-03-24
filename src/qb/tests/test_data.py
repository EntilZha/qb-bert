from qb.data import create_char_runs


def test_create_char_runs():
    text = "name this first united states president"
    expected = [
        ("name this ", 10),
        ("name this first unit", 20),
        ("name this first united states ", 30),
        ("name this first united states president", 40),
    ]
    result = create_char_runs(text, 10)
    assert expected == result
