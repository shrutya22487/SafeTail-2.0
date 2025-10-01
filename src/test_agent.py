from agent import get_subsets

def test_get_subsets_empty_set():
    # Edge case: empty input set
    result = get_subsets(set())
    assert result == [], "Expected empty list for empty input set"

def test_get_subsets_single_element():
    # Input: {1}
    result = get_subsets({1})
    assert result == [[1]], "Expected one subset for single-element set"

def test_get_subsets_two_elements():
    # Input: {1, 2}
    result = get_subsets({1, 2})
    expected = [[1], [2], [1, 2]]
    assert sorted(result) == sorted(expected), "Incorrect subsets for two-element set"

def test_get_subsets_three_elements():
    # Input: {1, 2, 3}
    result = get_subsets({1, 2, 3})
    expected = [
        [1], [2], [3],
        [1, 2], [1, 3], [2, 3],
        [1, 2, 3]
    ]
    assert sorted([sorted(sub) for sub in result]) == sorted([sorted(sub) for sub in expected]), "Incorrect subsets for three-element set"

def test_get_subsets_type_and_uniqueness():
    # Ensure all subsets are lists and unique
    input_set = {1, 2, 3}
    result = get_subsets(input_set)
    assert all(isinstance(sub, list) for sub in result), "All subsets should be lists"
    assert len(result) == len(set(tuple(sorted(sub)) for sub in result)), "Subsets should be unique"
