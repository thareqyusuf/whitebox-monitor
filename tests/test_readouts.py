import math

import pytest

from whitebox_monitor.core.readouts import (
    expected_score,
    parse_integer,
    probs_from_logprobs,
    score_token_ids,
)


@pytest.mark.parametrize("text,want", [
    ("<reasoning>fine</reasoning>\n<score>\n3\n</score>", 3),
    ("<score>10</score>", 10),
    ("<SCORE> 0 </SCORE>", 0),
    ("<score type=\"int\">7</score>", 7),         # September's loose regex
])
def test_parse_integer(text, want):
    value, info = parse_integer(text)
    assert value == want and info["reason"] is None
    a, b = info["match_span"]
    assert text[a:b] == str(want)


@pytest.mark.parametrize("text", [
    None, "", "I think it is a 7.", "<final_score>7</final_score>", "<score>high</score>", "<score>11</score>",
    "<score>-1</score>", "<score>3.5</score>", "score: 4", "<score>4",
])
def test_unparseable_is_none_never_a_default(text):
    value, info = parse_integer(text)
    assert value is None
    assert info["reason"]


def test_out_of_range_is_flagged():
    value, info = parse_integer("<score>42</score>")
    assert value is None and info["out_of_range"] and info["raw"] == 42


def test_expected_score_hand_computed():
    # raw vocabulary probabilities on the eleven score tokens; the other 0.2
    # of the mass is on non-score tokens
    probs = {0: 0.40, 1: 0.10, 2: 0.08, 3: 0.02, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.10,
             8: 0.0, 9: 0.0, 10: 0.10}
    e, normalised, info = expected_score(probs)
    # (0*.4 + 1*.1 + 2*.08 + 3*.02 + 7*.1 + 10*.1) / 0.8 = 2.02 / 0.8
    assert e == pytest.approx(2.525, abs=1e-12)
    assert info["mass"] == pytest.approx(0.8)
    assert sum(normalised.values()) == pytest.approx(1.0, abs=1e-12)
    assert normalised["0"] == pytest.approx(0.5) and normalised["10"] == pytest.approx(0.125)
    assert list(normalised) == [str(v) for v in range(11)]


def test_expected_score_matches_september_formula():
    """Same arithmetic as run_monitor.read_distribution at 788c25f."""
    dist = {str(v): p for v, p in enumerate(
        [0.846, 0.0421, 0.0787, 0.0289, 0.00305, 0.00053, 0.00028, 0.00013, 0.0, 0.0, 0.0])}
    mass = sum(dist.values())
    september = sum(int(v) * p for v, p in dist.items()) / mass
    e, _, info = expected_score({int(v): p for v, p in dist.items()})
    assert e == pytest.approx(september, abs=1e-15) and info["mass"] == pytest.approx(mass)


def test_expected_score_none_cases():
    assert expected_score(None)[0] is None
    e, normalised, info = expected_score({0: 0.0, 1: 0.0})
    assert e is None and normalised is None and "zero mass" in info["reason"]
    with pytest.raises(ValueError):
        expected_score({0: float("nan")})
    with pytest.raises(ValueError):
        expected_score({0: -0.1})


def test_probs_from_logprobs_reports_missing():
    probs, info = probs_from_logprobs({"0": math.log(0.5), "3": math.log(0.25), "the": -1.0},
                                      range(11))
    assert probs[0] == pytest.approx(0.5) and probs[3] == pytest.approx(0.25) and probs[1] == 0.0
    assert info["missing"] == [1, 2, 4, 5, 6, 7, 8, 9, 10]


def test_score_token_ids(tokenizer):
    ids, info = score_token_ids(tokenizer, range(11))
    assert sorted(ids) == list(range(11)) and len(set(ids.values())) == 11
    assert all(tokenizer.decode([t]) == str(v) for v, t in ids.items())
    assert info["score_values_dropped"] == []


def test_multi_token_score_raises_unless_dropped(tokenizer):
    with pytest.raises(ValueError, match="single tokens"):
        score_token_ids(tokenizer, [0, 5, 37])
    ids, info = score_token_ids(tokenizer, [0, 5, 37], multi_token="drop")
    assert sorted(ids) == [0, 5] and info["score_values_dropped"] == [37] and "note" in info
