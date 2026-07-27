import torch

from word2vec.evaluate import Question, evaluate_analogies, load_questions
from word2vec.vocab import Vocab


def test_evaluate_analogies_exact_arithmetic():
    # man:king :: woman:queen, constructed so king - man + woman == queen exactly
    id2word = ["man", "king", "woman", "queen", "dog"]
    vectors = torch.tensor(
        [
            [1.0, 0.0],  # man
            [1.0, 1.0],  # king
            [0.0, 1.0],  # woman
            [0.0, 2.0],  # queen = king - man + woman
            [-5.0, -5.0],  # dog: unrelated distractor
        ]
    )
    word2id = {w: i for i, w in enumerate(id2word)}

    q = Question(category="family", is_semantic=True, a=word2id["man"], b=word2id["king"], c=word2id["woman"], d=word2id["queen"])
    result = evaluate_analogies(vectors, [q], device=torch.device("cpu"))

    assert result["total"]["correct"] == 1
    assert result["total"]["accuracy"] == 1.0
    assert result["semantic"]["accuracy"] == 1.0
    assert "syntactic" not in result


def test_evaluate_analogies_excludes_input_words():
    # degenerate case: if the model just returned vec(c) unchanged, the
    # nearest neighbor search must not be allowed to answer with c itself
    id2word = ["a", "b", "c", "d"]
    vectors = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],  # correct answer points the opposite way
        ]
    )
    q = Question(category="x", is_semantic=False, a=0, b=1, c=2, d=3)
    result = evaluate_analogies(vectors, [q], device=torch.device("cpu"))
    # target = vec(b) - vec(a) + vec(c) = [1,0]; nearest among non-{a,b,c} is d
    assert result["total"]["correct"] == 1


def test_load_questions_parses_categories_and_lowercases(tmp_path):
    path = tmp_path / "questions.txt"
    path.write_text(
        ": capital-common-countries\nAthens Greece Oslo Norway\n"
        ": gram8-plural\nmouse mice dollar dollars\n",
        encoding="utf-8",
    )
    id2word = ["athens", "greece", "oslo", "norway", "mouse", "mice", "dollar", "dollars"]
    vocab = Vocab(word2id={w: i for i, w in enumerate(id2word)}, id2word=id2word, counts=[10] * len(id2word))

    questions, stats = load_questions(path, vocab)
    assert stats == {"total": 2, "skipped_oov": 0, "evaluated": 2}
    assert questions[0].category == "capital-common-countries"
    assert questions[0].is_semantic is True
    assert questions[1].category == "gram8-plural"
    assert questions[1].is_semantic is False


def test_load_questions_skips_oov():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "q.txt"
        path.write_text(": family\nbrother sister grandson granddaughter\n", encoding="utf-8")
        vocab = Vocab(word2id={"brother": 0, "sister": 1}, id2word=["brother", "sister"], counts=[5, 5])
        questions, stats = load_questions(path, vocab)
        assert stats == {"total": 1, "skipped_oov": 1, "evaluated": 0}
        assert questions == []
