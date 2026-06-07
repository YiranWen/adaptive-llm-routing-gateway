import pytest

from llm_router.feature_extractor import resolve_layer_index


def test_resolve_layer_index_second_to_last() -> None:
    assert resolve_layer_index(num_layers=24, layer_offset=-2) == 22


def test_resolve_layer_index_rejects_invalid_offset() -> None:
    with pytest.raises(IndexError):
        resolve_layer_index(num_layers=2, layer_offset=-3)


def test_mean_pool_hidden_states_ignores_padding() -> None:
    torch = pytest.importorskip("torch")
    from llm_router.feature_extractor import mean_pool_hidden_states

    hidden_states = torch.tensor(
        [
            [[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]],
            [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]],
        ]
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 0],
            [1, 1, 1],
        ]
    )

    pooled = mean_pool_hidden_states(hidden_states, attention_mask)

    assert torch.allclose(
        pooled,
        torch.tensor(
            [
                [2.0, 2.0],
                [6.0, 8.0],
            ]
        ),
    )

