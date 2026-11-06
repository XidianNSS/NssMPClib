from nssmpc.infra.utils.common import align_shape


def test_align_shape_preserves_matrix_dimensions():
    assert align_shape([1, 1], [1, 32]) == ([1, 1], [1, 32])
    assert align_shape([1, 32], [32, 1]) == ([1, 32], [32, 1])


def test_align_shape_removes_only_redundant_batch_dimensions():
    assert align_shape([1, 1, 16, 32], [1, 1, 32, 8]) == ([16, 32], [32, 8])
