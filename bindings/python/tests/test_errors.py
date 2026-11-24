from manifold3d import Error


def test_error_messages_are_unique():
    messages = [str(e) for e in Error]
    assert len(messages) == 15
    assert all(messages)
    assert "Unknown error" not in messages
    assert len(set(messages)) == len(messages)
    for e in Error:
        assert str(e) != repr(e)
