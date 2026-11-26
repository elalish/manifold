from manifold3d import Error


def test_error_names_and_messages():
    errors = list(Error)
    assert [e.value for e in errors] == list(range(len(errors)))
    names = [e.name for e in errors]
    messages = [str(e) for e in errors]
    assert all(names) and all(messages)
    assert "Unknown error" not in messages
    assert len(set(names)) == len(names)
    assert len(set(messages)) == len(messages)
    for e in errors:
        assert Error[e.name] is e
        assert str(e) != repr(e)
