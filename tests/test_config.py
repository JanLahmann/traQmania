from traqmania.config import load_config


def test_default_config_loads():
    config = load_config()
    assert config["circuit"]["n_qubits"] == 4
    assert config["server"]["port"] == 8000


def test_profile_overlay():
    config = load_config(profile="pi5")
    assert config["training"]["batch_size"] == 16      # overlaid
    assert config["training"]["warm_start"] is True    # overlaid
    assert config["physics"]["v_max"] == 22.0          # inherited


def test_unknown_profile_raises():
    import pytest

    with pytest.raises(FileNotFoundError):
        load_config(profile="nope")
