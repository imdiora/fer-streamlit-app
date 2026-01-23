from emotion_ai.config import load_config


def test_config_loads():
    cfg = load_config("config/config.yaml")
    assert cfg.model.num_classes > 0
    assert isinstance(cfg.labels, dict)
    assert 0 in cfg.labels
