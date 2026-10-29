"""The Tac-LeWM sequence input uses no rest subtraction."""


def parse_mosaic_rest_sub(value):
    if (
        value is None
        or value is False
        or str(value).lower() in ("off", "none", "false", "0")
    ):
        return "off"
    raise ValueError("TacMind-0 Tac-LeWM fine-tuning supports mosaic_rest_sub=off only")
