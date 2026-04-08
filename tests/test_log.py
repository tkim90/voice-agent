import io
import logging

from rich.console import Console
from rich.theme import Theme

from shuo.log import compact_livekit_cli_logs, setup_logging


class _FakeAgentsConsole:
    def __init__(self) -> None:
        self.buffer = io.StringIO()
        self.console = Console(
            file=self.buffer,
            width=120,
            color_system=None,
            force_terminal=False,
            legacy_windows=False,
            theme=Theme(
                {
                    "log.time": "none",
                    "log.level": "none",
                    "log.name": "none",
                    "log.message": "none",
                    "log.extra": "none",
                    "logging.level.info": "none",
                }
            ),
        )

    def _render_tag(self, renderable, tag_width=2):
        return renderable


def test_compact_livekit_cli_logs_removes_fixed_name_padding():
    from livekit.agents.cli import cli as livekit_cli

    compact_livekit_cli_logs()
    handler = livekit_cli.RichLoggingHandler(_FakeAgentsConsole())

    record = logging.LogRecord(
        name="shuo.Agent",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="Agent: Turn started",
        args=(),
        exc_info=None,
    )

    handler.emit(record)
    output = handler.c.buffer.getvalue()
    first_line = output.splitlines()[0]

    assert "shuo.Agent Agent: Turn started" in output
    assert "shuo.Agent         Agent: Turn started" not in output
    assert first_line[0].isdigit()


def test_setup_logging_can_skip_installing_root_handler():
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level

    try:
        root.handlers = []
        setup_logging(install_handler=False)
        assert root.handlers == []
    finally:
        root.handlers = original_handlers
        root.setLevel(original_level)
