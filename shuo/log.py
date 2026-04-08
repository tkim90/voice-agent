"""
Centralized logging for shuo.

Provides:
- Configured console logger with colors
- Logger for consistent event/lifecycle/action logging
- ServiceLogger for individual services
"""

import logging
import sys
from typing import Optional

from .types import (
    Event,
    StreamStartEvent, StreamStopEvent, MediaEvent,
    FluxStartOfTurnEvent, FluxEndOfTurnEvent,
    AgentTurnDoneEvent,
    Action,
    FeedFluxAction, StartAgentTurnAction, ResetAgentTurnAction,
    Phase,
)


# =============================================================================
# COLORS
# =============================================================================

class C:
    """ANSI color codes."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Colors
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright colors
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"


def _c(color: str, text: str) -> str:
    """Wrap text in color codes."""
    return color + text + C.RESET


def _quote(text: str, color: str = C.WHITE) -> str:
    """Wrap text in quotes with color."""
    return _c(color, '"' + text + '"')


def preview_text(text: str, limit: int = 80) -> str:
    """Compact single-line preview for logs."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


# =============================================================================
# LOGGING SETUP
# =============================================================================

class ColorFormatter(logging.Formatter):
    """Custom formatter with colors and clean timestamp."""

    def format(self, record: logging.LogRecord) -> str:
        # Millisecond-precision timestamps for latency debugging
        ms = int(record.msecs)
        ts = self.formatTime(record, "%H:%M:%S") + f".{ms:03d}"
        time_str = _c(C.DIM, ts)
        return time_str + " \u2502 " + record.getMessage()


def _quiet_noisy_loggers() -> None:
    """Reduce noise from dependency loggers."""
    root = logging.getLogger()
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("twilio").setLevel(logging.WARNING)
    logging.getLogger("twilio.http_client").setLevel(logging.WARNING)


def setup_logging(level: int = logging.INFO, *, install_handler: bool = True) -> None:
    """Configure logging for the application."""
    root = logging.getLogger()
    root.setLevel(level)

    if install_handler:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(ColorFormatter())
        console.setLevel(level)
        root.handlers = [console]

    _quiet_noisy_loggers()


def compact_livekit_cli_logs() -> None:
    """Patch LiveKit's rich logger to avoid fixed-width message indentation."""
    try:
        from livekit.agents.cli import cli as livekit_cli
    except ImportError:
        return

    handler_cls = livekit_cli.RichLoggingHandler
    if getattr(handler_cls, "_shuo_compact_logs", False):
        return

    def emit(self, record: logging.LogRecord) -> None:
        def middle_truncate(text: str, max_width: int) -> str:
            if len(text) <= max_width:
                return text
            if max_width <= 1:
                return "…"[:max_width]
            visible = max_width - 1
            left = visible // 2
            right = visible - left
            return text[:left] + "…" + text[-right:]

        has_exc = bool(
            (record.exc_info and record.exc_info != (None, None, None)) or record.exc_text
        )

        if has_exc:
            exc_info, exc_text = record.exc_info, record.exc_text
            record.exc_info = None
            record.exc_text = None
            try:
                message = self.format(record)
            finally:
                record.exc_info, record.exc_text = exc_info, exc_text
        else:
            message = self.format(record)

        max_name_width = 18

        output = livekit_cli.Table.grid(padding=(0, 1))
        output.add_column(style="log.time")
        output.add_column(style="log.level", width=6, no_wrap=True)
        output.add_column(
            style="log.name",
            max_width=max_name_width,
            no_wrap=True,
            overflow="ellipsis",
        )
        output.add_column(ratio=1, style="log.message")
        output.add_column(style="log.extra", no_wrap=True)

        row = []

        time_format = None if self.formatter is None else self.formatter.datefmt
        log_time = livekit_cli.datetime.datetime.fromtimestamp(record.created)
        log_time = log_time or self.c.console.get_datetime()

        log_time_display = (
            livekit_cli.Text(log_time.strftime(time_format))
            if time_format
            else livekit_cli.Text(log_time.strftime("%H:%M:%S.%f")[:-3])
        )

        if log_time_display == self._last_time:
            time_str = log_time_display.plain
            row.append(livekit_cli.Text(" " * len(time_str)))
        else:
            row.append(log_time_display)
            self._last_time = log_time_display

        level_text = livekit_cli.Text.styled(
            record.levelname.ljust(8),
            f"logging.level.{record.levelname.lower()}",
        )
        row.append(level_text)

        logger_name = middle_truncate(record.name, max_name_width)
        name_text = livekit_cli.Text(logger_name)
        row.append(name_text)

        msg_text = livekit_cli.Text(message)
        row.append(msg_text)

        console_width = self.c.console.width
        available_width = max(console_width - 6, 20)

        time_len = log_time_display.cell_len
        level_len = 8
        name_len = name_text.cell_len
        msg_len = msg_text.cell_len

        extra = {}
        livekit_cli._merge_record_extra(record, extra)

        extra_str = ""
        extra_len = 0
        if extra:
            extra_str = livekit_cli.json.dumps(
                extra,
                cls=livekit_cli.JsonFormatter.JsonEncoder,
                ensure_ascii=False,
            )
            extra_text = livekit_cli.Text(extra_str)
            extra_len = extra_text.cell_len

        spaces_between_columns = 4
        total_len_with_extra = (
            time_len + level_len + name_len + msg_len + extra_len + spaces_between_columns
        )

        inline_extra = bool(extra_str) and total_len_with_extra <= available_width

        if inline_extra:
            row.append(livekit_cli.Text(extra_str, style="log.extra"))
        else:
            row.append(livekit_cli.Text(" "))

        output.add_row(*row)

        try:
            self.c.console.print(output)

            if extra_str and not inline_extra:
                indent_width = time_len + 1 + level_len + 1 + name_len + 1
                indent = " " * (indent_width + 2)
                extra_line = livekit_cli.Text(indent + extra_str, style="log.extra")
                self.c.console.print(extra_line)

            if has_exc:
                self._print_plain_traceback(record)
        except Exception:
            self.handleError(record)

    handler_cls.emit = emit
    handler_cls._shuo_compact_logs = True


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


# =============================================================================
# LOGGER (unified lifecycle + event + action logging)
# =============================================================================

class Logger:
    """
    Unified logger for shuo.

    Class methods  -- lifecycle events (server, call, websocket, stream)
    Instance methods -- event/action/transition logging in the conversation loop
    """

    _logger = logging.getLogger("shuo")

    # ── Lifecycle (class methods) ────────────────────────────────────

    @classmethod
    def server_starting(cls, port: int) -> None:
        cls._logger.info("\U0001F680 " + _c(C.CYAN, "Server starting on port " + str(port)))

    @classmethod
    def server_ready(cls, url: str) -> None:
        cls._logger.info(_c(C.GREEN, "\u2713  Ready") + " " + _c(C.DIM, url))

    @classmethod
    def call_initiating(cls, phone: str) -> None:
        cls._logger.info("\U0001F4DE " + _c(C.CYAN, "Calling " + phone + "..."))

    @classmethod
    def call_initiated(cls, sid: str) -> None:
        cls._logger.info(
            _c(C.GREEN, "\u2713  Call initiated") + " " + _c(C.DIM, "SID: " + sid[:8] + "...")
        )

    @classmethod
    def websocket_connected(cls) -> None:
        cls._logger.info("\U0001F50C " + _c(C.CYAN, "WebSocket connected"))

    @classmethod
    def websocket_disconnected(cls) -> None:
        cls._logger.info("\U0001F50C " + _c(C.DIM, "WebSocket disconnected"))

    @classmethod
    def shutdown(cls) -> None:
        cls._logger.info("\U0001F44B " + _c(C.DIM, "Shutting down"))

    # ── Instance methods (conversation loop) ─────────────────────────

    def __init__(self, verbose: bool = False):
        self._events_logger = logging.getLogger("shuo.events")
        self._verbose = verbose

    def event(self, event: Event) -> None:
        """Log an incoming event."""

        if isinstance(event, MediaEvent):
            if self._verbose:
                size = len(event.audio_bytes)
                self._events_logger.debug(_c(C.DIM, "\u2190 MediaEvent (" + str(size) + " bytes)"))
            return

        if isinstance(event, StreamStartEvent):
            self._events_logger.info(
                _c(C.GREEN, "\u25B6  Stream started") + " " +
                _c(C.DIM, "SID: " + event.stream_sid[:8] + "...")
            )
            return

        if isinstance(event, StreamStopEvent):
            self._events_logger.info("\u23F9  " + _c(C.DIM, "Stream stopped"))
            return

        if isinstance(event, FluxEndOfTurnEvent):
            text = event.transcript
            if len(text) > 60:
                text = text[:57] + "..."
            self._events_logger.info(
                _c(C.GREEN, "\u2190") + " " +
                _c(C.BRIGHT_BLUE, "Flux") + " " +
                _c(C.GREEN, "EndOfTurn") + " " +
                _quote(text)
            )
            return

        if isinstance(event, FluxStartOfTurnEvent):
            self._events_logger.info(
                _c(C.BRIGHT_RED, "\u26A1") + " " +
                _c(C.BRIGHT_BLUE, "Flux") + " " +
                _c(C.BRIGHT_RED, "StartOfTurn") + " " +
                _c(C.DIM, "(barge-in)")
            )
            return

        if isinstance(event, AgentTurnDoneEvent):
            self._events_logger.info(
                _c(C.GREEN, "\u2190") + " " +
                _c(C.DIM, "Agent turn done")
            )
            return

    def action(self, action: Action) -> None:
        """Log an outgoing action."""

        if isinstance(action, FeedFluxAction):
            if self._verbose:
                size = len(action.audio_bytes)
                self._events_logger.debug(_c(C.DIM, "\u2192 FeedFlux (" + str(size) + " bytes)"))
            return

        if isinstance(action, StartAgentTurnAction):
            msg = action.transcript
            if len(msg) > 40:
                msg = msg[:37] + "..."
            self._events_logger.info(
                _c(C.YELLOW, "\u2192") + " " +
                _c(C.YELLOW, "Start") + " " +
                _c(C.BRIGHT_CYAN, "Agent") + " " +
                _quote(msg, C.DIM)
            )
            return

        if isinstance(action, ResetAgentTurnAction):
            self._events_logger.info(
                _c(C.YELLOW, "\u2192") + " " +
                _c(C.BRIGHT_RED, "Reset") + " " +
                _c(C.BRIGHT_CYAN, "Agent")
            )
            return

    def transition(self, old_phase: Phase, new_phase: Phase) -> None:
        """Log a phase transition (magenta)."""
        if old_phase != new_phase:
            self._events_logger.info(
                _c(C.MAGENTA, "\u25C6") + " " +
                _c(C.DIM, old_phase.name) + " " +
                _c(C.MAGENTA, "\u2192") + " " +
                _c(C.BRIGHT_MAGENTA, new_phase.name)
            )

    def error(self, msg: str, exc: Optional[Exception] = None) -> None:
        """Log an error (red)."""
        if exc:
            self._events_logger.error(
                _c(C.RED, "\u2717 " + msg + ":") + " " + _c(C.DIM, str(exc))
            )
        else:
            self._events_logger.error(_c(C.RED, "\u2717 " + msg))


# =============================================================================
# SERVICE LOGGING
# =============================================================================

class ServiceLogger:
    """Logger for individual services (Flux, LLM, TTS, Player, Agent)."""

    COLORS = {
        "Flux": C.BRIGHT_BLUE,
        "LLM": C.BRIGHT_MAGENTA,
        "TTS": C.BRIGHT_CYAN,
        "Player": C.WHITE,
        "Agent": C.BRIGHT_GREEN,
    }

    def __init__(self, service_name: str):
        self._logger = logging.getLogger("shuo." + service_name)
        self._name = service_name
        self._color = self.COLORS.get(service_name, C.WHITE)

    def connected(self) -> None:
        self._logger.info(
            _c(C.GREEN, "\u2713") + " " + _c(self._color, self._name) + " " + _c(C.DIM, "connected")
        )

    def disconnected(self) -> None:
        self._logger.debug(_c(C.DIM, "\u25CB " + self._name + " disconnected"))

    def cancelled(self) -> None:
        self._logger.debug(_c(C.DIM, "\u25CB " + self._name + " cancelled"))

    def error(self, msg: str, exc: Optional[Exception] = None) -> None:
        if exc:
            self._logger.error(
                _c(C.RED, "\u2717") + " " +
                _c(self._color, self._name + ":") + " " +
                msg + " " + _c(C.DIM, "(" + str(exc) + ")")
            )
        else:
            self._logger.error(
                _c(C.RED, "\u2717") + " " + _c(self._color, self._name + ":") + " " + msg
            )

    def warning(self, msg: str) -> None:
        self._logger.warning(
            _c(C.YELLOW, self._name + ":") + " " + _c(C.YELLOW, msg)
        )

    def debug(self, msg: str) -> None:
        self._logger.debug(_c(C.DIM, self._name + ": " + msg))

    def info(self, msg: str) -> None:
        self._logger.info(_c(self._color, self._name + ":") + " " + msg)
