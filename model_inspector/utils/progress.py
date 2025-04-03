from typing import Iterable, TypeVar, Iterator, Optional, Callable, Any, Dict, List, Union
import sys
import threading
import time
import logging
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Type variable for generics
T = TypeVar('T')


class ProgressFormat(Enum):
    """Output format for progress reporting."""
    PLAIN = 'plain'  # Simple text output
    BAR = 'bar'  # Progress bar
    PERCENTAGE = 'pct'  # Percentage only
    SPINNER = 'spinner'  # Animated spinner


@dataclass
class ProgressConfig:
    """Configuration for progress reporting."""
    format: ProgressFormat = ProgressFormat.BAR
    width: int = 50
    show_count: bool = True
    show_percentage: bool = True
    show_elapsed: bool = True
    show_eta: bool = True
    refresh_rate: float = 0.2
    spinner_chars: str = '⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
    use_colors: bool = True
    output_stream: Any = field(default=None)  # Change this line

    def __post_init__(self):
        """Initialize after dataclass creation."""
        # Default to stderr if no stream specified
        if self.output_stream is None:
            self.output_stream = sys.stderr


@dataclass
class ProgressState:
    """Internal state of a progress operation."""
    total: Optional[int] = None  # Total number of items
    current: int = 0  # Current progress
    started_at: float = field(default_factory=time.time)  # Start time
    last_update: float = 0  # Time of last update
    description: Optional[str] = None  # Description of the operation
    closed: bool = False  # Whether the progress has completed


def progress_iterator(
        iterable: Iterable[T],
        desc: Optional[str] = None,
        total: Optional[int] = None,
        config: Optional[ProgressConfig] = None,
        disable: bool = False
) -> Iterator[T]:
    """
    Wrap an iterable with a progress bar.

    Args:
        iterable: The iterable to wrap
        desc: Description for the progress bar
        total: Total number of items (calculated if None)
        config: Progress display configuration
        disable: Whether to disable the progress bar

    Yields:
        Items from the iterable
    """
    if disable:
        yield from iterable
        return

    # If we don't have a total and the iterable is sized, get the length
    if total is None and hasattr(iterable, '__len__'):
        try:
            total = len(iterable)
        except (TypeError, AttributeError):
            pass

    # Use default config if none provided
    if config is None:
        config = ProgressConfig()

    # Initialize state
    state = ProgressState(
        total=total,
        description=desc,
        started_at=time.time()
    )

    # Handle different progress formats
    if config.format == ProgressFormat.BAR:
        progress_handler = BarProgressHandler(state, config)
    elif config.format == ProgressFormat.PERCENTAGE:
        progress_handler = PercentageProgressHandler(state, config)
    elif config.format == ProgressFormat.SPINNER:
        progress_handler = SpinnerProgressHandler(state, config)
    else:  # Default to plain
        progress_handler = PlainProgressHandler(state, config)

    try:
        # Show initial progress
        progress_handler.update()

        # Process the iterable
        for i, item in enumerate(iterable):
            yield item

            # Update progress
            state.current = i + 1
            now = time.time()

            # Throttle updates by refresh rate
            if now - state.last_update >= config.refresh_rate:
                progress_handler.update()
                state.last_update = now

        # Ensure we show 100% at the end
        state.current = state.total if state.total is not None else state.current
        progress_handler.update()

    finally:
        # Ensure we close the progress indicator
        state.closed = True
        progress_handler.close()


class BaseProgressHandler:
    """Base class for progress display handlers."""

    def __init__(self, state: ProgressState, config: ProgressConfig):
        """
        Initialize the progress handler.

        Args:
            state: Progress state
            config: Progress configuration
        """
        self.state = state
        self.config = config

    def update(self) -> None:
        """Update the progress display."""
        raise NotImplementedError()

    def close(self) -> None:
        """Close the progress display."""
        raise NotImplementedError()

    def _format_time(self, seconds: float) -> str:
        """
        Format time in seconds to a human-readable string.

        Args:
            seconds: Time in seconds

        Returns:
            Formatted time string
        """
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            seconds %= 60
            return f"{minutes}m {seconds:.0f}s"
        else:
            hours = int(seconds / 3600)
            seconds %= 3600
            minutes = int(seconds / 60)
            return f"{hours}h {minutes}m"

    def _calculate_eta(self) -> Optional[float]:
        """
        Calculate estimated time remaining.

        Returns:
            Estimated seconds remaining or None if can't calculate
        """
        if (self.state.total is None or
                self.state.current == 0 or
                self.state.total == 0):
            return None

        elapsed = time.time() - self.state.started_at
        if elapsed <= 0:
            return None

        rate = self.state.current / elapsed
        if rate <= 0:
            return None

        remaining_items = self.state.total - self.state.current
        return remaining_items / rate


class BarProgressHandler(BaseProgressHandler):
    """Handler for progress bar display."""

    def update(self) -> None:
        """Update the progress bar display."""
        stream = self.config.output_stream

        # Calculate percentage
        pct = (self.state.current / self.state.total * 100) if self.state.total else 0

        # Elapsed time
        elapsed = time.time() - self.state.started_at

        # Build the progress string
        elements = []

        # Description
        if self.state.description:
            elements.append(f"{self.state.description}: ")

        # Item counts
        if self.config.show_count:
            if self.state.total:
                elements.append(f"{self.state.current}/{self.state.total} ")
            else:
                elements.append(f"{self.state.current} items ")

        # Progress bar
        if self.state.total:
            bar_width = self.config.width
            filled = int(pct / 100 * bar_width)
            bar = "█" * filled + "░" * (bar_width - filled)
            elements.append(f"|{bar}| ")

        # Percentage
        if self.config.show_percentage and self.state.total:
            elements.append(f"{pct:.1f}% ")

        # Elapsed time
        if self.config.show_elapsed:
            elements.append(f"[{self._format_time(elapsed)}] ")

        # ETA
        if self.config.show_eta and self.state.total:
            eta = self._calculate_eta()
            if eta is not None:
                elements.append(f"ETA: {self._format_time(eta)}")

        # Join all elements and print
        progress_line = "".join(elements)
        print(f"\r{progress_line}", end="", file=stream, flush=True)

    def close(self) -> None:
        """Close the progress bar with newline."""
        print(file=self.config.output_stream)


class PercentageProgressHandler(BaseProgressHandler):
    """Handler for percentage-only progress display."""

    def update(self) -> None:
        """Update the percentage display."""
        stream = self.config.output_stream

        # Calculate percentage
        pct = (self.state.current / self.state.total * 100) if self.state.total else 0

        # Build the progress string
        elements = []

        # Description
        if self.state.description:
            elements.append(f"{self.state.description}: ")

        # Percentage
        if self.state.total:
            elements.append(f"{pct:.1f}% ")
        else:
            elements.append(f"{self.state.current} items ")

        # Join all elements and print
        progress_line = "".join(elements)
        print(f"\r{progress_line}", end="", file=stream, flush=True)

    def close(self) -> None:
        """Close the progress display."""
        print(file=self.config.output_stream)


class SpinnerProgressHandler(BaseProgressHandler):
    """Handler for spinner animation progress display."""

    def __init__(self, state: ProgressState, config: ProgressConfig):
        """
        Initialize the spinner handler.

        Args:
            state: Progress state
            config: Progress configuration
        """
        super().__init__(state, config)
        self.spinner_idx = 0

    def update(self) -> None:
        """Update the spinner display."""
        stream = self.config.output_stream

        # Get the next spinner character
        spinner = self.config.spinner_chars[self.spinner_idx % len(self.config.spinner_chars)]
        self.spinner_idx += 1

        # Elapsed time
        elapsed = time.time() - self.state.started_at

        # Build the progress string
        elements = [spinner, " "]

        # Description
        if self.state.description:
            elements.append(f"{self.state.description} ")

        # Item counts
        if self.config.show_count:
            if self.state.total:
                elements.append(f"{self.state.current}/{self.state.total} ")
            else:
                elements.append(f"{self.state.current} items ")

        # Elapsed time
        if self.config.show_elapsed:
            elements.append(f"[{self._format_time(elapsed)}]")

        # Join all elements and print
        progress_line = "".join(elements)
        print(f"\r{progress_line}", end="", file=stream, flush=True)

    def close(self) -> None:
        """Close the spinner display."""
        print(file=self.config.output_stream)


class PlainProgressHandler(BaseProgressHandler):
    """Handler for plain text progress display."""

    def __init__(self, state: ProgressState, config: ProgressConfig):
        """
        Initialize the plain text handler.

        Args:
            state: Progress state
            config: Progress configuration
        """
        super().__init__(state, config)
        self.last_report = 0

    def update(self) -> None:
        """Update the plain progress display."""
        # Only output updates at certain points
        if self.state.total:
            pct = self.state.current / self.state.total * 100
            # Report at 0%, 25%, 50%, 75%, and 100%
            milestone = int(pct / 25) * 25
            if milestone > self.last_report:
                self.last_report = milestone
                self._report_progress()
        elif self.state.current % 100 == 0:
            # For unknown totals, report every 100 items
            self._report_progress()

    def _report_progress(self) -> None:
        """Print a progress report line."""
        stream = self.config.output_stream

        # Elapsed time
        elapsed = time.time() - self.state.started_at

        # Build the progress string
        elements = []

        # Description
        if self.state.description:
            elements.append(f"{self.state.description}: ")

        # Item counts
        if self.state.total:
            pct = self.state.current / self.state.total * 100
            elements.append(f"{self.state.current}/{self.state.total} ({pct:.1f}%)")
        else:
            elements.append(f"{self.state.current} items")

        # Elapsed time
        elements.append(f" in {self._format_time(elapsed)}")

        # Join all elements and print
        progress_line = "".join(elements)
        print(progress_line, file=stream, flush=True)

    def close(self) -> None:
        """Close the progress display."""
        self._report_progress()


class ProgressCallback:
    """
    Callback handler for reporting progress in a more customizable way.
    """

    def __init__(
            self,
            on_start: Optional[Callable[[], None]] = None,
            on_progress: Optional[Callable[[int, Optional[int], Dict[str, Any]], None]] = None,
            on_complete: Optional[Callable[[Any], None]] = None,
            on_error: Optional[Callable[[Exception], None]] = None,
            throttle_ms: int = 100,
            include_items: bool = False
    ):
        """
        Initialize the progress callback.
        """
        self.on_start = on_start
        self.on_progress = on_progress
        self.on_complete = on_complete
        self.on_error = on_error
        self.throttle_ms = throttle_ms
        self.include_items = include_items

        # Internal state
        self._last_update = 0
        self._started_at = 0
        self._processed_items = []

    def start(self, total: Optional[int] = None) -> None:
        """
        Call the on_start callback.
        """
        self._started_at = time.time()
        self._last_update = 0
        self._processed_items = []

        if self.on_start:
            self.on_start()

    def progress(
            self,
            current: int,
            total: Optional[int] = None,
            item: Any = None
    ) -> None:
        """
        Call the on_progress callback with throttling.
        """
        # Store item if requested
        if self.include_items and item is not None:
            self._processed_items.append(item)

        if self.on_progress:
            now = time.time()
            elapsed_ms = (now - self._last_update) * 1000

            # Apply throttling unless it's the first or last update
            is_first = current == 1
            is_last = total is not None and current == total

            if is_first or is_last or elapsed_ms >= self.throttle_ms:
                # Calculate additional progress info
                info = self._calculate_progress_info(current, total)

                # Call the callback
                self.on_progress(current, total, info)
                self._last_update = now

    def complete(self, result: Any) -> None:
        """
        Call the on_complete callback.

        Args:
            result: Result of the processing
        """
        if self.include_items:
            if isinstance(result, dict):
                result["processed_items"] = self._processed_items.copy()
            else:
                result = {
                    "result": result,
                    "processed_items": self._processed_items.copy()
                }

        if self.on_complete:
            self.on_complete(result)

    def error(self, exception: Exception) -> None:
        """
        Call the on_error callback.
        """
        if self.on_error:
            self.on_error(exception)

    def _calculate_progress_info(
            self,
            current: int,
            total: Optional[int]
    ) -> Dict[str, Any]:
        """
        Calculate additional progress information.
        """
        now = time.time()
        elapsed = now - self._started_at

        info = {
            'elapsed': elapsed,
            'elapsed_formatted': self._format_time(elapsed),
        }

        # Calculate percentage and ETA if total is known
        if total is not None and total > 0:
            percentage = (current / total) * 100
            info['percentage'] = percentage

            # Calculate ETA
            if current > 0:
                rate = current / elapsed if elapsed > 0 else 0
                remaining_items = total - current

                if rate > 0:
                    eta = remaining_items / rate
                    info['eta'] = eta
                    info['eta_formatted'] = self._format_time(eta)
                    info['estimated_total_time'] = elapsed + eta

        # Calculate processing rate
        if elapsed > 0:
            rate = current / elapsed
            info['rate'] = rate
            info['rate_formatted'] = f"{rate:.1f} items/s"

        return info

    def _format_time(self, seconds: float) -> str:
        """
        Format time in seconds to a human-readable string.
        """
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            seconds %= 60
            return f"{minutes}m {seconds:.0f}s"
        else:
            hours = int(seconds / 3600)
            seconds %= 3600
            minutes = int(seconds / 60)
            return f"{hours}h {minutes}m"
