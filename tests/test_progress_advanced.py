# tests/test_progress_advanced.py
"""
Advanced tests for progress reporting to improve coverage.
"""
import pytest
import time
import sys
import io
from contextlib import redirect_stdout

from model_inspector.utils.progress import (
    ProgressFormat, ProgressConfig, ProgressState,
    ProgressCallback, progress_iterator,
    BaseProgressHandler, BarProgressHandler,
    PercentageProgressHandler, SpinnerProgressHandler,
    PlainProgressHandler
)


class TestProgressHandlers:
    """Tests for the different progress handler implementations."""

    def test_base_handler(self):
        """Test the BaseProgressHandler class."""
        state = ProgressState(
            total=100,
            current=50,
            started_at=time.time() - 10  # Add this to ensure elapsed time
        )
        config = ProgressConfig()

        handler = BaseProgressHandler(state, config)

        # Test time formatting
        assert handler._format_time(30) == "30.0s"
        assert handler._format_time(90) == "1m 30s"
        assert handler._format_time(3700) == "1h 1m"

        # Test ETA calculation with non-zero progress
        eta = handler._calculate_eta()
        assert eta is not None and isinstance(eta, float)  # Fix this assertion

        # Test with no current progress (should not calculate ETA)
        state.current = 0
        assert handler._calculate_eta() is None

        # Test with no total (should not calculate ETA)
        state.total = None
        assert handler._calculate_eta() is None

        # Base handler's update and close methods should raise
        with pytest.raises(NotImplementedError):
            handler.update()

        with pytest.raises(NotImplementedError):
            handler.close()

    @pytest.mark.parametrize("handler_class", [
        BarProgressHandler,
        PercentageProgressHandler,
        SpinnerProgressHandler,
        PlainProgressHandler
    ])
    def test_handler_implementations(self, handler_class):
        """Test each progress handler implementation."""
        # Create a buffer to capture output
        buf = io.StringIO()

        # Create a state and config
        state = ProgressState(
            total=100,
            current=50,
            description="Testing",
            started_at=time.time() - 5  # Started 5 seconds ago
        )
        config = ProgressConfig(output_stream=buf)

        # Create the handler
        handler = handler_class(state, config)

        # Update should write to the buffer
        handler.update()
        assert len(buf.getvalue()) > 0

        # Empty the buffer
        buf.seek(0)
        buf.truncate()

        # Close should also write something
        handler.close()

        # For some handlers, close just writes a newline
        if handler_class is not PlainProgressHandler:
            assert len(buf.getvalue()) > 0 or buf.getvalue() == "\n"

    def test_progress_iterator(self):
        """Test the progress_iterator function."""
        # Create a list to iterate over
        items = list(range(10))

        # Create a buffer to capture output
        buf = io.StringIO()

        # Create a config that outputs to our buffer
        config = ProgressConfig(output_stream=buf)

        # Test with explicit total
        all_items = []
        for item in progress_iterator(items, total=10, config=config):
            all_items.append(item)

        # Verify we got all items
        assert all_items == items

        # Verify something was output
        assert len(buf.getvalue()) > 0

        # Empty the buffer
        buf.seek(0)
        buf.truncate()

        # Test with disabled progress
        for item in progress_iterator(items, disable=True, config=config):
            pass

        # Verify nothing was output when disabled
        assert buf.getvalue() == ""

        # Test with no total (should calculate from list)
        buf.seek(0)
        buf.truncate()
        for item in progress_iterator(items, config=config):
            pass

        # Verify something was output
        assert len(buf.getvalue()) > 0


class TestProgressCallback:
    """Tests for the ProgressCallback class."""

    def test_callback_methods(self):
        """Test all methods of the ProgressCallback class."""
        # Create storage for callback values
        values = {
            'start_called': False,
            'progress_calls': [],
            'complete_value': None,
            'error_value': None
        }

        # Create callback functions
        def on_start():
            values['start_called'] = True

        def on_progress(current, total, info):
            values['progress_calls'].append((current, total, info))

        def on_complete(result):
            values['complete_value'] = result

        def on_error(exception):
            values['error_value'] = str(exception)

        # Create the callback
        callback = ProgressCallback(
            on_start=on_start,
            on_progress=on_progress,
            on_complete=on_complete,
            on_error=on_error,
            throttle_ms=10  # Low value for testing
        )

        # Test start
        callback.start()
        assert values['start_called'] is True

        # Test progress
        callback.progress(1, 10)
        time.sleep(0.02)  # Ensure throttle time passes
        callback.progress(2, 10)
        time.sleep(0.02)  # Ensure throttle time passes
        callback.progress(3, 10)

        # Should have recorded all calls
        assert len(values['progress_calls']) == 3

        # Verify the structure of progress calls
        first_call = values['progress_calls'][0]
        assert first_call[0] == 1  # current
        assert first_call[1] == 10  # total
        assert isinstance(first_call[2], dict)  # info dict
        assert 'elapsed' in first_call[2]
        assert 'percentage' in first_call[2]

        # Test throttling
        callback.progress(4, 10)  # Immediate call
        callback.progress(5, 10)  # Should be throttled
        assert len(values['progress_calls']) == 3  # Should not have increased due to throttling

        # Test complete
        result = {"status": "success"}
        callback.complete(result)
        assert values['complete_value'] == result

        # Test error
        test_exception = ValueError("Test Error")
        callback.error(test_exception)
        assert values['error_value'] == "Test Error"

        # Test throttle bypass for last item
        callback.progress(10, 10)  # Last item should bypass throttle
        assert len(values['progress_calls']) == 4
        assert values['progress_calls'][-1][0] == 10  # Verify last progress call

    def test_progress_info_calculation(self):
        """Test the progress information calculation."""
        # Create a callback
        callback = ProgressCallback()

        # Call the calculation with known values
        info = callback._calculate_progress_info(50, 100)

        # Check that the info contains expected keys
        assert 'elapsed' in info
        assert 'elapsed_formatted' in info
        assert 'percentage' in info
        assert info['percentage'] == 50.0

        # Should contain rate information
        assert 'rate' in info
        assert 'rate_formatted' in info

        # Should contain ETA if available
        if 'eta' in info:
            assert 'eta_formatted' in info

        # Test with no total
        info = callback._calculate_progress_info(50, None)

        # Should not contain percentage or ETA
        assert 'percentage' not in info
        assert 'eta' not in info

        # Test with zero progress
        info = callback._calculate_progress_info(0, 100)

        # Should not have ETA with zero progress
        assert 'eta' not in info

        # Test time formatting
        assert callback._format_time(30) == "30.0s"
        assert callback._format_time(90) == "1m 30s"
        assert callback._format_time(3700) == "1h 1m"

    def test_item_collection(self):
        """Test collecting processed items in the callback."""
        # Create callback that collects items
        callback = ProgressCallback(include_items=True)

        # Add some progress with items
        callback.progress(1, 3, item="item1")
        callback.progress(2, 3, item="item2")
        callback.progress(3, 3, item="item3")

        # Items should be stored
        assert callback._processed_items == ["item1", "item2", "item3"]

        # Test that items are included in completion
        result = {}
        callback.complete(result)

        # The result should have the items added
        assert result["processed_items"] == ["item1", "item2", "item3"]
