"""
Event Processing System for Backtesting Engine

This module implements the event-driven architecture that powers the backtesting
system. The event queue processes market data, signals, orders, and fills in 
chronological order, ensuring realistic simulation of trading operations.
"""

from collections import deque
from typing import List, Optional, Callable, Dict, Any
from datetime import datetime
import logging
from threading import Lock

from .interfaces import Event, EventType, MarketEvent, SignalEvent, OrderEvent, FillEvent

logger = logging.getLogger(__name__)


class EventQueue:
    """
    Thread-safe event queue for managing backtesting events.
    
    Events are processed in chronological order, with support for 
    event priorities and filtering.
    """
    
    def __init__(self, max_size: Optional[int] = None):
        """
        Initialize event queue.
        
        Args:
            max_size: Maximum queue size (None for unlimited)
        """
        self._queue = deque(maxlen=max_size)
        self._lock = Lock()
        self._event_count = 0
        self._processed_count = 0
        
    def put(self, event: Event) -> None:
        """
        Add event to queue.
        
        Args:
            event: Event to add
        """
        with self._lock:
            # Insert event in chronological order
            inserted = False
            for i, existing_event in enumerate(self._queue):
                if event.timestamp <= existing_event.timestamp:
                    self._queue.insert(i, event)
                    inserted = True
                    break
            
            if not inserted:
                self._queue.append(event)
            
            self._event_count += 1
            
        logger.debug(f"Added {event.event_type.value} event at {event.timestamp}")
    
    def get(self) -> Optional[Event]:
        """
        Get next event from queue.
        
        Returns:
            Next event or None if queue is empty
        """
        with self._lock:
            if self._queue:
                event = self._queue.popleft()
                self._processed_count += 1
                logger.debug(f"Processing {event.event_type.value} event at {event.timestamp}")
                return event
            return None
    
    def peek(self) -> Optional[Event]:
        """
        Peek at next event without removing it.
        
        Returns:
            Next event or None if queue is empty
        """
        with self._lock:
            return self._queue[0] if self._queue else None
    
    def size(self) -> int:
        """Get current queue size."""
        with self._lock:
            return len(self._queue)
    
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return self.size() == 0
    
    def clear(self) -> None:
        """Clear all events from queue."""
        with self._lock:
            self._queue.clear()
            
    def get_statistics(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self._lock:
            return {
                'current_size': len(self._queue),
                'total_events': self._event_count,
                'processed_events': self._processed_count,
                'pending_events': len(self._queue)
            }


class EventProcessor:
    """
    Event processor that handles event routing and execution.
    
    The processor maintains handlers for different event types and
    ensures events are processed in the correct order.
    """
    
    def __init__(self, queue: EventQueue):
        """
        Initialize event processor.
        
        Args:
            queue: Event queue to process
        """
        self.queue = queue
        self.handlers: Dict[EventType, List[Callable]] = {}
        self._processing_enabled = True
        self._current_time = None
        
    def register_handler(self, event_type: EventType, handler: Callable) -> None:
        """
        Register event handler for specific event type.
        
        Args:
            event_type: Type of event to handle
            handler: Handler function that accepts event as parameter
        """
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)
        logger.info(f"Registered handler for {event_type.value} events")
    
    def unregister_handler(self, event_type: EventType, handler: Callable) -> None:
        """
        Unregister event handler.
        
        Args:
            event_type: Event type
            handler: Handler to remove
        """
        if event_type in self.handlers:
            try:
                self.handlers[event_type].remove(handler)
                logger.info(f"Unregistered handler for {event_type.value} events")
            except ValueError:
                logger.warning(f"Handler not found for {event_type.value}")
    
    def process_next_event(self) -> bool:
        """
        Process next event in queue.
        
        Returns:
            True if event was processed, False if queue is empty
        """
        if not self._processing_enabled:
            return False
            
        event = self.queue.get()
        if event is None:
            return False
        
        self._current_time = event.timestamp
        self._process_event(event)
        return True
    
    def process_all_events(self) -> int:
        """
        Process all events in queue.
        
        Returns:
            Number of events processed
        """
        processed_count = 0
        while self.process_next_event():
            processed_count += 1
        return processed_count
    
    def process_events_until(self, end_time: datetime) -> int:
        """
        Process events until specified time.
        
        Args:
            end_time: Time to process events until
            
        Returns:
            Number of events processed
        """
        processed_count = 0
        
        while True:
            next_event = self.queue.peek()
            if next_event is None or next_event.timestamp > end_time:
                break
                
            if self.process_next_event():
                processed_count += 1
            else:
                break
                
        return processed_count
    
    def _process_event(self, event: Event) -> None:
        """
        Process individual event by calling registered handlers.
        
        Args:
            event: Event to process
        """
        handlers = self.handlers.get(event.event_type, [])
        
        if not handlers:
            logger.warning(f"No handlers registered for {event.event_type.value}")
            return
        
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in {event.event_type.value} handler: {e}")
                # Continue processing other handlers
    
    def enable_processing(self) -> None:
        """Enable event processing."""
        self._processing_enabled = True
        
    def disable_processing(self) -> None:
        """Disable event processing."""
        self._processing_enabled = False
    
    def get_current_time(self) -> Optional[datetime]:
        """Get current processing time."""
        return self._current_time
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processor statistics."""
        handler_counts = {et.value: len(handlers) for et, handlers in self.handlers.items()}
        
        return {
            'processing_enabled': self._processing_enabled,
            'current_time': self._current_time,
            'registered_handlers': handler_counts,
            'queue_stats': self.queue.get_statistics()
        }


class EventLogger:
    """
    Event logger for debugging and analysis.
    
    Logs all events passing through the system with configurable
    detail levels and filtering options.
    """
    
    def __init__(self, log_level: str = "INFO", log_file: Optional[str] = None):
        """
        Initialize event logger.
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            log_file: Optional file to log to
        """
        self.logger = logging.getLogger(f"{__name__}.EventLogger")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        if log_file:
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.event_counts: Dict[EventType, int] = {}
        self._enabled_types = set(EventType)
    
    def log_event(self, event: Event) -> None:
        """
        Log an event.
        
        Args:
            event: Event to log
        """
        if event.event_type not in self._enabled_types:
            return
        
        self.event_counts[event.event_type] = self.event_counts.get(event.event_type, 0) + 1
        
        if event.event_type == EventType.MARKET:
            self._log_market_event(event)
        elif event.event_type == EventType.SIGNAL:
            self._log_signal_event(event)
        elif event.event_type == EventType.ORDER:
            self._log_order_event(event)
        elif event.event_type == EventType.FILL:
            self._log_fill_event(event)
        else:
            self.logger.info(f"{event.event_type.value}: {event.timestamp}")
    
    def _log_market_event(self, event: MarketEvent) -> None:
        """Log market event."""
        self.logger.debug(
            f"MARKET: {event.timestamp} {event.symbol} "
            f"Price={event.price:.4f} Volume={event.volume}"
        )
    
    def _log_signal_event(self, event: SignalEvent) -> None:
        """Log signal event."""
        self.logger.info(
            f"SIGNAL: {event.timestamp} {event.symbol} "
            f"Type={event.signal_type} Strength={event.strength:.3f}"
        )
    
    def _log_order_event(self, event: OrderEvent) -> None:
        """Log order event."""
        self.logger.info(
            f"ORDER: {event.timestamp} {event.order_id} "
            f"{event.side} {event.quantity} {event.symbol} @ {event.price}"
        )
    
    def _log_fill_event(self, event: FillEvent) -> None:
        """Log fill event."""
        self.logger.info(
            f"FILL: {event.timestamp} {event.order_id} "
            f"{event.quantity} @ {event.fill_price:.4f} "
            f"Commission={event.commission:.2f} Slippage={event.slippage:.4f}"
        )
    
    def enable_event_type(self, event_type: EventType) -> None:
        """Enable logging for specific event type."""
        self._enabled_types.add(event_type)
    
    def disable_event_type(self, event_type: EventType) -> None:
        """Disable logging for specific event type."""
        self._enabled_types.discard(event_type)
    
    def get_event_counts(self) -> Dict[str, int]:
        """Get event counts by type."""
        return {et.value: count for et, count in self.event_counts.items()}
    
    def reset_counts(self) -> None:
        """Reset event counts."""
        self.event_counts.clear()


# Event creation helper functions
def create_market_event(timestamp: datetime, symbol: str, price: float, 
                       volume: float, **kwargs) -> MarketEvent:
    """Create market event with validation."""
    return MarketEvent(
        timestamp=timestamp,
        symbol=symbol,
        price=price,
        volume=volume,
        **kwargs
    )


def create_signal_event(timestamp: datetime, symbol: str, signal_type: str,
                       strength: float, regime_probabilities: Dict[str, float],
                       expected_return: float, risk_estimate: float,
                       **kwargs) -> SignalEvent:
    """Create signal event with validation."""
    if not 0.0 <= strength <= 1.0:
        raise ValueError("Signal strength must be between 0.0 and 1.0")
        
    return SignalEvent(
        timestamp=timestamp,
        symbol=symbol,
        signal_type=signal_type,
        strength=strength,
        regime_probabilities=regime_probabilities,
        expected_return=expected_return,
        risk_estimate=risk_estimate,
        metadata=kwargs
    )


def create_order_event(timestamp: datetime, order_id: str, symbol: str,
                      order_type: str, side: str, quantity: float,
                      **kwargs) -> OrderEvent:
    """Create order event with validation."""
    if quantity <= 0:
        raise ValueError("Order quantity must be positive")
        
    return OrderEvent(
        timestamp=timestamp,
        order_id=order_id,
        symbol=symbol,
        order_type=order_type,
        side=side,
        quantity=quantity,
        **kwargs
    )


def create_fill_event(timestamp: datetime, order_id: str, symbol: str,
                     quantity: float, fill_price: float, commission: float,
                     slippage: float, execution_timestamp: datetime) -> FillEvent:
    """Create fill event with validation."""
    if quantity <= 0:
        raise ValueError("Fill quantity must be positive")
    if fill_price <= 0:
        raise ValueError("Fill price must be positive")
        
    return FillEvent(
        timestamp=timestamp,
        order_id=order_id,
        symbol=symbol,
        quantity=quantity,
        fill_price=fill_price,
        commission=commission,
        slippage=slippage,
        execution_timestamp=execution_timestamp
    )