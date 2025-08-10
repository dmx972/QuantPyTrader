"""
Core Backtesting Engine

This module implements the main BacktestEngine class that orchestrates the
entire backtesting process using an event-driven architecture. The engine
supports both standard backtesting and walk-forward analysis.
"""

from typing import Dict, List, Optional, Any, Type
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json

from .interfaces import (
    IDataHandler, IStrategy, IPortfolio, IExecutionHandler, IRiskManager,
    IPerformanceAnalyzer, BacktestConfig, BacktestResults, EventType
)
from .events import EventQueue, EventProcessor, EventLogger

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Main backtesting engine that coordinates all system components.
    
    The engine follows an event-driven architecture where market data,
    trading signals, orders, and fills are processed as events in 
    chronological order.
    """
    
    def __init__(self, 
                 config: BacktestConfig,
                 data_handler: IDataHandler,
                 strategy: IStrategy,
                 portfolio: IPortfolio,
                 execution_handler: IExecutionHandler,
                 risk_manager: Optional[IRiskManager] = None,
                 performance_analyzer: Optional[IPerformanceAnalyzer] = None):
        """
        Initialize backtesting engine.
        
        Args:
            config: Backtesting configuration
            data_handler: Market data provider
            strategy: Trading strategy
            portfolio: Portfolio manager
            execution_handler: Order execution simulator
            risk_manager: Optional risk management system
            performance_analyzer: Optional performance analyzer
        """
        # Validate configuration
        config_errors = config.validate()
        if config_errors:
            raise ValueError(f"Invalid configuration: {config_errors}")
        
        self.config = config
        self.data_handler = data_handler
        self.strategy = strategy
        self.portfolio = portfolio
        self.execution_handler = execution_handler
        self.risk_manager = risk_manager
        self.performance_analyzer = performance_analyzer
        
        # Event system
        self.event_queue = EventQueue()
        self.event_processor = EventProcessor(self.event_queue)
        self.event_logger = EventLogger(log_level="INFO")
        
        # State tracking
        self.current_datetime = None
        self.is_running = False
        self.results = None
        
        # Performance tracking
        self.portfolio_history = []
        self.trade_history = []
        self.regime_history = []
        self.statistics = {}
        
        # Setup event handlers
        self._setup_event_handlers()
        
        logger.info(f"BacktestEngine initialized for period {config.start_date} to {config.end_date}")
    
    def _setup_event_handlers(self) -> None:
        """Setup event handlers for different event types."""
        
        # Market event handlers
        self.event_processor.register_handler(EventType.MARKET, self._handle_market_event)
        
        # Signal event handlers  
        self.event_processor.register_handler(EventType.SIGNAL, self._handle_signal_event)
        
        # Order event handlers
        self.event_processor.register_handler(EventType.ORDER, self._handle_order_event)
        
        # Fill event handlers
        self.event_processor.register_handler(EventType.FILL, self._handle_fill_event)
        
        # Portfolio update handlers
        self.event_processor.register_handler(EventType.PORTFOLIO_UPDATE, self._handle_portfolio_update)
        
        # Regime change handlers
        self.event_processor.register_handler(EventType.REGIME_CHANGE, self._handle_regime_change)
        
        # Register event logger to capture all events
        for event_type in EventType:
            self.event_processor.register_handler(event_type, self.event_logger.log_event)
    
    def run(self) -> BacktestResults:
        """
        Execute the backtest.
        
        Returns:
            Comprehensive backtest results
        """
        logger.info("Starting backtest execution")
        start_time = datetime.now()
        
        try:
            self.is_running = True
            
            if self.config.walk_forward_enabled:
                results = self._run_walk_forward()
            else:
                results = self._run_standard()
            
            end_time = datetime.now()
            results.start_time = start_time
            results.end_time = end_time
            results.total_runtime = (end_time - start_time).total_seconds()
            
            self.results = results
            logger.info(f"Backtest completed in {results.total_runtime:.2f} seconds")
            
            return results
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            raise
        finally:
            self.is_running = False
    
    def _run_standard(self) -> BacktestResults:
        """
        Execute standard backtest over entire period.
        
        Returns:
            Backtest results
        """
        logger.info("Running standard backtest")
        
        # Initialize results container
        results = BacktestResults(
            config=self.config,
            start_time=datetime.now(),
            end_time=datetime.now(),
            total_runtime=0.0,
            initial_capital=self.config.initial_capital,
            final_capital=0.0,
            total_return=0.0,
            annualized_return=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            calmar_ratio=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            average_win=0.0,
            average_loss=0.0,
            profit_factor=0.0,
            var_95=0.0,
            cvar_95=0.0,
            beta=0.0,
            alpha=0.0,
            information_ratio=0.0,
            portfolio_history=[],
            trade_history=[],
            regime_history=[]
        )
        
        # Main backtest loop
        while self.data_handler.has_data():
            # Update market data
            if not self.data_handler.update_bars():
                break
            
            self.current_datetime = self.data_handler.get_current_datetime()
            
            # Create market event from latest data
            # This would be implemented by the specific data handler
            # For now, we'll simulate the event creation process
            
            # Process all events for this time step
            self._process_time_step()
            
            # Record portfolio state if configured
            if self.config.save_portfolio_history:
                self._record_portfolio_state()
        
        # Calculate final metrics
        results = self._calculate_final_results(results)
        
        return results
    
    def _run_walk_forward(self) -> BacktestResults:
        """
        Execute walk-forward analysis.
        
        Returns:
            Backtest results with walk-forward analysis
        """
        logger.info("Running walk-forward analysis")
        
        training_days = self.config.training_periods
        test_days = self.config.test_periods
        refit_frequency = self.config.refit_frequency
        
        walk_forward_results = []
        current_start = self.config.start_date
        
        while current_start < self.config.end_date:
            # Define training and test periods
            training_end = current_start + timedelta(days=training_days)
            test_start = training_end
            test_end = min(test_start + timedelta(days=test_days), self.config.end_date)
            
            if test_start >= self.config.end_date:
                break
            
            logger.info(f"Walk-forward period: Train {current_start} to {training_end}, "
                       f"Test {test_start} to {test_end}")
            
            # Train strategy on training period
            self._train_strategy_period(current_start, training_end)
            
            # Test on out-of-sample period
            period_results = self._test_strategy_period(test_start, test_end)
            walk_forward_results.append(period_results)
            
            # Move to next period
            current_start = test_start + timedelta(days=refit_frequency)
        
        # Combine walk-forward results
        combined_results = self._combine_walk_forward_results(walk_forward_results)
        combined_results.walk_forward_results = walk_forward_results
        
        return combined_results
    
    def _process_time_step(self) -> None:
        """Process all events for current time step."""
        # Process events until no more events for current time
        events_processed = self.event_processor.process_events_until(
            self.current_datetime + timedelta(microseconds=1)
        )
        
        if events_processed > 0:
            logger.debug(f"Processed {events_processed} events at {self.current_datetime}")
    
    def _handle_market_event(self, event) -> None:
        """Handle market data events."""
        # Update portfolio with latest market data
        self.portfolio.update_market_data(event)
        
        # Update execution handler
        self.execution_handler.set_market_data(event)
        
        # Generate trading signals from strategy
        signals = self.strategy.calculate_signals(event)
        
        # Add signal events to queue
        for signal in signals:
            self.event_queue.put(signal)
    
    def _handle_signal_event(self, event) -> None:
        """Handle trading signal events."""
        # Update strategy state
        self.strategy.update_state(event)
        
        # Generate orders from portfolio
        orders = self.portfolio.update_signal(event)
        
        # Risk management validation
        if self.risk_manager:
            validated_orders = []
            for order in orders:
                if self.risk_manager.validate_order(order, self.portfolio):
                    validated_orders.append(order)
                else:
                    logger.warning(f"Order {order.order_id} rejected by risk manager")
            orders = validated_orders
        
        # Add order events to queue
        for order in orders:
            self.event_queue.put(order)
    
    def _handle_order_event(self, event) -> None:
        """Handle order events."""
        # Execute order through execution handler
        fills = self.execution_handler.execute_order(event)
        
        # Add fill events to queue
        for fill in fills:
            self.event_queue.put(fill)
    
    def _handle_fill_event(self, event) -> None:
        """Handle trade fill events."""
        # Update portfolio with fill
        self.portfolio.update_fill(event)
        
        # Record trade in history
        if self.config.save_trade_history:
            self.trade_history.append({
                'timestamp': event.timestamp,
                'order_id': event.order_id,
                'symbol': event.symbol,
                'quantity': event.quantity,
                'price': event.fill_price,
                'commission': event.commission,
                'slippage': event.slippage
            })
    
    def _handle_portfolio_update(self, event) -> None:
        """Handle portfolio update events."""
        # This could be used for rebalancing, risk checks, etc.
        if self.risk_manager:
            violations = self.risk_manager.check_risk_limits(self.portfolio)
            if violations:
                logger.warning(f"Risk violations detected: {violations}")
    
    def _handle_regime_change(self, event) -> None:
        """Handle regime change events."""
        # Record regime changes for analysis
        if self.config.save_regime_history:
            self.regime_history.append({
                'timestamp': event.timestamp,
                'regime_data': event.data
            })
    
    def _record_portfolio_state(self) -> None:
        """Record current portfolio state."""
        portfolio_summary = self.portfolio.get_portfolio_summary()
        portfolio_summary['timestamp'] = self.current_datetime
        self.portfolio_history.append(portfolio_summary)
    
    def _train_strategy_period(self, start_date: datetime, end_date: datetime) -> None:
        """Train strategy on specified period."""
        logger.info(f"Training strategy from {start_date} to {end_date}")
        # Implementation depends on specific strategy requirements
        # For now, we'll assume the strategy handles its own training
        pass
    
    def _test_strategy_period(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Test strategy on specified period."""
        logger.info(f"Testing strategy from {start_date} to {end_date}")
        
        # Create temporary config for this period
        period_config = BacktestConfig(
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.config.initial_capital
        )
        
        # Run backtest for this period
        # This is a simplified implementation
        return {
            'start_date': start_date,
            'end_date': end_date,
            'return': 0.0,  # Placeholder
            'sharpe': 0.0,  # Placeholder
            'max_drawdown': 0.0  # Placeholder
        }
    
    def _combine_walk_forward_results(self, period_results: List[Dict[str, Any]]) -> BacktestResults:
        """Combine results from walk-forward periods."""
        # This is a placeholder implementation
        # Real implementation would aggregate metrics across periods
        
        return BacktestResults(
            config=self.config,
            start_time=datetime.now(),
            end_time=datetime.now(),
            total_runtime=0.0,
            initial_capital=self.config.initial_capital,
            final_capital=self.config.initial_capital,  # Placeholder
            total_return=0.0,
            annualized_return=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            calmar_ratio=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            average_win=0.0,
            average_loss=0.0,
            profit_factor=0.0,
            var_95=0.0,
            cvar_95=0.0,
            beta=0.0,
            alpha=0.0,
            information_ratio=0.0,
            portfolio_history=self.portfolio_history,
            trade_history=self.trade_history,
            regime_history=self.regime_history
        )
    
    def _calculate_final_results(self, results: BacktestResults) -> BacktestResults:
        """Calculate final performance metrics."""
        
        # Get final portfolio value
        results.final_capital = self.portfolio.get_current_portfolio_value()
        
        # Calculate basic returns
        results.total_return = (results.final_capital - results.initial_capital) / results.initial_capital
        
        # Calculate annualized return
        days = (self.config.end_date - self.config.start_date).days
        if days > 0:
            results.annualized_return = ((results.final_capital / results.initial_capital) ** (365.0 / days)) - 1.0
        
        # Use performance analyzer if available
        if self.performance_analyzer:
            detailed_metrics = self.performance_analyzer.calculate_metrics(self.portfolio_history)
            
            # Update results with detailed metrics
            results.volatility = detailed_metrics.get('volatility', 0.0)
            results.sharpe_ratio = detailed_metrics.get('sharpe_ratio', 0.0)
            results.max_drawdown = detailed_metrics.get('max_drawdown', 0.0)
            results.calmar_ratio = detailed_metrics.get('calmar_ratio', 0.0)
            
            # Trade statistics
            if self.trade_history:
                results.total_trades = len(self.trade_history)
                # Additional trade statistics would be calculated here
        
        # Store historical data
        results.portfolio_history = self.portfolio_history
        results.trade_history = self.trade_history
        results.regime_history = self.regime_history
        
        return results
    
    def save_results(self, filepath: str) -> None:
        """
        Save backtest results to file.
        
        Args:
            filepath: Path to save results
        """
        if not self.results:
            raise ValueError("No results to save. Run backtest first.")
        
        # Convert results to serializable format
        results_dict = {
            'config': {
                'start_date': self.config.start_date.isoformat(),
                'end_date': self.config.end_date.isoformat(),
                'initial_capital': self.config.initial_capital,
                # Add other config fields as needed
            },
            'performance': self.results.get_summary(),
            'portfolio_history': self.portfolio_history,
            'trade_history': self.trade_history,
            'regime_history': self.regime_history,
            'statistics': self.statistics
        }
        
        # Save to file
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filepath}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get runtime statistics."""
        return {
            'engine_stats': {
                'current_time': self.current_datetime,
                'is_running': self.is_running,
                'portfolio_records': len(self.portfolio_history),
                'trade_records': len(self.trade_history),
                'regime_records': len(self.regime_history)
            },
            'event_stats': self.event_processor.get_statistics(),
            'event_counts': self.event_logger.get_event_counts()
        }