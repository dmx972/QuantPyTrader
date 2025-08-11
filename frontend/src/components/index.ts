/**
 * QuantPyTrader Components - Main Export File
 * 
 * Centralized export for all design system components
 */

export { TradingCard } from './TradingCard';
export type { TradingCardProps } from './TradingCard';

export { MetricDisplay } from './MetricDisplay';
export type { MetricDisplayProps } from './MetricDisplay';

export { RegimeGauge } from './RegimeGauge';
export type { RegimeGaugeProps } from './RegimeGauge';

export { DataTable } from './DataTable';
export type { DataTableProps, ColumnDef } from './DataTable';

// Component exports for backwards compatibility with existing imports
export { default as TradingCardDefault } from './TradingCard';
export { default as MetricDisplayDefault } from './MetricDisplay';
export { default as RegimeGaugeDefault } from './RegimeGauge';
export { default as DataTableDefault } from './DataTable';