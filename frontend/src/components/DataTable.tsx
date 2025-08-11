/**
 * DataTable Component
 * 
 * High-performance data table with virtualization for large datasets,
 * sorting, filtering, and trading-specific formatting.
 */

import React, { useMemo, useState, useCallback } from 'react';
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TableSortLabel,
  TextField,
  Box,
  Typography,
  Chip,
  IconButton,
  Menu,
  MenuItem,
  Checkbox,
  FormControlLabel
} from '@mui/material';
import { styled } from '@mui/material/styles';
import {
  FilterList,
  MoreVert,
  TrendingUp,
  TrendingDown,
  TrendingFlat
} from '@mui/icons-material';
import { FixedSizeList as List } from 'react-window';
import { useQuantTheme } from '../theme';

// Types
export interface ColumnDef<T = any> {
  id: string;
  label: string;
  accessor: keyof T | ((row: T) => any);
  type?: 'text' | 'number' | 'currency' | 'percentage' | 'date' | 'regime' | 'trend';
  sortable?: boolean;
  filterable?: boolean;
  width?: number;
  minWidth?: number;
  align?: 'left' | 'center' | 'right';
  format?: (value: any, row: T) => React.ReactNode;
  cellProps?: (value: any, row: T) => Record<string, any>;
}

export interface DataTableProps<T = any> {
  data: T[];
  columns: ColumnDef<T>[];
  height?: number;
  rowHeight?: number;
  loading?: boolean;
  emptyMessage?: string;
  sortable?: boolean;
  filterable?: boolean;
  virtualizeRows?: boolean;
  stickyHeader?: boolean;
  onRowClick?: (row: T, index: number) => void;
  onSort?: (columnId: string, direction: 'asc' | 'desc') => void;
  className?: string;
}

// Styled Components
const StyledTableContainer = styled(TableContainer)(({ theme }) => ({
  border: `1px solid ${theme.palette.divider}`,
  borderRadius: theme.spacing(1),
  backgroundColor: theme.palette.background.paper,
  '& .MuiTable-root': {
    minWidth: '100%',
  },
}));

const StyledTableHead = styled(TableHead)(({ theme }) => ({
  backgroundColor: theme.palette.background.default,
  '& .MuiTableCell-head': {
    fontWeight: 600,
    fontSize: '0.875rem',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
    borderBottom: `2px solid ${theme.palette.divider}`,
    padding: theme.spacing(1.5, 2),
  },
}));

const StyledTableRow = styled(TableRow)(({ theme }) => ({
  cursor: 'pointer',
  transition: 'background-color 0.2s ease',
  '&:hover': {
    backgroundColor: theme.palette.action.hover,
  },
  '&:nth-of-type(even)': {
    backgroundColor: theme.palette.action.selected,
  },
}));

const FilterHeader = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  gap: theme.spacing(2),
  padding: theme.spacing(1, 2),
  borderBottom: `1px solid ${theme.palette.divider}`,
  backgroundColor: theme.palette.background.default,
}));

const CellContent = styled(Box)<{ align?: 'left' | 'center' | 'right' }>(({ align = 'left' }) => ({
  display: 'flex',
  alignItems: 'center',
  justifyContent: align === 'left' ? 'flex-start' : align === 'right' ? 'flex-end' : 'center',
  gap: '4px',
}));

// Utility functions
const formatValue = (value: any, type: string = 'text', theme: any): React.ReactNode => {
  if (value === null || value === undefined || value === '') {
    return <span style={{ color: theme.colors.text.tertiary }}>—</span>;
  }

  switch (type) {
    case 'number':
      return (
        <span style={{ fontFamily: theme.fonts.mono }}>
          {typeof value === 'number' ? value.toLocaleString() : value}
        </span>
      );
      
    case 'currency':
      return (
        <span style={{ fontFamily: theme.fonts.mono }}>
          {typeof value === 'number' 
            ? new Intl.NumberFormat('en-US', {
                style: 'currency',
                currency: 'USD',
                minimumFractionDigits: 2,
                maximumFractionDigits: 2,
              }).format(value)
            : value
          }
        </span>
      );
      
    case 'percentage':
      return (
        <span 
          style={{ 
            fontFamily: theme.fonts.mono,
            color: value > 0 ? theme.colors.success : value < 0 ? theme.colors.error : theme.colors.text.primary
          }}
        >
          {typeof value === 'number' ? `${value >= 0 ? '+' : ''}${value.toFixed(2)}%` : value}
        </span>
      );
      
    case 'date':
      return (
        <span style={{ fontFamily: theme.fonts.mono }}>
          {value instanceof Date ? value.toLocaleDateString() : value}
        </span>
      );
      
    case 'regime':
      const regimeColors: Record<string, string> = {
        bull: theme.colors.regime.bull,
        bear: theme.colors.regime.bear,
        sideways: theme.colors.regime.sideways,
        crisis: theme.colors.regime.crisis,
      };
      
      return (
        <Chip
          label={value}
          size="small"
          sx={{
            backgroundColor: `${regimeColors[value?.toLowerCase()] || theme.colors.text.secondary}20`,
            color: regimeColors[value?.toLowerCase()] || theme.colors.text.secondary,
            fontWeight: 600,
            fontSize: '0.75rem',
            border: `1px solid ${regimeColors[value?.toLowerCase()] || theme.colors.text.secondary}40`,
          }}
        />
      );
      
    case 'trend':
      const trendValue = parseFloat(value);
      const TrendIcon = trendValue > 0 ? TrendingUp : trendValue < 0 ? TrendingDown : TrendingFlat;
      const trendColor = trendValue > 0 ? theme.colors.success : trendValue < 0 ? theme.colors.error : theme.colors.text.secondary;
      
      return (
        <Box display="flex" alignItems="center" gap={0.5}>
          <TrendIcon sx={{ fontSize: '1rem', color: trendColor }} />
          <span style={{ fontFamily: theme.fonts.mono, color: trendColor }}>
            {trendValue > 0 ? '+' : ''}{trendValue.toFixed(2)}%
          </span>
        </Box>
      );
      
    default:
      return value?.toString() || '';
  }
};

const getColumnValue = <T,>(row: T, accessor: keyof T | ((row: T) => any)): any => {
  if (typeof accessor === 'function') {
    return accessor(row);
  }
  return row[accessor];
};

// Row component for virtualization
interface VirtualRowProps {
  index: number;
  style: React.CSSProperties;
  data: {
    items: any[];
    columns: ColumnDef[];
    onRowClick?: (row: any, index: number) => void;
    theme: any;
  };
}

const VirtualRow: React.FC<VirtualRowProps> = ({ index, style, data }) => {
  const { items, columns, onRowClick, theme } = data;
  const row = items[index];

  const handleClick = () => {
    if (onRowClick) {
      onRowClick(row, index);
    }
  };

  return (
    <div style={style}>
      <StyledTableRow onClick={handleClick}>
        {columns.map((column) => {
          const value = getColumnValue(row, column.accessor);
          const cellProps = column.cellProps ? column.cellProps(value, row) : {};
          
          return (
            <TableCell
              key={column.id}
              align={column.align || 'left'}
              sx={{
                minWidth: column.minWidth,
                width: column.width,
                padding: '8px 16px',
              }}
              {...cellProps}
            >
              <CellContent align={column.align}>
                {column.format 
                  ? column.format(value, row)
                  : formatValue(value, column.type, theme)
                }
              </CellContent>
            </TableCell>
          );
        })}
      </StyledTableRow>
    </div>
  );
};

/**
 * DataTable Component
 * 
 * High-performance data table with virtualization, sorting, and filtering
 * optimized for financial data display.
 */
export const DataTable = <T,>({
  data,
  columns,
  height = 400,
  rowHeight = 56,
  loading = false,
  emptyMessage = 'No data available',
  sortable = true,
  filterable = true,
  virtualizeRows = true,
  stickyHeader = true,
  onRowClick,
  onSort,
  className
}: DataTableProps<T>) => {
  const { theme } = useQuantTheme();
  const [sortConfig, setSortConfig] = useState<{ key: string; direction: 'asc' | 'desc' } | null>(null);
  const [filters, setFilters] = useState<Record<string, string>>({});
  const [columnMenuAnchor, setColumnMenuAnchor] = useState<null | HTMLElement>(null);

  // Handle sorting
  const handleSort = useCallback((columnId: string) => {
    if (!sortable) return;

    const direction = 
      sortConfig?.key === columnId && sortConfig.direction === 'asc' ? 'desc' : 'asc';
    
    setSortConfig({ key: columnId, direction });
    
    if (onSort) {
      onSort(columnId, direction);
    }
  }, [sortConfig, sortable, onSort]);

  // Process data with sorting and filtering
  const processedData = useMemo(() => {
    let result = [...data];

    // Apply filters
    Object.entries(filters).forEach(([columnId, filterValue]) => {
      if (filterValue) {
        const column = columns.find(col => col.id === columnId);
        if (column) {
          result = result.filter(row => {
            const value = getColumnValue(row, column.accessor);
            return value?.toString().toLowerCase().includes(filterValue.toLowerCase());
          });
        }
      }
    });

    // Apply sorting
    if (sortConfig && !onSort) {
      const column = columns.find(col => col.id === sortConfig.key);
      if (column) {
        result.sort((a, b) => {
          const aValue = getColumnValue(a, column.accessor);
          const bValue = getColumnValue(b, column.accessor);
          
          if (aValue === bValue) return 0;
          
          let comparison = 0;
          if (typeof aValue === 'number' && typeof bValue === 'number') {
            comparison = aValue - bValue;
          } else {
            comparison = String(aValue).localeCompare(String(bValue));
          }
          
          return sortConfig.direction === 'asc' ? comparison : -comparison;
        });
      }
    }

    return result;
  }, [data, columns, filters, sortConfig, onSort]);

  const handleFilterChange = (columnId: string, value: string) => {
    setFilters(prev => ({
      ...prev,
      [columnId]: value
    }));
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height={height}>
        <Typography color="text.secondary">Loading...</Typography>
      </Box>
    );
  }

  if (processedData.length === 0) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height={height}>
        <Typography color="text.secondary">{emptyMessage}</Typography>
      </Box>
    );
  }

  return (
    <StyledTableContainer className={className} sx={{ height }}>
      {/* Filters */}
      {filterable && (
        <FilterHeader>
          {columns
            .filter(col => col.filterable !== false)
            .map(column => (
              <TextField
                key={`filter-${column.id}`}
                label={`Filter ${column.label}`}
                variant="outlined"
                size="small"
                value={filters[column.id] || ''}
                onChange={(e) => handleFilterChange(column.id, e.target.value)}
                sx={{ minWidth: 150 }}
              />
            ))
          }
        </FilterHeader>
      )}

      <Table stickyHeader={stickyHeader}>
        <StyledTableHead>
          <TableRow>
            {columns.map((column) => (
              <TableCell
                key={column.id}
                align={column.align || 'left'}
                sx={{
                  minWidth: column.minWidth,
                  width: column.width,
                }}
                sortDirection={sortConfig?.key === column.id ? sortConfig.direction : false}
              >
                {sortable && column.sortable !== false ? (
                  <TableSortLabel
                    active={sortConfig?.key === column.id}
                    direction={sortConfig?.key === column.id ? sortConfig.direction : 'asc'}
                    onClick={() => handleSort(column.id)}
                    sx={{ fontWeight: 'inherit' }}
                  >
                    {column.label}
                  </TableSortLabel>
                ) : (
                  column.label
                )}
              </TableCell>
            ))}
          </TableRow>
        </StyledTableHead>

        <TableBody>
          {virtualizeRows ? (
            <TableRow>
              <TableCell colSpan={columns.length} sx={{ padding: 0 }}>
                <List
                  height={height - 120} // Account for header and filters
                  width="100%"
                  itemCount={processedData.length}
                  itemSize={rowHeight}
                  itemData={{
                    items: processedData,
                    columns,
                    onRowClick,
                    theme
                  }}
                >
                  {VirtualRow}
                </List>
              </TableCell>
            </TableRow>
          ) : (
            processedData.map((row, index) => (
              <StyledTableRow
                key={index}
                onClick={() => onRowClick?.(row, index)}
              >
                {columns.map((column) => {
                  const value = getColumnValue(row, column.accessor);
                  const cellProps = column.cellProps ? column.cellProps(value, row) : {};
                  
                  return (
                    <TableCell
                      key={column.id}
                      align={column.align || 'left'}
                      sx={{
                        minWidth: column.minWidth,
                        width: column.width,
                      }}
                      {...cellProps}
                    >
                      <CellContent align={column.align}>
                        {column.format 
                          ? column.format(value, row)
                          : formatValue(value, column.type, theme)
                        }
                      </CellContent>
                    </TableCell>
                  );
                })}
              </StyledTableRow>
            ))
          )}
        </TableBody>
      </Table>
    </StyledTableContainer>
  );
};

export default DataTable;