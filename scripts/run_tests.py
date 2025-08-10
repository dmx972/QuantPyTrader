#!/usr/bin/env python3
"""
QuantPyTrader Test Runner

Comprehensive test execution script with various test suites and reporting options.
Designed for both local development and CI/CD environments.
"""

import argparse
import os
import sys
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import json


class TestRunner:
    """Comprehensive test runner for QuantPyTrader."""
    
    def __init__(self, project_root: Optional[Path] = None):
        """Initialize test runner."""
        self.project_root = project_root or Path(__file__).parent.parent
        self.test_dir = self.project_root / 'tests'
        self.results = {}
        self.start_time = None
        self.total_duration = 0
    
    def run_command(self, cmd: List[str], description: str, 
                   timeout: Optional[int] = None) -> tuple[bool, str]:
        """Run a command and capture output."""
        print(f"\n{'='*60}")
        print(f"Running: {description}")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            duration = time.time() - start_time
            success = result.returncode == 0
            
            print(f"Duration: {duration:.2f}s")
            print(f"Return code: {result.returncode}")
            
            if result.stdout:
                print(f"\nSTDOUT:\n{result.stdout}")
            if result.stderr:
                print(f"\nSTDERR:\n{result.stderr}")
            
            self.results[description] = {
                'success': success,
                'duration': duration,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
            return success, result.stdout + result.stderr
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            print(f"TIMEOUT after {duration:.2f}s")
            self.results[description] = {
                'success': False,
                'duration': duration,
                'return_code': -1,
                'error': 'Timeout expired'
            }
            return False, "Timeout expired"
        
        except Exception as e:
            duration = time.time() - start_time
            print(f"ERROR: {e}")
            self.results[description] = {
                'success': False,
                'duration': duration,
                'return_code': -1,
                'error': str(e)
            }
            return False, str(e)
    
    def run_unit_tests(self, coverage: bool = True, parallel: bool = False) -> bool:
        """Run unit tests."""
        cmd = ['python', '-m', 'pytest', 'tests/']
        
        if coverage:
            cmd.extend(['--cov=backtesting', '--cov-report=xml', '--cov-report=html'])
        
        if parallel:
            cmd.extend(['-n', 'auto'])
        
        cmd.extend([
            '-v', '--tb=short',
            '-m', 'not slow and not performance',
            '--timeout=300'
        ])
        
        success, _ = self.run_command(cmd, "Unit Tests", timeout=600)
        return success
    
    def run_integration_tests(self) -> bool:
        """Run integration tests."""
        cmd = [
            'python', '-m', 'pytest', 
            'tests/test_integration.py',
            'tests/test_simple_integration.py', 
            'tests/test_simple_end_to_end.py',
            '-v', '--tb=short',
            '--timeout=600'
        ]
        
        success, _ = self.run_command(cmd, "Integration Tests", timeout=900)
        return success
    
    def run_performance_tests(self) -> bool:
        """Run performance benchmarks."""
        cmd = [
            'python', '-m', 'pytest',
            'tests/test_performance_benchmarks.py',
            '-v', '--tb=short',
            '-m', 'performance',
            '--timeout=1200'
        ]
        
        success, _ = self.run_command(cmd, "Performance Tests", timeout=1500)
        return success
    
    def run_end_to_end_tests(self) -> bool:
        """Run end-to-end workflow tests."""
        cmd = [
            'python', '-m', 'pytest',
            'tests/test_end_to_end_workflows.py',
            'tests/test_data_generator_functionality.py',
            '-v', '--tb=short',
            '--timeout=900'
        ]
        
        success, _ = self.run_command(cmd, "End-to-End Tests", timeout=1200)
        return success
    
    def run_code_quality_checks(self) -> bool:
        """Run code quality checks."""
        checks = [
            (['python', '-m', 'black', '--check', '--diff', '.'], "Black Formatting Check"),
            (['python', '-m', 'isort', '--check-only', '--diff', '.'], "Import Sorting Check"),
            (['python', '-m', 'flake8', '.', '--max-line-length=120'], "Flake8 Linting"),
        ]
        
        all_passed = True
        for cmd, description in checks:
            try:
                success, _ = self.run_command(cmd, description, timeout=120)
                all_passed = all_passed and success
            except Exception:
                # Code quality checks are optional - don't fail the entire suite
                print(f"Skipping {description} - tool not available")
        
        return all_passed
    
    def run_security_checks(self) -> bool:
        """Run security checks."""
        checks = [
            (['python', '-m', 'bandit', '-r', 'backtesting/', '--severity-level', 'medium'], 
             "Bandit Security Check"),
            (['python', '-m', 'safety', 'check'], "Safety Dependency Check"),
        ]
        
        all_passed = True
        for cmd, description in checks:
            try:
                success, _ = self.run_command(cmd, description, timeout=180)
                # Security checks are warnings - don't fail build
                print(f"{description}: {'PASSED' if success else 'WARNINGS'}")
            except Exception:
                print(f"Skipping {description} - tool not available")
        
        return all_passed
    
    def validate_imports(self) -> bool:
        """Validate critical imports work."""
        import_tests = [
            "from backtesting.results.storage import ResultsStorage",
            "from backtesting.export import quick_export", 
            "from backtesting.results.report_generator import ReportGenerator",
            "from tests.test_utils import create_sample_portfolio_history",
            "from tests.test_data_generators import generate_comprehensive_test_dataset"
        ]
        
        print(f"\n{'='*60}")
        print("Validating Critical Imports")
        print(f"{'='*60}")
        
        all_passed = True
        for import_stmt in import_tests:
            try:
                exec(import_stmt)
                print(f"✓ {import_stmt}")
            except Exception as e:
                print(f"✗ {import_stmt}: {e}")
                all_passed = False
        
        self.results["Import Validation"] = {
            'success': all_passed,
            'duration': 0.1
        }
        
        return all_passed
    
    def run_smoke_tests(self) -> bool:
        """Run quick smoke tests to validate basic functionality."""
        print(f"\n{'='*60}")
        print("Running Smoke Tests")
        print(f"{'='*60}")
        
        try:
            # Test basic functionality
            from backtesting.results.storage import ResultsStorage
            from tests.test_utils import create_sample_portfolio_history
            from datetime import date
            import tempfile
            
            # Create temporary storage
            with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
                storage = ResultsStorage(tmp.name)
                
                # Create test backtest
                backtest_id = storage.create_backtest_session(
                    strategy_name="Smoke Test",
                    strategy_type="BE_EMA_MMCUKF", 
                    backtest_name="Quick Validation",
                    start_date=date(2024, 1, 1),
                    end_date=date(2024, 1, 7)
                )
                
                # Test data creation
                portfolio_history = create_sample_portfolio_history(
                    date(2024, 1, 1), date(2024, 1, 7), 50000.0
                )
                
                # Store results
                results = {
                    'portfolio_history': portfolio_history,
                    'performance': {
                        'total_return': 0.02, 'volatility': 0.1, 
                        'sharpe_ratio': 0.8, 'max_drawdown': -0.01
                    }
                }
                
                storage.store_backtest_results(backtest_id, results)
                
                # Verify retrieval
                summary = storage.get_backtest_summary(backtest_id)
                portfolio_data = storage.get_portfolio_data(backtest_id)
                
                assert summary is not None
                assert len(portfolio_data) > 0
                
                print("✓ Storage functionality working")
                print("✓ Data generation working")
                print("✓ Basic workflow functional")
                
                # Cleanup
                os.unlink(tmp.name)
                
                self.results["Smoke Tests"] = {
                    'success': True,
                    'duration': 1.0
                }
                
                return True
                
        except Exception as e:
            print(f"✗ Smoke tests failed: {e}")
            self.results["Smoke Tests"] = {
                'success': False,
                'duration': 1.0,
                'error': str(e)
            }
            return False
    
    def generate_report(self, output_file: Optional[Path] = None) -> Dict:
        """Generate test execution report."""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r.get('success', False))
        total_duration = sum(r.get('duration', 0) for r in self.results.values())
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': total_tests - passed_tests,
                'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
                'total_duration': total_duration
            },
            'results': self.results
        }
        
        # Print summary
        print(f"\n{'='*70}")
        print("TEST EXECUTION SUMMARY")
        print(f"{'='*70}")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {report['summary']['success_rate']:.1f}%")
        print(f"Total Duration: {total_duration:.2f}s")
        print()
        
        for test_name, result in self.results.items():
            status = "✓ PASS" if result.get('success', False) else "✗ FAIL"
            duration = result.get('duration', 0)
            print(f"{status} {test_name} ({duration:.2f}s)")
        
        print(f"{'='*70}")
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Report saved to: {output_file}")
        
        return report
    
    def run_full_suite(self, include_performance: bool = False, 
                      include_extended: bool = False) -> bool:
        """Run the complete test suite."""
        self.start_time = time.time()
        
        print(f"{'='*70}")
        print("QUANTPYTRADER COMPREHENSIVE TEST SUITE")
        print(f"{'='*70}")
        print(f"Start Time: {datetime.now().isoformat()}")
        print(f"Project Root: {self.project_root}")
        print(f"Include Performance Tests: {include_performance}")
        print(f"Include Extended Tests: {include_extended}")
        print(f"{'='*70}")
        
        # Test sequence
        test_sequence = [
            ("Import Validation", self.validate_imports),
            ("Smoke Tests", self.run_smoke_tests),
            ("Code Quality", self.run_code_quality_checks),
            ("Security Checks", self.run_security_checks),
            ("Unit Tests", lambda: self.run_unit_tests(coverage=True)),
            ("Integration Tests", self.run_integration_tests),
        ]
        
        if include_performance:
            test_sequence.append(("Performance Tests", self.run_performance_tests))
        
        if include_extended:
            test_sequence.append(("End-to-End Tests", self.run_end_to_end_tests))
        
        # Execute tests
        overall_success = True
        for test_name, test_func in test_sequence:
            print(f"\n🔍 Starting: {test_name}")
            try:
                success = test_func()
                if not success:
                    print(f"❌ {test_name} FAILED")
                    overall_success = False
                else:
                    print(f"✅ {test_name} PASSED")
            except Exception as e:
                print(f"💥 {test_name} ERROR: {e}")
                overall_success = False
        
        # Generate final report
        self.total_duration = time.time() - self.start_time
        report = self.generate_report()
        
        return overall_success


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='QuantPyTrader Test Runner')
    
    parser.add_argument('--unit', action='store_true', 
                       help='Run unit tests only')
    parser.add_argument('--integration', action='store_true',
                       help='Run integration tests only') 
    parser.add_argument('--performance', action='store_true',
                       help='Run performance tests')
    parser.add_argument('--end-to-end', action='store_true',
                       help='Run end-to-end tests')
    parser.add_argument('--smoke', action='store_true',
                       help='Run smoke tests only')
    parser.add_argument('--quality', action='store_true',
                       help='Run code quality checks only')
    parser.add_argument('--full', action='store_true',
                       help='Run full test suite')
    parser.add_argument('--extended', action='store_true',
                       help='Include extended tests in full suite')
    parser.add_argument('--report', type=str,
                       help='Save test report to file')
    parser.add_argument('--no-coverage', action='store_true',
                       help='Skip coverage reporting')
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = TestRunner()
    
    success = True
    
    # Run specific test types
    if args.unit:
        success = runner.run_unit_tests(coverage=not args.no_coverage)
    elif args.integration:
        success = runner.run_integration_tests()
    elif args.performance:
        success = runner.run_performance_tests()
    elif args.end_to_end:
        success = runner.run_end_to_end_tests()
    elif args.smoke:
        success = runner.run_smoke_tests()
    elif args.quality:
        success = runner.run_code_quality_checks()
    elif args.full:
        success = runner.run_full_suite(
            include_performance=args.performance,
            include_extended=args.extended
        )
    else:
        # Default: run essential tests
        success = runner.run_full_suite(
            include_performance=False,
            include_extended=False
        )
    
    # Generate report
    report_file = Path(args.report) if args.report else None
    runner.generate_report(report_file)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()