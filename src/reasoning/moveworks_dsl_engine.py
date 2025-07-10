"""
Moveworks DSL Validation Engine

Implements DSL rule validation for slots following Moveworks patterns:
- Basic comparisons: value > 0, value == "active"
- Time functions: $PARSE_TIME(value) > $TIME()
- String functions: $CONTAINS(value, "text")
- Array functions: value IN ["option1", "option2"]
- User functions: user.role == "manager"

This follows the official Moveworks DSL syntax.
"""

import re
import ast
import operator
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import json


@dataclass
class DSLValidationResult:
    """Result of DSL validation."""
    is_valid: bool
    error_message: str = ""
    parsed_rule: str = ""
    execution_details: Dict[str, Any] = None


class MoveworksDSLEngine:
    """
    Moveworks DSL validation engine for slot validation rules.
    Supports the DSL patterns found in Moveworks documentation.
    """
    
    def __init__(self):
        self.operators = {
            '>': operator.gt,
            '<': operator.lt,
            '>=': operator.ge,
            '<=': operator.le,
            '==': operator.eq,
            '!=': operator.ne,
            'IN': lambda x, y: x in y,
            'NOT IN': lambda x, y: x not in y
        }
        
        self.functions = {
            '$TIME': self._time_function,
            '$PARSE_TIME': self._parse_time_function,
            '$CONTAINS': self._contains_function,
            '$LENGTH': self._length_function,
            '$UPPER': self._upper_function,
            '$LOWER': self._lower_function,
            '$TRIM': self._trim_function
        }

        # Create function mapping without $ prefix for evaluation
        self.eval_functions = {
            'TIME': self._time_function,
            'PARSE_TIME': self._parse_time_function,
            'CONTAINS': self._contains_function,
            'LENGTH': self._length_function,
            'UPPER': self._upper_function,
            'LOWER': self._lower_function,
            'TRIM': self._trim_function
        }
    
    def validate_rule(self, rule: str, value: Any, context: Dict[str, Any] = None) -> DSLValidationResult:
        """
        Validate a value against a DSL rule.
        
        Args:
            rule: DSL rule string (e.g., "value > 0", "$PARSE_TIME(value) > $TIME()")
            value: Value to validate
            context: Additional context (user attributes, etc.)
        
        Returns:
            DSLValidationResult with validation outcome
        """
        if not rule or rule.strip() == "":
            return DSLValidationResult(is_valid=True, parsed_rule="No rule specified")
        
        try:
            # Prepare evaluation context
            eval_context = {
                'value': value,
                'user': context.get('user', {}) if context else {},
                **self.eval_functions  # Use functions without $ prefix
            }
            
            # Parse and execute the rule
            parsed_rule = self._parse_rule(rule)
            result = self._evaluate_rule(parsed_rule, eval_context)
            
            return DSLValidationResult(
                is_valid=bool(result),
                parsed_rule=parsed_rule,
                execution_details={'original_rule': rule, 'result': result}
            )
            
        except Exception as e:
            return DSLValidationResult(
                is_valid=False,
                error_message=f"DSL validation error: {str(e)}",
                parsed_rule=rule
            )
    
    def _parse_rule(self, rule: str) -> str:
        """Parse DSL rule into Python-evaluable expression."""
        # Handle IN operator
        rule = re.sub(r'\bIN\b', 'in', rule)
        rule = re.sub(r'\bNOT IN\b', 'not in', rule)

        # Handle function calls - remove the $ prefix
        for func_name in self.functions.keys():
            # Replace $FUNCTION_NAME( with FUNCTION_NAME(
            pattern = rf'\{re.escape(func_name)}\('
            replacement = f'{func_name[1:]}('  # Remove the $ prefix
            rule = re.sub(pattern, replacement, rule)

        return rule
    
    def _evaluate_rule(self, rule: str, context: Dict[str, Any]) -> bool:
        """Safely evaluate parsed DSL rule."""
        # Use eval with restricted context for safety
        # In production, consider using a proper expression parser
        try:
            return eval(rule, {"__builtins__": {}}, context)
        except Exception as e:
            raise ValueError(f"Rule evaluation failed: {str(e)}")
    
    # DSL Functions
    def _time_function(self) -> datetime:
        """$TIME() - Returns current datetime."""
        return datetime.now()
    
    def _parse_time_function(self, time_str: str) -> datetime:
        """$PARSE_TIME(value) - Parse time string to datetime."""
        if isinstance(time_str, datetime):
            return time_str

        # Try common time formats
        formats = [
            "%Y-%m-%d",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%m/%d/%Y",
            "%d/%m/%Y",
            "%Y-%m-%dT%H:%M:%S.%f"
        ]

        for fmt in formats:
            try:
                return datetime.strptime(str(time_str), fmt)
            except ValueError:
                continue

        # Try parsing ISO format
        try:
            from dateutil.parser import parse
            return parse(str(time_str))
        except ImportError:
            pass
        except Exception:
            pass

        raise ValueError(f"Cannot parse time: {time_str}")
    
    def _contains_function(self, text: str, substring: str) -> bool:
        """$CONTAINS(text, substring) - Check if text contains substring."""
        return str(substring) in str(text)
    
    def _length_function(self, value: Any) -> int:
        """$LENGTH(value) - Get length of value."""
        if hasattr(value, '__len__'):
            return len(value)
        return len(str(value))
    
    def _upper_function(self, text: str) -> str:
        """$UPPER(text) - Convert to uppercase."""
        return str(text).upper()
    
    def _lower_function(self, text: str) -> str:
        """$LOWER(text) - Convert to lowercase."""
        return str(text).lower()
    
    def _trim_function(self, text: str) -> str:
        """$TRIM(text) - Remove whitespace."""
        return str(text).strip()


class MoveworksDSLValidator:
    """High-level DSL validator for Moveworks slots."""
    
    def __init__(self):
        self.engine = MoveworksDSLEngine()
    
    def validate_slot_value(self, slot_name: str, value: Any, validation_rule: str, 
                          user_context: Dict[str, Any] = None) -> DSLValidationResult:
        """
        Validate slot value using DSL rule.
        
        Examples of validation rules:
        - "value > 0" - Number must be positive
        - "value == TRUE" - Boolean must be true
        - "$PARSE_TIME(value) > $TIME()" - Date must be in future
        - "user.role == 'manager'" - User must be manager
        - "value IN ['option1', 'option2']" - Value must be in list
        """
        context = {
            'slot_name': slot_name,
            'user': user_context or {}
        }
        
        return self.engine.validate_rule(validation_rule, value, context)
    
    def create_common_validation_rules(self) -> Dict[str, str]:
        """Return common DSL validation rules."""
        return {
            # Numeric validations
            "positive_number": "value > 0",
            "non_negative": "value >= 0",
            "percentage": "value >= 0 and value <= 100",
            
            # String validations
            "non_empty_string": "$LENGTH(value) > 0",
            "email_format": "$CONTAINS(value, '@')",
            "min_length_3": "$LENGTH(value) >= 3",
            
            # Date validations
            "future_date": "$PARSE_TIME(value) > $TIME()",
            "not_past_date": "$PARSE_TIME(value) >= $TIME()",
            
            # Boolean validations
            "must_be_true": "value == True",
            "must_acknowledge": "value == True",
            
            # Choice validations
            "valid_priority": "value IN ['Low', 'Medium', 'High', 'Critical']",
            "valid_status": "value IN ['New', 'In Progress', 'Resolved', 'Closed']",
            
            # User validations
            "manager_only": "user.role == 'manager'",
            "it_department": "user.department == 'IT'",
            "active_user": "user.status == 'active'"
        }
    
    def test_validation_rules(self) -> Dict[str, Any]:
        """Test common validation rules with sample data."""
        test_cases = [
            # Positive number
            {"rule": "value > 0", "value": 5, "expected": True},
            {"rule": "value > 0", "value": -1, "expected": False},
            
            # Future date
            {"rule": "$PARSE_TIME(value) > $TIME()", "value": "2025-12-31", "expected": True},
            {"rule": "$PARSE_TIME(value) > $TIME()", "value": "2020-01-01", "expected": False},
            
            # String contains
            {"rule": "$CONTAINS(value, '@')", "value": "user@example.com", "expected": True},
            {"rule": "$CONTAINS(value, '@')", "value": "invalid-email", "expected": False},
            
            # IN operator
            {"rule": "value IN ['A', 'B', 'C']", "value": "B", "expected": True},
            {"rule": "value IN ['A', 'B', 'C']", "value": "D", "expected": False},
            
            # User context
            {"rule": "user.role == 'manager'", "value": "anything", 
             "context": {"user": {"role": "manager"}}, "expected": True},
            {"rule": "user.role == 'manager'", "value": "anything", 
             "context": {"user": {"role": "employee"}}, "expected": False}
        ]
        
        results = []
        for test in test_cases:
            result = self.engine.validate_rule(
                test["rule"], 
                test["value"], 
                test.get("context", {})
            )
            results.append({
                "rule": test["rule"],
                "value": test["value"],
                "expected": test["expected"],
                "actual": result.is_valid,
                "passed": result.is_valid == test["expected"],
                "error": result.error_message
            })
        
        return {
            "total_tests": len(results),
            "passed": sum(1 for r in results if r["passed"]),
            "failed": sum(1 for r in results if not r["passed"]),
            "results": results
        }


# Global DSL validator instance
dsl_validator = MoveworksDSLValidator()
