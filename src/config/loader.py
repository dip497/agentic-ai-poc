"""
Configuration loader for Moveworks-style conversational AI system.

This module loads and validates the Moveworks-style configuration from YAML files,
converting them into the appropriate Pydantic models.
"""

import yaml
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

from models.moveworks import (
    Plugin, ConversationalProcess, Slot, Activity,
    DataType, SlotInferencePolicy, ActivityType,
    ResolverStrategy, ResolverMethod, ResolverMethodType, StaticOption,
    ConfirmationPolicy, InputMapping, OutputMapping
)


logger = logging.getLogger(__name__)


class MoveworksConfigLoader:
    """Loads and validates Moveworks-style configuration."""
    
    def __init__(self, config_path: str = "config/moveworks_config.yml"):
        """Initialize the config loader."""
        self.config_path = Path(config_path)
        self.config_data: Optional[Dict[str, Any]] = None
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            self.config_data = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {self.config_path}")
        return self.config_data
    
    def load_plugins(self) -> List[Plugin]:
        """Load plugins from configuration."""
        if not self.config_data:
            self.load_config()
        
        plugins = []
        
        for plugin_config in self.config_data.get("plugins", []):
            plugin = self._create_plugin_from_config(plugin_config)
            plugins.append(plugin)
        
        logger.info(f"Loaded {len(plugins)} plugins")
        return plugins
    
    def _create_plugin_from_config(self, plugin_config: Dict[str, Any]) -> Plugin:
        """Create a Plugin from configuration."""
        processes = []
        
        for process_config in plugin_config.get("conversational_processes", []):
            process = self._create_process_from_config(process_config)
            processes.append(process)
        
        return Plugin(
            name=plugin_config["name"],
            description=plugin_config["description"],
            conversational_processes=processes,
            access_policies=plugin_config.get("access_policies"),
            metadata=plugin_config.get("metadata", {})
        )
    
    def _create_process_from_config(self, process_config: Dict[str, Any]) -> ConversationalProcess:
        """Create a ConversationalProcess from configuration."""
        # Load slots
        slots = []
        for slot_config in process_config.get("slots", []):
            slot = self._create_slot_from_config(slot_config)
            slots.append(slot)
        
        # Load activities
        activities = []
        for activity_config in process_config.get("activities", []):
            activity = self._create_activity_from_config(activity_config)
            activities.append(activity)
        
        return ConversationalProcess(
            title=process_config["title"],
            description=process_config["description"],
            trigger_utterances=process_config["trigger_utterances"],
            slots=slots,
            activities=activities,
            decision_policies=process_config.get("decision_policies"),
            metadata=process_config.get("metadata", {})
        )
    
    def _create_slot_from_config(self, slot_config: Dict[str, Any]) -> Slot:
        """Create a Slot from configuration."""
        # Parse data type
        data_type = DataType(slot_config["data_type"])
        
        # Parse inference policy
        inference_policy = SlotInferencePolicy(slot_config["slot_inference_policy"])
        
        # Parse resolver strategy if present
        resolver_strategy = None
        if "resolver_strategy" in slot_config:
            resolver_strategy = self._create_resolver_strategy_from_config(
                slot_config["resolver_strategy"]
            )
        
        return Slot(
            name=slot_config["name"],
            data_type=data_type,
            slot_description=slot_config["slot_description"],
            slot_validation_policy=slot_config.get("slot_validation_policy"),
            slot_validation_description=slot_config.get("slot_validation_description"),
            slot_inference_policy=inference_policy,
            resolver_strategy=resolver_strategy
        )
    
    def _create_resolver_strategy_from_config(self, resolver_config: Dict[str, Any]) -> ResolverStrategy:
        """Create a ResolverStrategy from configuration."""
        methods = []
        for method_config in resolver_config["methods"]:
            method = self._create_resolver_method_from_config(method_config)
            methods.append(method)

        return ResolverStrategy(
            name=resolver_config["name"],
            data_type=resolver_config["data_type"],
            description=resolver_config["description"],
            methods=methods
        )

    def _create_resolver_method_from_config(self, method_config: Dict[str, Any]) -> ResolverMethod:
        """Create a ResolverMethod from configuration."""
        method_type = ResolverMethodType(method_config["method_type"])

        # Parse static options if present
        static_options = None
        if "static_options" in method_config:
            static_options = [
                StaticOption(
                    display_value=option["display_value"],
                    raw_value=option["raw_value"]
                )
                for option in method_config["static_options"]
            ]

        return ResolverMethod(
            name=method_config["method_name"],
            method_type=method_type,
            static_options=static_options,
            vector_store_name=method_config.get("vector_store_name"),
            similarity_threshold=method_config.get("similarity_threshold"),
            max_results=method_config.get("max_results"),
            api_endpoint=method_config.get("api_endpoint"),
            api_config=method_config.get("api_config"),
            custom_function=method_config.get("custom_function")
        )
    
    def _create_activity_from_config(self, activity_config: Dict[str, Any]) -> Activity:
        """Create an Activity from configuration."""
        activity_type = ActivityType(activity_config["activity_type"])
        
        # Parse confirmation policy
        confirmation_policy = ConfirmationPolicy.NO_CONFIRMATION
        if "confirmation_policy" in activity_config:
            if activity_config["confirmation_policy"] == "Require consent from the user":
                confirmation_policy = ConfirmationPolicy.REQUIRE_CONSENT
        
        # Parse input mapping
        input_mapping = None
        if "input_mapping" in activity_config:
            input_mapping = InputMapping(mappings=activity_config["input_mapping"])
        
        # Parse output mapping
        output_mapping = None
        if "output_mapping" in activity_config:
            output_mapping = OutputMapping(
                dot_walk_path=activity_config["output_mapping"].get("dot_walk_path"),
                output_key=activity_config["output_mapping"]["output_key"]
            )
        
        return Activity(
            activity_type=activity_type,
            required_slots=activity_config.get("required_slots", []),
            confirmation_policy=confirmation_policy,
            action_name=activity_config.get("action_name"),
            input_mapping=input_mapping,
            output_mapping=output_mapping,
            content_text=activity_config.get("content_text"),
            content_html=activity_config.get("content_html"),
            decision_cases=activity_config.get("decision_cases"),
            metadata=activity_config.get("metadata", {})
        )
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration."""
        if not self.config_data:
            self.load_config()
        
        return self.config_data.get("llm", {
            "default_model": "gpt-4o-mini",
            "temperature": 0.1,
            "max_tokens": 2000
        })
    
    def get_builtin_actions_config(self) -> Dict[str, Any]:
        """Get built-in actions configuration."""
        if not self.config_data:
            self.load_config()
        
        return self.config_data.get("builtin_actions", {})
    
    def get_custom_actions_config(self) -> Dict[str, Any]:
        """Get custom actions configuration."""
        if not self.config_data:
            self.load_config()
        
        return self.config_data.get("custom_actions", {})
    
    def get_vector_stores_config(self) -> Dict[str, Any]:
        """Get vector stores configuration."""
        if not self.config_data:
            self.load_config()
        
        return self.config_data.get("vector_stores", {})
    
    def get_access_policies(self) -> Dict[str, Any]:
        """Get access control policies."""
        if not self.config_data:
            self.load_config()
        
        return self.config_data.get("access_policies", {})
    
    def get_business_rules(self) -> Dict[str, Any]:
        """Get business rules."""
        if not self.config_data:
            self.load_config()
        
        return self.config_data.get("business_rules", {})
    
    def validate_config(self) -> List[str]:
        """Validate the loaded configuration."""
        errors = []
        
        if not self.config_data:
            try:
                self.load_config()
            except Exception as e:
                errors.append(f"Failed to load config: {e}")
                return errors
        
        # Validate plugins
        if "plugins" not in self.config_data:
            errors.append("No plugins defined in configuration")
        else:
            for i, plugin_config in enumerate(self.config_data["plugins"]):
                plugin_errors = self._validate_plugin_config(plugin_config, i)
                errors.extend(plugin_errors)
        
        return errors
    
    def _validate_plugin_config(self, plugin_config: Dict[str, Any], index: int) -> List[str]:
        """Validate a plugin configuration."""
        errors = []
        prefix = f"Plugin {index}"
        
        # Required fields
        required_fields = ["name", "description", "conversational_processes"]
        for field in required_fields:
            if field not in plugin_config:
                errors.append(f"{prefix}: Missing required field '{field}'")
        
        # Validate processes
        if "conversational_processes" in plugin_config:
            for j, process_config in enumerate(plugin_config["conversational_processes"]):
                process_errors = self._validate_process_config(process_config, f"{prefix}.Process {j}")
                errors.extend(process_errors)
        
        return errors
    
    def _validate_process_config(self, process_config: Dict[str, Any], prefix: str) -> List[str]:
        """Validate a process configuration."""
        errors = []
        
        # Required fields
        required_fields = ["title", "description", "trigger_utterances"]
        for field in required_fields:
            if field not in process_config:
                errors.append(f"{prefix}: Missing required field '{field}'")
        
        # Validate trigger utterances
        if "trigger_utterances" in process_config:
            if not isinstance(process_config["trigger_utterances"], list):
                errors.append(f"{prefix}: trigger_utterances must be a list")
            elif len(process_config["trigger_utterances"]) == 0:
                errors.append(f"{prefix}: trigger_utterances cannot be empty")
        
        # Validate slots
        if "slots" in process_config:
            for k, slot_config in enumerate(process_config["slots"]):
                slot_errors = self._validate_slot_config(slot_config, f"{prefix}.Slot {k}")
                errors.extend(slot_errors)
        
        # Validate activities
        if "activities" in process_config:
            for k, activity_config in enumerate(process_config["activities"]):
                activity_errors = self._validate_activity_config(activity_config, f"{prefix}.Activity {k}")
                errors.extend(activity_errors)
        
        return errors
    
    def _validate_slot_config(self, slot_config: Dict[str, Any], prefix: str) -> List[str]:
        """Validate a slot configuration."""
        errors = []
        
        # Required fields
        required_fields = ["name", "data_type", "slot_description", "slot_inference_policy"]
        for field in required_fields:
            if field not in slot_config:
                errors.append(f"{prefix}: Missing required field '{field}'")
        
        # Validate data type
        if "data_type" in slot_config:
            try:
                DataType(slot_config["data_type"])
            except ValueError:
                errors.append(f"{prefix}: Invalid data_type '{slot_config['data_type']}'")
        
        # Validate inference policy
        if "slot_inference_policy" in slot_config:
            try:
                SlotInferencePolicy(slot_config["slot_inference_policy"])
            except ValueError:
                errors.append(f"{prefix}: Invalid slot_inference_policy '{slot_config['slot_inference_policy']}'")
        
        return errors
    
    def _validate_activity_config(self, activity_config: Dict[str, Any], prefix: str) -> List[str]:
        """Validate an activity configuration."""
        errors = []
        
        # Required fields
        if "activity_type" not in activity_config:
            errors.append(f"{prefix}: Missing required field 'activity_type'")
        else:
            try:
                activity_type = ActivityType(activity_config["activity_type"])
                
                # Type-specific validation
                if activity_type == ActivityType.ACTION and "action_name" not in activity_config:
                    errors.append(f"{prefix}: Action activity missing 'action_name'")
                elif activity_type == ActivityType.CONTENT:
                    if "content_text" not in activity_config and "content_html" not in activity_config:
                        errors.append(f"{prefix}: Content activity missing 'content_text' or 'content_html'")
                elif activity_type == ActivityType.DECISION and "decision_cases" not in activity_config:
                    errors.append(f"{prefix}: Decision activity missing 'decision_cases'")
            
            except ValueError:
                errors.append(f"{prefix}: Invalid activity_type '{activity_config['activity_type']}'")
        
        return errors


# Example usage
def load_moveworks_config(config_path: str = "config/moveworks_config.yml") -> Dict[str, Any]:
    """Load and validate Moveworks configuration."""
    loader = MoveworksConfigLoader(config_path)
    
    # Validate configuration
    errors = loader.validate_config()
    if errors:
        logger.error("Configuration validation errors:")
        for error in errors:
            logger.error(f"  - {error}")
        raise ValueError(f"Configuration validation failed with {len(errors)} errors")
    
    # Load plugins
    plugins = loader.load_plugins()
    
    return {
        "plugins": plugins,
        "llm_config": loader.get_llm_config(),
        "builtin_actions_config": loader.get_builtin_actions_config(),
        "custom_actions_config": loader.get_custom_actions_config(),
        "vector_stores_config": loader.get_vector_stores_config(),
        "access_policies": loader.get_access_policies(),
        "business_rules": loader.get_business_rules()
    }
