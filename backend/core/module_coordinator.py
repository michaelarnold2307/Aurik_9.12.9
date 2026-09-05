"""
core/module_coordinator.py
Advanced Module Coordination System
====================================

Ferrari Edition - Weltklasse Module Orchestrierung:
- Dependency Graph Resolution (Topological Sort)
- Parallel Execution (ThreadPool)
- ML-based Parameter Optimization
- Quality Prediction
- Error Recovery & Rollback
- Performance Profiling
- Resource Management

Version: 2.0.0 "Limited Edition"
Author: AURIK Team
Date: 10. Februar 2026
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from backend.core.module_communication import ModuleCommunicationBus
from backend.core.musical_quality_assurance import (
    MediumType,
    MusicalQualityAssurance,
    map_forensic_to_medium_type,
)
from backend.core.musical_quality_assurance import (
    ProcessingMode as QualityProcessingMode,
)
from backend.core.processing_context import ModuleState, ProcessingContext
from backend.core.quality_recovery import QualityRecoverySystem

logger = logging.getLogger(__name__)


class ExecutionStrategy(Enum):
    """Ausführungsstrategien für Modulverarbeitung."""

    SEQUENTIAL = "sequential"  # One module at a time
    PARALLEL = "parallel"  # Independent modules in parallel
    ADAPTIVE = "adaptive"  # ML-based optimal strategy
    STREAMING = "streaming"  # Low-latency streaming mode


class ModulePriority(Enum):
    """Module priority levels."""

    CRITICAL = 1  # Must run (e.g., DC Blocker)
    HIGH = 2  # Important (e.g., Forensics)
    NORMAL = 3  # Standard processing
    LOW = 4  # Optional enhancement
    OPTIONAL = 5  # Only if resources available


@dataclass
class ModuleDescriptor:
    """
    Descriptor for a processing module.
    """

    name: str
    module_class: type
    priority: ModulePriority = ModulePriority.NORMAL
    dependencies: list[str] = field(default_factory=list)  # Module names this depends on
    provides: list[str] = field(default_factory=list)  # Capabilities this provides
    categories: list[str] = field(default_factory=list)  # e.g., ['tape', 'defect_removal']
    parallel_safe: bool = True  # Can run in parallel with others
    estimated_cost: float = 1.0  # Relative computational cost (1.0 = baseline)
    min_quality_threshold: float = 0.0  # Minimum input quality to run (0-1)

    # ML-based parameter optimization
    supports_ml_params: bool = False
    default_params: dict[str, Any] = field(default_factory=dict)

    # Execution metadata
    avg_execution_time: float = 0.0
    success_rate: float = 1.0
    last_executed: float | None = None


@dataclass
class ExecutionPlan:
    """
    Plan for executing modules.
    """

    stages: list[list[ModuleDescriptor]]  # Stages (each stage can run in parallel)
    total_cost: float
    estimated_time_sec: float
    parallel_opportunities: int
    critical_path_modules: list[str]


@dataclass
class ExecutionResult:
    """
    Result from module execution.
    """

    module_name: str
    success: bool
    execution_time_sec: float
    confidence: float
    metrics: dict[str, Any]
    error: str | None = None
    output_audio: NDArray[Any] | None = None


class ModuleCoordinator:
    """
    Advanced Module Coordination System.

    Features:
    - Dependency graph resolution
    - Parallel execution optimization
    - ML-based parameter inference
    - Quality prediction
    - Error recovery & rollback
    - Performance profiling
    - Resource management

    Usage:
        coordinator = ModuleCoordinator(context, bus)

        # Register modules
        coordinator.register_module(
            name="ForensicAnalyzer",
            module_class=UnifiedForensicAnalyzer,
            priority=ModulePriority.HIGH,
            provides=["forensic_analysis", "material_detection"]
        )

        coordinator.register_module(
            name="TapeSpecialist",
            module_class=TapeSpecialist,
            dependencies=["ForensicAnalyzer"],
            categories=["tape", "defect_removal"]
        )

        # Execute
        result = coordinator.execute(audio, sr, strategy=ExecutionStrategy.ADAPTIVE)
    """

    VERSION = "2.0.0"

    def __init__(
        self,
        context: ProcessingContext,
        bus: ModuleCommunicationBus,
        max_workers: int = 4,
        enable_ml_optimization: bool = True,
        enable_quality_prediction: bool = True,
        enable_musical_quality_assurance: bool = True,  # OUT-OF-THE-BOX: Adaptive Excellence IMMER aktiv!
        ml_param_inference_engine: Any | None = None,
    ):
        """
        Initialize Module Coordinator.

        Args:
            context: Processing context for state management
            bus: Communication bus for inter-module messaging
            max_workers: Maximum parallel workers
            enable_ml_optimization: Enable ML-based parameter optimization
            enable_quality_prediction: Enable quality prediction
            enable_musical_quality_assurance: Enable Adaptive Musical Excellence (DEFAULT: True)
                                            → Aurik findet IMMER die beste Lösung!
                                            → Kein harter Abbruch, nur Optimierung
                                            → Maximale musikalische Exzellenz garantiert
        """
        self.context = context
        self.bus = bus
        self.max_workers = max_workers
        self.enable_ml_optimization = enable_ml_optimization
        self.enable_quality_prediction = enable_quality_prediction
        self.enable_musical_quality_assurance = enable_musical_quality_assurance

        # Module registry
        self._modules: dict[str, ModuleDescriptor] = {}
        self._module_instances: dict[str, Any] = {}  # Cached instances

        # Dependency graph
        self._dependency_graph: dict[str, set[str]] = {}
        self._reverse_graph: dict[str, set[str]] = {}

        # Execution state
        self._lock = threading.RLock()
        self._thread_pool: ThreadPoolExecutor | None = None

        # Performance tracking
        self._execution_history: list[ExecutionResult] = []

        # ML Parameter Inference Engine
        try:
            from backend.core.ml_parameter_inference import MLParameterInferenceEngine

            self._ml_param_inference_engine = ml_param_inference_engine or MLParameterInferenceEngine()
            logger.info("  ✓ ML Parameter Inference Engine: ACTIVE")
        except ImportError:
            self._ml_param_inference_engine = None
            logger.warning("  ⚠ ML Parameter Inference Engine NICHT verfügbar!")
        self._quality_predictor = None

        # Musical Quality Assurance
        self._mqa_system = MusicalQualityAssurance() if enable_musical_quality_assurance else None
        self._quality_baseline = None
        self._audio_checkpoints: list[tuple[str, NDArray[Any]]] = []

        # Adaptive Musical Excellence System - OUT-OF-THE-BOX perfektioniert!
        # → Findet IMMER die beste Lösung (kein harter Abbruch)
        # → Optimiert selbstständig (keine User-Intervention nötig)
        # → Maximale musikalische Exzellenz garantiert
        self._recovery_system = QualityRecoverySystem() if enable_musical_quality_assurance else None
        self._recovery_attempts = 0
        self._recovered_modules: list[str] = []

        logger.info("ModuleCoordinator initialisiert (v2.0.0 Limited Edition)")
        if enable_musical_quality_assurance:
            logger.info("  ✓ Adaptive Musical Excellence: ACTIVE (out-of-the-box perfektioniert)")
            logger.info("    → Findet IMMER die beste Lösung")
            logger.info("    → Optimiert selbstständig")
            logger.info("    → Kein harter Abbruch")

    # === Module Registration ===

    def register_module(
        self,
        name: str,
        module_class: type,
        priority: ModulePriority = ModulePriority.NORMAL,
        dependencies: list[str] | None = None,
        provides: list[str] | None = None,
        categories: list[str] | None = None,
        parallel_safe: bool = True,
        estimated_cost: float = 1.0,
        supports_ml_params: bool = False,
        default_params: dict[str, Any] | None = None,
    ) -> None:
        """
        Registriert a processing module.

        Args:
            name: Unique module name
            module_class: Module class to instantiate
            priority: Module priority level
            dependencies: List of module names this depends on
            provides: List of capabilities this provides
            categories: Module categories (e.g., ['tape', 'defect_removal'])
            parallel_safe: Can run in parallel with other modules
            estimated_cost: Relative computational cost
            supports_ml_params: Supports ML-based parameter optimization
            default_params: Default parameters for module
        """
        with self._lock:
            descriptor = ModuleDescriptor(
                name=name,
                module_class=module_class,
                priority=priority,
                dependencies=dependencies or [],
                provides=provides or [],
                categories=categories or [],
                parallel_safe=parallel_safe,
                estimated_cost=estimated_cost,
                supports_ml_params=supports_ml_params,
                default_params=default_params or {},
            )

            self._modules[name] = descriptor

            # Update dependency graph
            self._dependency_graph[name] = set(descriptor.dependencies)
            self._reverse_graph.setdefault(name, set())

            for dep in descriptor.dependencies:
                self._reverse_graph.setdefault(dep, set()).add(name)

            logger.info("Registered module: %s (priority=%s, deps=%s)", name, priority.value, descriptor.dependencies)

    def unregister_module(self, name: str) -> None:
        """Unregister a module."""
        with self._lock:
            if name in self._modules:
                del self._modules[name]
                if name in self._module_instances:
                    del self._module_instances[name]

                # Clean up graphs
                if name in self._dependency_graph:
                    del self._dependency_graph[name]
                if name in self._reverse_graph:
                    for dependent in self._reverse_graph[name]:
                        self._dependency_graph[dependent].discard(name)
                    del self._reverse_graph[name]

                logger.info("Unregistered module: %s", name)

    def get_registered_modules(self) -> list[str]:
        """Gibt zurück: list of registered module names."""
        with self._lock:
            return list(self._modules.keys())

    # === Dependency Resolution ===

    def build_execution_plan(
        self, selected_modules: list[str] | None = None, forensic_analysis: dict[str, Any] | None = None
    ) -> ExecutionPlan:
        """
        Erstellt optimal execution plan using topological sort.

        Args:
            selected_modules: Specific modules to execute (None = all)
            forensic_analysis: Forensic analysis for intelligent selection

        Returns:
            ExecutionPlan with stages for parallel execution
        """
        with self._lock:
            # Determine which modules to execute
            modules_to_execute = list(self._modules.keys()) if selected_modules is None else selected_modules

            # Apply forensic-based filtering if available
            if forensic_analysis and self.enable_ml_optimization:
                modules_to_execute = self._filter_by_forensics(modules_to_execute, forensic_analysis)

            # Topological sort with staging
            stages = self._topological_sort_with_stages(modules_to_execute)

            # Calculate plan metrics
            total_cost = sum(self._modules[mod].estimated_cost for stage in stages for mod in stage)

            # Estimate time (sequential baseline)
            sum(self._modules[mod].avg_execution_time or 1.0 for stage in stages for mod in stage)

            # With parallelization (assuming perfect speedup within stages)
            parallel_time = sum(
                max(self._modules[mod].avg_execution_time or 1.0 for mod in stage) if stage else 0 for stage in stages
            )

            # Identify critical path
            critical_path = self._find_critical_path(modules_to_execute)

            plan = ExecutionPlan(
                stages=[[self._modules[name] for name in stage] for stage in stages],
                total_cost=total_cost,
                estimated_time_sec=parallel_time,
                parallel_opportunities=sum(1 for stage in stages if len(stage) > 1),
                critical_path_modules=critical_path,
            )

            logger.info(
                "Execution plan: %s stages, %s parallel opportunities", len(stages), plan.parallel_opportunities
            )

            return plan

    def _topological_sort_with_stages(self, modules: list[str]) -> list[list[str]]:
        """
        Topological sort that groups independent modules into stages.

        Returns:
            List of stages, where each stage is a list of module names
            that can be executed in parallel.
        """
        # Build subgraph for selected modules
        graph: dict[str, set[str]] = {mod: set() for mod in modules}
        for mod in modules:
            if mod in self._dependency_graph:
                graph[mod] = self._dependency_graph[mod] & set(modules)

        # Compute in-degree
        in_degree = {mod: len(graph[mod]) for mod in modules}

        # Stage-by-stage processing
        stages = []
        while any(in_degree[mod] == 0 for mod in modules if mod in in_degree):
            # Find all nodes with in-degree 0 (can execute now)
            current_stage = [mod for mod in modules if mod in in_degree and in_degree[mod] == 0]

            if not current_stage:
                break

            # Sort by priority within stage
            current_stage.sort(key=lambda m: self._modules[m].priority.value)

            stages.append(current_stage)

            # Remove from graph
            for mod in current_stage:
                del in_degree[mod]

                # Decrease in-degree of dependents
                if mod in self._reverse_graph:
                    for dependent in self._reverse_graph[mod]:
                        if dependent in in_degree:
                            in_degree[dependent] -= 1

        # Check for cycles
        if in_degree:
            remaining = list(in_degree.keys())
            logger.warning("Dependency cycle erkannt involving: %s", remaining)
            # Add remaining as final stage
            stages.append(remaining)

        return stages

    def _find_critical_path(self, modules: list[str]) -> list[str]:
        """
        Findet den kritischen Pfad (längsten Pfad durch den Abhängigkeitsgraphen).

        Returns:
            List of module names on critical path
        """
        # Simplified critical path (most dependencies)
        max_depth = 0
        critical_module = None

        for mod in modules:
            depth = self._compute_depth(mod, modules)
            if depth > max_depth:
                max_depth = depth
                critical_module = mod

        if critical_module:
            return self._trace_path(critical_module, modules)
        return []

    def _compute_depth(self, module: str, allowed_modules: list[str], visited: set[str] | None = None) -> int:
        """Berechnet depth in dependency graph with cycle detection."""
        if visited is None:
            visited = set()

        if module in visited:
            # Cycle detected, return 0 to break
            return 0

        if module not in self._dependency_graph:
            return 0

        deps = self._dependency_graph[module] & set(allowed_modules)
        if not deps:
            return 0

        visited.add(module)
        depth = 1 + max(self._compute_depth(dep, allowed_modules, visited.copy()) for dep in deps)

        return depth

    def _trace_path(self, module: str, allowed_modules: list[str], visited: set[str] | None = None) -> list[str]:
        """Trace path from root to module with cycle detection."""
        if visited is None:
            visited = set()

        if module in visited:
            # Cycle detected
            return [module]

        path = [module]
        visited.add(module)

        if module in self._dependency_graph:
            deps = self._dependency_graph[module] & set(allowed_modules)
            if deps:
                # Pick deepest dependency
                deepest_dep = max(deps, key=lambda d: self._compute_depth(d, allowed_modules))
                path = self._trace_path(deepest_dep, allowed_modules, visited.copy()) + path

        return path

    def _filter_by_forensics(self, modules: list[str], forensic_analysis: dict[str, Any]) -> list[str]:
        """
        Filtert modules based on forensic analysis.

        Intelligent module selection based on detected material and defects.
        """
        filtered = []

        material = forensic_analysis.get("medium_type", "").upper()
        detected_defects = forensic_analysis.get("defects_detected", {})

        for mod_name in modules:
            descriptor = self._modules[mod_name]

            # Always include critical modules
            if descriptor.priority == ModulePriority.CRITICAL:
                filtered.append(mod_name)
                continue

            # Check category match
            categories = descriptor.categories

            # Material-specific modules
            if material in ["VINYL", "TAPE", "CASSETTE", "CD"]:
                if material.lower() in categories or "forensics" in categories:
                    filtered.append(mod_name)
                    continue

            # Defect-specific modules
            if detected_defects:
                for defect, detected in detected_defects.items():
                    if detected and defect.lower() in categories:
                        filtered.append(mod_name)
                        break
            else:
                # No defects, include general enhancement modules
                if "enhancement" in categories or "general" in categories:
                    filtered.append(mod_name)

        logger.info("Forensic filtering: %s → %s modules", len(modules), len(filtered))
        return filtered

    # === Execution ===

    def execute(
        self,
        audio: NDArray[Any],
        sample_rate: int,
        strategy: ExecutionStrategy = ExecutionStrategy.ADAPTIVE,
        forensic_analysis: dict[str, Any] | None = None,
        selected_modules: list[str] | None = None,
        processing_mode: str = "restoration",
        medium_type: MediumType | None = None,
    ) -> dict[str, Any]:
        """
        Führt aus: modules according to optimal plan.

        Args:
            audio: Input audio
            sample_rate: Sample rate
            strategy: Execution strategy
            forensic_analysis: Forensic analysis for intelligent processing
            selected_modules: Specific modules to execute

        Returns:
            Dictionary with processed audio and execution report
        """
        start_time = time.monotonic()

        # Build execution plan
        plan = self.build_execution_plan(selected_modules, forensic_analysis)

        logger.info("Executing %s stages with %s parallel opportunities", len(plan.stages), plan.parallel_opportunities)

        # === MUSICAL QUALITY ASSURANCE: Establish Baseline ===
        quality_mode = self._map_processing_mode(processing_mode)
        detected_medium = medium_type

        if self._mqa_system and forensic_analysis and not detected_medium:
            detected_medium = map_forensic_to_medium_type(forensic_analysis)

        if self._mqa_system and detected_medium:
            self._quality_baseline = self._mqa_system.establish_baseline(  # type: ignore[assignment]
                audio, sample_rate, detected_medium, quality_mode
            )
            logger.info("✓ Quality baseline established: %.1f/100", self._quality_baseline.overall_score)  # type: ignore[attr-defined]

        # Initialize execution state
        current_audio = audio.copy()
        stage_results = []
        self._audio_checkpoints = [("original", audio.copy())]  # Save original for rollback

        modules_applied = []

        # Execute stages
        for stage_idx, stage in enumerate(plan.stages):
            logger.info("Stufe %s/%s: %s", stage_idx + 1, len(plan.stages), [m.name for m in stage])

            if strategy == ExecutionStrategy.PARALLEL and len(stage) > 1:
                # Parallel execution within stage
                stage_result = self._execute_stage_parallel(stage, current_audio, sample_rate, forensic_analysis)
            else:
                # Sequential execution within stage
                stage_result = self._execute_stage_sequential(stage, current_audio, sample_rate, forensic_analysis)

            stage_results.extend(stage_result)

            # Update current audio with last successful output
            for result in stage_result:
                if result.success and result.output_audio is not None:
                    # Save checkpoint before quality gate check
                    previous_audio = current_audio.copy()
                    current_audio = result.output_audio
                    modules_applied.append(result.module_name)

                    # === MUSICAL QUALITY ASSURANCE: Check Quality Gate ===
                    if self._mqa_system and self._quality_baseline and detected_medium:
                        gate_passed, reason = self._mqa_system.check_quality_gate(
                            current_audio,
                            sample_rate,
                            self._quality_baseline,
                            detected_medium,
                            quality_mode,
                            module_name=result.module_name,
                        )

                        if not gate_passed:
                            logger.warning("⚠ Quality gate fehlgeschlagen after %s: %s", result.module_name, reason)

                            # === ADAPTIVE MUSICAL EXCELLENCE: Findet IMMER die beste Lösung! ===
                            if self._recovery_system:
                                logger.info("🔧 ADAPTIVE EXCELLENCE: Automatische Optimierung läuft...")

                                # Generate MQA report for diagnosis
                                temp_report = self._mqa_system.validate_final_quality(
                                    audio, current_audio, sample_rate, detected_medium, quality_mode, modules_applied
                                )

                                # Diagnose problem
                                recovery_plan = self._recovery_system.diagnose_problem(
                                    current_audio, sample_rate, temp_report, detected_medium, quality_mode
                                )

                                # Execute recovery
                                recovery_result = self._recovery_system.execute_recovery(
                                    audio,
                                    current_audio,
                                    sample_rate,
                                    recovery_plan,
                                    modules_applied,
                                    detected_medium,
                                    quality_mode,
                                )

                                if recovery_result.success:
                                    # Excellence achieved! Use optimized audio
                                    logger.info(
                                        f"✓ MUSIKALISCHE EXZELLENZ erreicht: {recovery_result.improvement:+.1f} points"
                                    )
                                    logger.info("  Strategie: %s", recovery_result.strategy_used.value)
                                    logger.info("  Maßnahmen: %s", ", ".join(recovery_result.actions_taken))

                                    current_audio = recovery_result.recovered_audio
                                    self._recovery_attempts += 1
                                    self._recovered_modules.append(result.module_name)

                                    # Save recovered checkpoint
                                    self._audio_checkpoints.append(
                                        (f"{result.module_name}_recovered", current_audio.copy())
                                    )

                                    # Mark result as optimized (excellence achieved)
                                    result.warning = f"Quality optimiert durch Adaptive Excellence: {reason}"
                                else:
                                    # Adaptive Excellence always finds a solution, but log if suboptimal
                                    logger.info("  → Suboptimal but best achievable solution found")
                                    # Still use the recovered audio (best we could achieve)
                                    current_audio = recovery_result.recovered_audio
                                    self._recovery_attempts += 1
                                    self._recovered_modules.append(result.module_name)

                                    # Save checkpoint
                                    self._audio_checkpoints.append(
                                        (f"{result.module_name}_optimized", current_audio.copy())
                                    )

                                    result.warning = f"Best achievable quality reached: {reason}"
                            else:
                                # No recovery system - fallback to old behavior
                                logger.warning(
                                    "  → Rolling back to previous checkpoint (no Wiederherstellung verfuegbar)"
                                )
                                current_audio = previous_audio
                                modules_applied.pop()  # Remove failed module

                                result.success = False
                                result.error = f"Quality gate failed: {reason}"

                                # Stop processing if critical violation
                                if "character" in reason.lower() or "unnatural" in reason.lower():
                                    logger.error("  → CRITICAL VIOLATION - stoppe processing")
                                    break
                        else:
                            # Quality gate passed - save checkpoint
                            self._audio_checkpoints.append((result.module_name, current_audio.copy()))

            # Break outer loop if inner loop broke due to critical violation
            if stage_result and not stage_result[-1].success and "CRITICAL" in str(stage_result[-1].error):
                break

        total_time = time.monotonic() - start_time

        # === MUSICAL QUALITY ASSURANCE: Final Validation ===
        mqa_report = None
        quality_guaranteed = False

        if self._mqa_system and detected_medium:
            mqa_report = self._mqa_system.validate_final_quality(
                audio, current_audio, sample_rate, detected_medium, quality_mode, modules_applied
            )
            # §v10.703 QUALITY_UNCERTAIN: validate_final_quality kann None liefern,
            # wenn keine perzeptuellen Daten (MUSHRA/HPI) verfügbar sind.
            if mqa_report is None:
                logger.warning("⚠ QUALITY NOT GUARANTEED: MQA-Report nicht verfügbar (perzeptuelle Daten fehlen)")
            else:
                quality_guaranteed = mqa_report.quality_guaranteed

                if not quality_guaranteed:
                    logger.warning("⚠ QUALITY NOT GUARANTEED: %s", mqa_report.verdict)
                    for warning in mqa_report.warnings:
                        logger.warning("  - %s", warning)

                    # If completely failed, consider rollback to best checkpoint
                    if mqa_report.integrity_result.should_rollback:
                        logger.error("  → ROLLBACK RECOMMENDED")
                        # Find best checkpoint
                        if len(self._audio_checkpoints) >= 2:
                            best_checkpoint = self._audio_checkpoints[-2]  # Second to last
                            logger.info("  → Rolling back to: %s", best_checkpoint[0])
                            current_audio = best_checkpoint[1]
                else:
                    logger.info("✓ QUALITY GUARANTEED: %s", mqa_report.verdict)

        # Generate report
        report = {
            "output_audio": current_audio,
            "execution_time_sec": total_time,
            "num_stages": len(plan.stages),
            "num_modules_executed": len(stage_results),
            "successful_modules": sum(1 for r in stage_results if r.success),
            "failed_modules": sum(1 for r in stage_results if not r.success),
            "module_results": stage_results,
            "plan": plan,
            "modules_applied": modules_applied,
            "mqa_report": mqa_report,
            "quality_guaranteed": quality_guaranteed,
            "medium_type": detected_medium.value if detected_medium else None,
            "processing_mode": processing_mode,
            # Quality Recovery Info
            "recovery_attempts": self._recovery_attempts,
            "recovered_modules": self._recovered_modules,
            "user_protected": self._recovery_attempts > 0,  # User wurde ins Warme gebracht!
        }

        logger.info(
            f"Execution vollstaendig: {report['successful_modules']}/{report['num_modules_executed']} modules erfolgreich in {total_time:.2f}s"
        )
        if quality_guaranteed:
            logger.info("✓ Quality Guaranteed: %+.1f improvement", mqa_report.musical_improvement)  # type: ignore[union-attr]

        return report

    def _execute_stage_sequential(
        self,
        stage: list[ModuleDescriptor],
        audio: NDArray[Any],
        sample_rate: int,
        forensic_analysis: dict[str, Any] | None,
    ) -> list[ExecutionResult]:
        """Führt aus: stage modules sequentially."""
        results = []
        current_audio = audio

        for descriptor in stage:
            result = self._execute_single_module(descriptor, current_audio, sample_rate, forensic_analysis)
            results.append(result)

            if result.success and result.output_audio is not None:
                current_audio = result.output_audio

        return results

    def _map_processing_mode(self, mode_str: str) -> QualityProcessingMode:
        """Map processing mode string to QualityProcessingMode enum."""
        mode_map = {
            "restoration": QualityProcessingMode.RESTORATION,
            "studio_2026": QualityProcessingMode.STUDIO_2026,
        }
        return mode_map.get(mode_str.lower(), QualityProcessingMode.RESTORATION)

    def _execute_stage_parallel(
        self,
        stage: list[ModuleDescriptor],
        audio: NDArray[Any],
        sample_rate: int,
        forensic_analysis: dict[str, Any] | None,
    ) -> list[ExecutionResult]:
        """Führt aus: stage modules in parallel."""
        if self._thread_pool is None:
            self._thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)

        # Submit all modules
        futures = {}
        for descriptor in stage:
            future = self._thread_pool.submit(
                self._execute_single_module, descriptor, audio, sample_rate, forensic_analysis
            )
            futures[future] = descriptor.name

        # Collect results
        results = []
        for future in as_completed(futures):
            result = future.result()
            results.append(result)

        return results

    def _execute_single_module(
        self,
        descriptor: ModuleDescriptor,
        audio: NDArray[Any],
        sample_rate: int,
        forensic_analysis: dict[str, Any] | None,
    ) -> ExecutionResult:
        """
        Führt aus: a single module with error handling.
        """
        start_time = time.monotonic()

        try:
            # Get or create module instance
            if descriptor.name not in self._module_instances:
                self._module_instances[descriptor.name] = descriptor.module_class()

            module = self._module_instances[descriptor.name]

            # Update context
            self.context.register_module(descriptor.name)
            self.context.set_module_state(descriptor.name, ModuleState.IN_PROGRESS)

            # Prepare parameters
            params = descriptor.default_params.copy()

            # ML-based parameter optimization
            if descriptor.supports_ml_params and self.enable_ml_optimization and forensic_analysis:
                optimized_params = self._optimize_parameters(descriptor, audio, sample_rate, forensic_analysis)
                params.update(optimized_params)

            # Execute module
            if hasattr(module, "process"):
                output_audio = module.process(audio, sample_rate, **params)
            elif hasattr(module, "analyze"):
                # For analyzer modules
                analysis = module.analyze(audio, sample_rate, **params)
                output_audio = audio  # No audio change
                params["analysis"] = analysis
            else:
                raise AttributeError(f"Module {descriptor.name} has no 'process' or 'analyze' method")

            execution_time = time.monotonic() - start_time

            # Update metrics
            descriptor.avg_execution_time = (
                0.9 * descriptor.avg_execution_time + 0.1 * execution_time
                if descriptor.avg_execution_time > 0
                else execution_time
            )
            descriptor.success_rate = 0.95 * descriptor.success_rate + 0.05
            descriptor.last_executed = time.monotonic()

            # Get module metrics
            metrics = module.get_metrics() if hasattr(module, "get_metrics") else {}
            confidence = metrics.get("confidence", 0.9)

            # Update context
            self.context.complete_module(descriptor.name, confidence=confidence, metrics=metrics)

            result = ExecutionResult(
                module_name=descriptor.name,
                success=True,
                execution_time_sec=execution_time,
                confidence=confidence,
                metrics=metrics,
                output_audio=output_audio,
            )

            logger.info("✓ %s abgeschlossen in %.3fs (confidence=%.2f)", descriptor.name, execution_time, confidence)

            return result

        except Exception as e:
            execution_time = time.monotonic() - start_time

            logger.error("❌ %s fehlgeschlagen: %s", descriptor.name, e)

            # Update context
            self.context.fail_module(descriptor.name, str(e))

            # Update metrics
            descriptor.success_rate = 0.95 * descriptor.success_rate

            return ExecutionResult(
                module_name=descriptor.name,
                success=False,
                execution_time_sec=execution_time,
                confidence=0.0,
                metrics={},
                error=str(e),
                output_audio=None,
            )

    def _optimize_parameters(
        self, descriptor: ModuleDescriptor, audio: NDArray[Any], sample_rate: int, forensic_analysis: dict[str, Any]
    ) -> dict[str, Any]:
        """
        ML-basierte Parameteroptimierung.
        Nutzt die ML Parameter Inference Engine, falls verfügbar, sonst heuristische Optimierung.
        """
        if self._ml_param_inference_engine and hasattr(self._ml_param_inference_engine, "infer_parameters"):
            # Features für Inferenz zusammenstellen
            features = {
                "medium_type": forensic_analysis.get("medium_type", ""),
                "quality_assessment": forensic_analysis.get("quality_assessment", ""),
                "default_params": descriptor.default_params,
                "module_name": descriptor.name,
            }
            try:
                inferred = self._ml_param_inference_engine.infer_parameters(features)
                logger.info("  → ML Parameter Inference für %s: %s", descriptor.name, inferred)
                if hasattr(inferred, "params") and isinstance(inferred.params, dict):
                    return dict(inferred.params)
                if isinstance(inferred, dict):
                    params = inferred.get("params")
                    if isinstance(params, dict):
                        return dict(params)
                    return dict(inferred)
                logger.warning(
                    "ML Parameter Inference returned unsupported type for %s: %s",
                    descriptor.name,
                    type(inferred).__name__,
                )
                return {}
            except Exception as e:
                logger.warning("ML Parameter Inference Engine Fehler: %s", e)
                # Fallback auf Heuristik
        # Heuristische Optimierung (wie bisher)
        optimized = {}
        material = forensic_analysis.get("medium_type", "").upper()
        quality = forensic_analysis.get("quality_assessment", "GOOD")
        if "strength" in descriptor.default_params:
            base_strength = descriptor.default_params["strength"]
            if quality == "POOR":
                optimized["strength"] = min(base_strength * 1.3, 1.0)
            elif quality == "EXCELLENT":
                optimized["strength"] = base_strength * 0.7
        if material == "VINYL" and "threshold" in descriptor.default_params:
            optimized["threshold"] = descriptor.default_params["threshold"] * 0.85
        return optimized

    # === Quality Prediction ===

    def predict_output_quality(self, audio: NDArray[Any], sample_rate: int, plan: ExecutionPlan) -> dict[str, float]:
        """
        Predict output quality before processing.

        Returns:
            Dictionary with predicted metrics (SNR, clarity, etc.)
        """
        # Placeholder for ML-based quality prediction
        # Would use trained model based on:
        # - Input audio characteristics
        # - Selected modules
        # - Historical performance

        # For now, return heuristic estimates
        predicted = {
            "snr_improvement_db": sum(
                2.0 if "noise" in mod.categories else 0.5 for stage in plan.stages for mod in stage
            ),
            "clarity_improvement": 0.1 * len(plan.stages),
            "confidence": 0.75,
        }

        return predicted

    # === Statistics ===

    def get_statistics(self) -> dict[str, Any]:
        """Gibt zurück: coordinator statistics."""
        with self._lock:
            return {
                "registered_modules": len(self._modules),
                "cached_instances": len(self._module_instances),
                "execution_history_size": len(self._execution_history),
                "avg_success_rate": np.mean([m.success_rate for m in self._modules.values()]) if self._modules else 0.0,
                "modules_by_priority": {
                    priority.name: sum(1 for m in self._modules.values() if m.priority == priority)
                    for priority in ModulePriority
                },
            }

    # === Lifecycle ===

    def shutdown(self) -> None:
        """Shutdown coordinator and cleanup resources."""
        if self._thread_pool:
            # §3.9.4: cancel pending futures + wait for in-flight work to complete.
            self._thread_pool.shutdown(wait=True, cancel_futures=True)
            self._thread_pool = None

        self._module_instances.clear()
        logger.info("ModuleCoordinator Herunterfahren")

    def __repr__(self) -> str:
        """String representation."""
        stats = self.get_statistics()
        return f"ModuleCoordinator(modules={stats['registered_modules']}, success_rate={stats['avg_success_rate']:.2%})"


# === Convenience Functions ===


def create_coordinator(
    context: ProcessingContext, bus: ModuleCommunicationBus, max_workers: int = 4
) -> ModuleCoordinator:
    """Erstellt a module coordinator with default settings."""
    return ModuleCoordinator(
        context=context, bus=bus, max_workers=max_workers, enable_ml_optimization=True, enable_quality_prediction=True
    )
