"""Materials Database and Library Management.

Provides unified access to all 100+ materials in the catalog.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from goodyear_quantum_pilot.materials.base import (
    Material,
    MaterialCategory,
    PropertyType,
)
from goodyear_quantum_pilot.materials.elastomers import ELASTOMER_CATALOG
from goodyear_quantum_pilot.materials.rubbers import RUBBER_CATALOG
from goodyear_quantum_pilot.materials.quantum_engineered import QUANTUM_MATERIAL_CATALOG
from goodyear_quantum_pilot.materials.nanoarchitectures import NANO_CATALOG
from goodyear_quantum_pilot.materials.self_healing import HEALING_CATALOG

logger = logging.getLogger(__name__)


class MaterialsLibrary:
    """Unified materials library providing access to all catalogs.
    
    Aggregates all material catalogs and provides search, filter,
    and comparison capabilities.
    """
    
    def __init__(self) -> None:
        """Initialize materials library."""
        self._catalogs: dict[str, dict[str, Material]] = {
            "elastomers": ELASTOMER_CATALOG,
            "rubbers": RUBBER_CATALOG,
            "quantum": QUANTUM_MATERIAL_CATALOG,
            "nano": NANO_CATALOG,
            "healing": HEALING_CATALOG,
        }
        
        # Build master index
        self._index: dict[str, Material] = {}
        for catalog in self._catalogs.values():
            self._index.update(catalog)
        
        logger.info(f"MaterialsLibrary initialized with {len(self._index)} materials")
    
    @property
    def total_materials(self) -> int:
        """Total number of materials in library."""
        return len(self._index)
    
    @property
    def all_material_ids(self) -> list[str]:
        """List of all material IDs."""
        return list(self._index.keys())
    
    def get(self, material_id: str) -> Material | None:
        """Get material by ID.
        
        Args:
            material_id: Material identifier
            
        Returns:
            Material if found, None otherwise
        """
        return self._index.get(material_id)
    
    def __getitem__(self, material_id: str) -> Material:
        """Get material by ID with [] syntax.
        
        Args:
            material_id: Material identifier
            
        Returns:
            Material
            
        Raises:
            KeyError: If material not found
        """
        if material_id not in self._index:
            raise KeyError(f"Material '{material_id}' not found")
        return self._index[material_id]
    
    def __contains__(self, material_id: str) -> bool:
        """Check if material exists in library."""
        return material_id in self._index
    
    def __iter__(self) -> Iterator[Material]:
        """Iterate over all materials."""
        return iter(self._index.values())
    
    def __len__(self) -> int:
        """Number of materials."""
        return len(self._index)
    
    def filter_by_category(self, category: MaterialCategory) -> list[Material]:
        """Get all materials of a specific category.
        
        Args:
            category: Material category to filter by
            
        Returns:
            List of matching materials
        """
        return [m for m in self._index.values() if m.category == category]
    
    def filter_by_property(
        self,
        prop_type: PropertyType,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> list[Material]:
        """Filter materials by property value range.
        
        Args:
            prop_type: Property type to filter on
            min_value: Minimum value (inclusive)
            max_value: Maximum value (inclusive)
            
        Returns:
            List of matching materials
        """
        results = []
        
        for material in self._index.values():
            value = material.get_property(prop_type)
            if value is None:
                continue
            
            if min_value is not None and value < min_value:
                continue
            if max_value is not None and value > max_value:
                continue
            
            results.append(material)
        
        return results
    
    def search(self, query: str) -> list[Material]:
        """Search materials by name or description.
        
        Args:
            query: Search query string
            
        Returns:
            List of matching materials
        """
        query_lower = query.lower()
        
        return [
            m for m in self._index.values()
            if query_lower in m.name.lower() or query_lower in m.description.lower()
        ]
    
    def get_catalog(self, catalog_name: str) -> dict[str, Material]:
        """Get a specific catalog.
        
        Args:
            catalog_name: Name of catalog (elastomers, rubbers, quantum, nano, healing)
            
        Returns:
            Dictionary of materials in catalog
        """
        if catalog_name not in self._catalogs:
            raise ValueError(f"Unknown catalog: {catalog_name}")
        return self._catalogs[catalog_name]
    
    def compare_materials(
        self,
        material_ids: list[str],
        properties: list[PropertyType] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Compare multiple materials on specified properties.
        
        Args:
            material_ids: List of material IDs to compare
            properties: Properties to compare (default: all available)
            
        Returns:
            Dictionary with comparison data
        """
        comparison = {}
        
        # Get all properties if not specified
        if properties is None:
            properties = list(PropertyType)
        
        for mat_id in material_ids:
            material = self.get(mat_id)
            if material is None:
                continue
            
            comparison[mat_id] = {
                "name": material.name,
                "category": material.category.name,
                "properties": {},
            }
            
            for prop in properties:
                value = material.get_property(prop)
                if value is not None:
                    comparison[mat_id]["properties"][prop.name] = value
        
        return comparison
    
    def rank_by_property(
        self,
        prop_type: PropertyType,
        ascending: bool = False,
        top_n: int | None = None,
    ) -> list[tuple[str, float]]:
        """Rank materials by a property.
        
        Args:
            prop_type: Property to rank by
            ascending: Sort ascending (default: descending)
            top_n: Return only top N results
            
        Returns:
            List of (material_id, value) tuples sorted by value
        """
        rankings = []
        
        for mat_id, material in self._index.items():
            value = material.get_property(prop_type)
            if value is not None:
                rankings.append((mat_id, value))
        
        rankings.sort(key=lambda x: x[1], reverse=not ascending)
        
        if top_n is not None:
            rankings = rankings[:top_n]
        
        return rankings
    
    def get_best_for_application(
        self,
        application: str,
        top_n: int = 5,
    ) -> list[tuple[str, float]]:
        """Get best materials for a specific tire application.
        
        Args:
            application: Application type (passenger, performance, racing, off_road, etc.)
            top_n: Number of top materials to return
            
        Returns:
            List of (material_id, score) tuples
        """
        scores = []
        
        for mat_id, material in self._index.items():
            try:
                performance = material.predict_tire_performance(application)
                overall_score = performance.get("overall", 0)
                scores.append((mat_id, overall_score))
            except Exception:
                continue
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]
    
    def generate_report(
        self,
        material_ids: list[str] | None = None,
        format: str = "dict",
    ) -> dict[str, Any] | str:
        """Generate comprehensive materials report.
        
        Args:
            material_ids: Specific materials to report (default: all)
            format: Output format ('dict' or 'markdown')
            
        Returns:
            Report as dictionary or markdown string
        """
        if material_ids is None:
            materials = list(self._index.values())
        else:
            materials = [self._index[mid] for mid in material_ids if mid in self._index]
        
        report = {
            "summary": {
                "total_materials": len(materials),
                "by_category": {},
            },
            "materials": [],
        }
        
        # Count by category
        for cat in MaterialCategory:
            count = sum(1 for m in materials if m.category == cat)
            if count > 0:
                report["summary"]["by_category"][cat.name] = count
        
        # Add material details
        for material in materials:
            report["materials"].append({
                "id": material.material_id,
                "name": material.name,
                "category": material.category.name,
                "properties": {k.name: v.value for k, v in material.properties.items()},
            })
        
        if format == "markdown":
            return self._report_to_markdown(report)
        
        return report
    
    def _report_to_markdown(self, report: dict[str, Any]) -> str:
        """Convert report dictionary to markdown."""
        lines = [
            "# Materials Library Report",
            "",
            "## Summary",
            f"- Total materials: {report['summary']['total_materials']}",
            "",
            "### By Category",
        ]
        
        for cat, count in report["summary"]["by_category"].items():
            lines.append(f"- {cat}: {count}")
        
        lines.extend(["", "## Materials", ""])
        
        for mat in report["materials"][:20]:  # Limit for brevity
            lines.append(f"### {mat['name']} ({mat['id']})")
            lines.append(f"Category: {mat['category']}")
            lines.append("")
        
        return "\n".join(lines)
    
    def export_to_json(self, filepath: str | Path) -> None:
        """Export library to JSON file.
        
        Args:
            filepath: Output file path
        """
        filepath = Path(filepath)
        
        data = {
            "version": "1.0.0",
            "total_materials": self.total_materials,
            "materials": [],
        }
        
        for material in self._index.values():
            data["materials"].append(material.to_dict())
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported {len(data['materials'])} materials to {filepath}")


class MaterialDatabase:
    """Database interface for materials with caching and persistence."""
    
    _instance: "MaterialDatabase | None" = None
    
    def __new__(cls) -> "MaterialDatabase":
        """Singleton pattern for database."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self) -> None:
        """Initialize database."""
        if self._initialized:
            return
        
        self._library = MaterialsLibrary()
        self._cache: dict[str, Any] = {}
        self._initialized = True
    
    @property
    def library(self) -> MaterialsLibrary:
        """Get the materials library."""
        return self._library
    
    def get_material(self, material_id: str) -> Material | None:
        """Get material with caching."""
        return self._library.get(material_id)
    
    def query(
        self,
        category: MaterialCategory | None = None,
        min_tensile: float | None = None,
        max_cost: float | None = None,
        quantum_enhanced: bool | None = None,
    ) -> list[Material]:
        """Query materials with multiple criteria.
        
        Args:
            category: Filter by category
            min_tensile: Minimum tensile strength (MPa)
            max_cost: Maximum total cost (USD/kg)
            quantum_enhanced: Filter for quantum-enhanced materials
            
        Returns:
            List of matching materials
        """
        results = list(self._library)
        
        if category is not None:
            results = [m for m in results if m.category == category]
        
        if min_tensile is not None:
            results = [
                m for m in results
                if (m.get_property(PropertyType.TENSILE_STRENGTH) or 0) >= min_tensile
            ]
        
        if max_cost is not None:
            results = [
                m for m in results
                if m.economic.total_cost <= max_cost
            ]
        
        if quantum_enhanced is not None:
            results = [
                m for m in results
                if (m.category == MaterialCategory.QUANTUM_ENGINEERED) == quantum_enhanced
            ]
        
        return results


# Convenience functions

def load_material(material_id: str) -> Material | None:
    """Load a material by ID.
    
    Args:
        material_id: Material identifier
        
    Returns:
        Material if found
    """
    db = MaterialDatabase()
    return db.get_material(material_id)


def search_materials(query: str) -> list[Material]:
    """Search materials by query string.
    
    Args:
        query: Search string
        
    Returns:
        List of matching materials
    """
    db = MaterialDatabase()
    return db.library.search(query)


def get_all_materials() -> MaterialsLibrary:
    """Get the complete materials library.
    
    Returns:
        MaterialsLibrary instance
    """
    return MaterialDatabase().library


# Initialize library statistics
_lib = MaterialsLibrary()
LIBRARY_STATS = {
    "total_materials": _lib.total_materials,
    "categories": {
        "synthetic_elastomers": len(_lib.filter_by_category(MaterialCategory.SYNTHETIC_ELASTOMER)),
        "natural_rubbers": len(_lib.filter_by_category(MaterialCategory.NATURAL_RUBBER)),
        "quantum_engineered": len(_lib.filter_by_category(MaterialCategory.QUANTUM_ENGINEERED)),
        "nanoarchitectures": len(_lib.filter_by_category(MaterialCategory.NANOARCHITECTURE)),
        "self_healing": len(_lib.filter_by_category(MaterialCategory.SELF_HEALING)),
    },
}
