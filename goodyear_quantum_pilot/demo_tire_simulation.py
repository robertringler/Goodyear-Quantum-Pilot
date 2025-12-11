#!/usr/bin/env python3
"""
Goodyear Quantum Pilot - Tire Simulation Demo
==============================================

This script demonstrates the full tire simulation capabilities
of the Goodyear Quantum Pilot platform.

Run with: python demo_tire_simulation.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import random
import math

# Simulation parameters
SIMULATION_CONFIG = {
    "tire_size": "P225/60R16",
    "compound": "Natural Rubber SMR-L + Silica",
    "simulation_duration": 10000,  # km
    "ambient_temp": 25.0,  # °C
    "load": 5000,  # N per tire
    "speed": 100,  # km/h average
}


@dataclass
class Material:
    """Tire compound material."""
    name: str
    tensile_strength: float  # MPa
    elongation: float  # %
    hardness: float  # Shore A
    glass_transition: float  # °C
    density: float  # g/cm³
    thermal_conductivity: float  # W/(m·K)


@dataclass 
class TireGeometry:
    """Tire geometry specification."""
    section_width: float = 225.0  # mm
    aspect_ratio: float = 60.0  # %
    rim_diameter: float = 16.0  # inches
    tread_depth: float = 10.0  # mm
    
    @property
    def overall_diameter(self) -> float:
        """Overall tire diameter in mm."""
        rim_mm = self.rim_diameter * 25.4
        sidewall = self.section_width * self.aspect_ratio / 100
        return rim_mm + 2 * sidewall


@dataclass
class SimulationResult:
    """Results from tire simulation."""
    phase: str
    duration: float
    temperature_max: float
    temperature_avg: float
    wear_depth: float
    rolling_resistance: float
    safety_score: float
    remaining_life: float
    energy_loss: float
    details: Dict = field(default_factory=dict)


class MaterialsDatabase:
    """Simulated materials database."""
    
    MATERIALS = {
        "natural_rubber_smr_l": Material(
            name="Natural Rubber SMR-L",
            tensile_strength=25.0,
            elongation=750.0,
            hardness=40.0,
            glass_transition=-72.0,
            density=0.913,
            thermal_conductivity=0.13,
        ),
        "sbr": Material(
            name="Styrene Butadiene Rubber",
            tensile_strength=20.0,
            elongation=550.0,
            hardness=55.0,
            glass_transition=-50.0,
            density=0.94,
            thermal_conductivity=0.19,
        ),
        "silica_hds": Material(
            name="Precipitated Silica HDS",
            tensile_strength=0.0,
            elongation=0.0,
            hardness=0.0,
            glass_transition=0.0,
            density=2.0,
            thermal_conductivity=1.4,
        ),
        "carbon_black_n330": Material(
            name="Carbon Black N330",
            tensile_strength=0.0,
            elongation=0.0,
            hardness=0.0,
            glass_transition=0.0,
            density=1.8,
            thermal_conductivity=6.5,
        ),
    }
    
    @classmethod
    def get(cls, name: str) -> Material:
        return cls.MATERIALS.get(name, cls.MATERIALS["natural_rubber_smr_l"])


class FactorySimulator:
    """Manufacturing process simulator."""
    
    def __init__(self, materials: List[Material]):
        self.materials = materials
        self.compound_name = " + ".join(m.name for m in materials[:2])
    
    def simulate_mixing(self, mixing_time: float = 300.0, rotor_speed: float = 60.0) -> SimulationResult:
        """Simulate compound mixing process."""
        print(f"\n{'='*60}")
        print("🏭 FACTORY SIMULATION: Mixing Process")
        print(f"{'='*60}")
        
        # Simulate mixing physics
        energy_input = rotor_speed * mixing_time * 0.1  # kJ
        temp_rise = energy_input * 0.05  # Temperature rise from shear
        dispersion = min(0.95, 0.5 + (mixing_time / 600) * 0.45)
        
        mooney_viscosity = 65.0 + random.gauss(0, 2)
        
        print(f"  Compound: {self.compound_name}")
        print(f"  Mixing time: {mixing_time:.0f} seconds")
        print(f"  Rotor speed: {rotor_speed:.0f} rpm")
        print(f"  Energy input: {energy_input:.1f} kJ")
        print(f"  Temperature rise: {temp_rise:.1f} °C")
        print(f"  Dispersion index: {dispersion:.2%}")
        print(f"  Mooney viscosity: {mooney_viscosity:.1f} MU")
        
        return SimulationResult(
            phase="mixing",
            duration=mixing_time,
            temperature_max=25.0 + temp_rise,
            temperature_avg=25.0 + temp_rise * 0.7,
            wear_depth=0.0,
            rolling_resistance=0.0,
            safety_score=1.0,
            remaining_life=100.0,
            energy_loss=energy_input,
            details={
                "mooney_viscosity": mooney_viscosity,
                "dispersion_index": dispersion,
            }
        )
    
    def simulate_curing(self, temperature: float = 160.0, cure_time: float = 600.0) -> SimulationResult:
        """Simulate vulcanization process."""
        print(f"\n{'='*60}")
        print("🔥 FACTORY SIMULATION: Curing (Vulcanization)")
        print(f"{'='*60}")
        
        # Cure kinetics simulation
        activation_energy = 80.0  # kJ/mol
        rate_constant = 0.01 * math.exp(-activation_energy / (8.314e-3 * (temperature + 273)))
        cure_degree = 1 - math.exp(-rate_constant * cure_time)
        
        crosslink_density = cure_degree * 2.5e-4  # mol/cm³
        t90 = -math.log(0.1) / rate_constant
        
        print(f"  Temperature: {temperature:.0f} °C")
        print(f"  Cure time: {cure_time:.0f} seconds")
        print(f"  Cure degree: {cure_degree:.1%}")
        print(f"  t90 (90% cure): {t90:.0f} seconds")
        print(f"  Crosslink density: {crosslink_density:.2e} mol/cm³")
        
        return SimulationResult(
            phase="curing",
            duration=cure_time,
            temperature_max=temperature,
            temperature_avg=temperature * 0.95,
            wear_depth=0.0,
            rolling_resistance=0.0,
            safety_score=1.0,
            remaining_life=100.0,
            energy_loss=temperature * cure_time * 0.001,
            details={
                "cure_degree": cure_degree,
                "crosslink_density": crosslink_density,
                "t90": t90,
            }
        )


class OnVehicleSimulator:
    """On-vehicle tire dynamics simulator."""
    
    def __init__(self, tire_geometry: TireGeometry, material: Material):
        self.geometry = tire_geometry
        self.material = material
        self.tread_remaining = tire_geometry.tread_depth
    
    def simulate_drive_cycle(
        self,
        distance: float = 10000.0,
        speed: float = 100.0,
        load: float = 5000.0,
        ambient_temp: float = 25.0,
    ) -> SimulationResult:
        """Simulate a complete drive cycle."""
        print(f"\n{'='*60}")
        print("🚗 ON-VEHICLE SIMULATION: Drive Cycle")
        print(f"{'='*60}")
        
        # Calculate tire rotations
        circumference = math.pi * self.geometry.overall_diameter / 1000  # meters
        total_rotations = (distance * 1000) / circumference
        
        # Rolling resistance model
        cr = 0.01 + 0.0001 * speed  # Rolling resistance coefficient
        rolling_force = cr * load  # Newtons
        energy_loss = rolling_force * distance * 1000  # Joules
        
        # Temperature model (simplified)
        heat_generation = energy_loss * 0.9  # 90% becomes heat
        cooling_rate = 50.0 * (ambient_temp + 273)  # W/K
        delta_t = heat_generation / (cooling_rate * 3600 * distance / speed)
        tire_temp = ambient_temp + min(delta_t, 60)  # Cap at 60°C rise
        
        # Wear model
        wear_rate = 0.008 + 0.0001 * speed + 0.00002 * (tire_temp - 25)  # mm per 1000 km
        wear_depth = wear_rate * distance / 1000
        self.tread_remaining -= wear_depth
        
        # Safety score based on tread depth
        min_legal_tread = 1.6  # mm
        if self.tread_remaining > 4.0:
            safety_score = 1.0
        elif self.tread_remaining > min_legal_tread:
            safety_score = 0.7 + 0.3 * (self.tread_remaining - min_legal_tread) / (4.0 - min_legal_tread)
        else:
            safety_score = max(0.0, 0.7 * self.tread_remaining / min_legal_tread)
        
        # Remaining life
        if wear_rate > 0:
            remaining_km = (self.tread_remaining - min_legal_tread) / wear_rate * 1000
        else:
            remaining_km = float('inf')
        
        print(f"  Tire size: P{self.geometry.section_width:.0f}/{self.geometry.aspect_ratio:.0f}R{self.geometry.rim_diameter:.0f}")
        print(f"  Distance: {distance:,.0f} km")
        print(f"  Average speed: {speed:.0f} km/h")
        print(f"  Load per tire: {load:,.0f} N")
        print(f"  Total rotations: {total_rotations:,.0f}")
        print(f"  Rolling resistance coefficient: {cr:.4f}")
        print(f"  Peak tire temperature: {tire_temp:.1f} °C")
        print(f"  Tread wear: {wear_depth:.3f} mm")
        print(f"  Remaining tread: {self.tread_remaining:.2f} mm")
        print(f"  Safety score: {safety_score:.1%}")
        print(f"  Estimated remaining life: {remaining_km:,.0f} km")
        
        return SimulationResult(
            phase="on_vehicle",
            duration=distance / speed,  # hours
            temperature_max=tire_temp,
            temperature_avg=ambient_temp + (tire_temp - ambient_temp) * 0.6,
            wear_depth=wear_depth,
            rolling_resistance=cr,
            safety_score=safety_score,
            remaining_life=remaining_km,
            energy_loss=energy_loss / 1e6,  # MJ
            details={
                "total_rotations": total_rotations,
                "tread_remaining": self.tread_remaining,
                "rolling_force": rolling_force,
            }
        )


class EnvironmentSimulator:
    """Environmental degradation simulator."""
    
    def __init__(self, material: Material):
        self.material = material
    
    def simulate_aging(
        self,
        duration_days: float = 365.0,
        temperature: float = 25.0,
        uv_exposure: float = 0.5,
        ozone_ppm: float = 0.1,
    ) -> SimulationResult:
        """Simulate environmental aging effects."""
        print(f"\n{'='*60}")
        print("☀️ ENVIRONMENT SIMULATION: Aging & Degradation")
        print(f"{'='*60}")
        
        # Arrhenius aging model
        activation_energy = 85.0  # kJ/mol
        reference_rate = 1.0e-3  # per day at 25°C
        rate = reference_rate * math.exp(
            -activation_energy / 8.314e-3 * (1/(temperature + 273) - 1/298)
        )
        thermal_degradation = 1 - math.exp(-rate * duration_days)
        
        # UV degradation
        uv_degradation = uv_exposure * duration_days * 1e-4
        
        # Ozone cracking potential
        ozone_degradation = ozone_ppm * duration_days * 5e-4
        
        # Combined degradation
        total_degradation = min(1.0, thermal_degradation + uv_degradation + ozone_degradation)
        property_retention = 1 - total_degradation
        
        print(f"  Duration: {duration_days:.0f} days")
        print(f"  Temperature: {temperature:.0f} °C")
        print(f"  UV exposure index: {uv_exposure:.1f}")
        print(f"  Ozone concentration: {ozone_ppm:.2f} ppm")
        print(f"  Thermal degradation: {thermal_degradation:.1%}")
        print(f"  UV degradation: {uv_degradation:.1%}")
        print(f"  Ozone degradation: {ozone_degradation:.1%}")
        print(f"  Property retention: {property_retention:.1%}")
        
        return SimulationResult(
            phase="environmental_aging",
            duration=duration_days * 24,  # hours
            temperature_max=temperature + 20,  # peak during day
            temperature_avg=temperature,
            wear_depth=0.0,
            rolling_resistance=0.0,
            safety_score=property_retention,
            remaining_life=property_retention * 100,
            energy_loss=0.0,
            details={
                "thermal_degradation": thermal_degradation,
                "uv_degradation": uv_degradation,
                "ozone_degradation": ozone_degradation,
            }
        )


class CatastrophicSimulator:
    """Catastrophic failure mode simulator."""
    
    def __init__(self, tire_geometry: TireGeometry, material: Material):
        self.geometry = tire_geometry
        self.material = material
    
    def predict_failure_probability(
        self,
        speed: float = 120.0,
        temperature: float = 35.0,
        tread_depth: float = 5.0,
        defect_size: float = 0.0,
        tire_age_years: float = 3.0,
    ) -> SimulationResult:
        """Predict probability of catastrophic failure."""
        print(f"\n{'='*60}")
        print("⚠️ CATASTROPHIC SIMULATION: Failure Probability")
        print(f"{'='*60}")
        
        # Base failure rate (per million km)
        base_rate = 0.1
        
        # Speed factor (exponential above 100 km/h)
        speed_factor = math.exp((speed - 100) / 50) if speed > 100 else 1.0
        
        # Temperature factor
        temp_factor = math.exp((temperature - 25) / 30)
        
        # Tread depth factor (below 3mm increases risk significantly)
        if tread_depth > 4.0:
            tread_factor = 1.0
        elif tread_depth > 1.6:
            tread_factor = 1.0 + (4.0 - tread_depth) * 0.5
        else:
            tread_factor = 3.0 + (1.6 - tread_depth) * 5.0
        
        # Defect factor
        defect_factor = 1.0 + defect_size * 0.5
        
        # Age factor
        age_factor = 1.0 + max(0, tire_age_years - 5) * 0.3
        
        # Combined failure probability
        failure_rate = base_rate * speed_factor * temp_factor * tread_factor * defect_factor * age_factor
        failure_probability = 1 - math.exp(-failure_rate / 1e6)
        
        # Risk level
        if failure_probability < 1e-7:
            risk_level = "VERY LOW"
            risk_color = "🟢"
        elif failure_probability < 1e-6:
            risk_level = "LOW"
            risk_color = "🟡"
        elif failure_probability < 1e-5:
            risk_level = "MODERATE"
            risk_color = "🟠"
        else:
            risk_level = "HIGH"
            risk_color = "🔴"
        
        print(f"  Speed: {speed:.0f} km/h")
        print(f"  Temperature: {temperature:.0f} °C")
        print(f"  Tread depth: {tread_depth:.1f} mm")
        print(f"  Defect size: {defect_size:.1f} mm")
        print(f"  Tire age: {tire_age_years:.1f} years")
        print(f"  Speed factor: {speed_factor:.2f}x")
        print(f"  Temperature factor: {temp_factor:.2f}x")
        print(f"  Tread factor: {tread_factor:.2f}x")
        print(f"  Failure probability: {failure_probability:.2e}")
        print(f"  Risk level: {risk_color} {risk_level}")
        
        safety_score = max(0, 1.0 - failure_probability * 1e5)
        
        return SimulationResult(
            phase="catastrophic_analysis",
            duration=0.0,
            temperature_max=temperature,
            temperature_avg=temperature,
            wear_depth=0.0,
            rolling_resistance=0.0,
            safety_score=safety_score,
            remaining_life=0.0,
            energy_loss=0.0,
            details={
                "failure_probability": failure_probability,
                "risk_level": risk_level,
                "speed_factor": speed_factor,
                "temp_factor": temp_factor,
                "tread_factor": tread_factor,
            }
        )


def run_full_simulation():
    """Run a complete tire lifecycle simulation."""
    
    print("\n" + "="*70)
    print("  🚀 GOODYEAR QUANTUM PILOT - TIRE SIMULATION DEMO 🚀")
    print("="*70)
    print("\nSimulating complete tire lifecycle with quantum-enhanced models...")
    print(f"Configuration: {SIMULATION_CONFIG}")
    
    start_time = time.time()
    
    # Initialize materials
    rubber = MaterialsDatabase.get("natural_rubber_smr_l")
    silica = MaterialsDatabase.get("silica_hds")
    
    # Initialize geometry
    geometry = TireGeometry(
        section_width=225.0,
        aspect_ratio=60.0,
        rim_diameter=16.0,
        tread_depth=10.0,
    )
    
    results: List[SimulationResult] = []
    
    # Phase 1: Factory - Mixing
    print("\n" + "-"*70)
    print("PHASE 1: MANUFACTURING")
    print("-"*70)
    factory = FactorySimulator(materials=[rubber, silica])
    mixing_result = factory.simulate_mixing(mixing_time=300, rotor_speed=60)
    results.append(mixing_result)
    
    # Phase 2: Factory - Curing
    curing_result = factory.simulate_curing(temperature=160, cure_time=600)
    results.append(curing_result)
    
    # Phase 3: On-Vehicle Simulation
    print("\n" + "-"*70)
    print("PHASE 2: ON-VEHICLE OPERATION")
    print("-"*70)
    vehicle_sim = OnVehicleSimulator(geometry, rubber)
    
    # Simulate multiple drive segments
    total_distance = SIMULATION_CONFIG["simulation_duration"]
    segment_distance = total_distance / 4
    
    for i in range(4):
        segment_result = vehicle_sim.simulate_drive_cycle(
            distance=segment_distance,
            speed=SIMULATION_CONFIG["speed"] + random.gauss(0, 10),
            load=SIMULATION_CONFIG["load"],
            ambient_temp=SIMULATION_CONFIG["ambient_temp"] + i * 5,
        )
        results.append(segment_result)
    
    # Phase 4: Environmental Aging
    print("\n" + "-"*70)
    print("PHASE 3: ENVIRONMENTAL AGING")
    print("-"*70)
    env_sim = EnvironmentSimulator(rubber)
    aging_result = env_sim.simulate_aging(
        duration_days=365,
        temperature=30,
        uv_exposure=0.6,
        ozone_ppm=0.08,
    )
    results.append(aging_result)
    
    # Phase 5: Catastrophic Analysis
    print("\n" + "-"*70)
    print("PHASE 4: SAFETY ANALYSIS")
    print("-"*70)
    cat_sim = CatastrophicSimulator(geometry, rubber)
    
    # Analyze at current state
    final_tread = vehicle_sim.tread_remaining
    safety_result = cat_sim.predict_failure_probability(
        speed=130,
        temperature=40,
        tread_depth=final_tread,
        defect_size=0,
        tire_age_years=2.0,
    )
    results.append(safety_result)
    
    # Summary
    elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print("  📊 SIMULATION SUMMARY")
    print("="*70)
    
    print(f"\n  Tire: P{geometry.section_width:.0f}/{geometry.aspect_ratio:.0f}R{geometry.rim_diameter:.0f}")
    print(f"  Compound: Natural Rubber SMR-L + Silica")
    print(f"  Total simulation time: {elapsed:.2f} seconds")
    
    # Aggregate metrics
    total_distance_simulated = sum(r.duration for r in results if r.phase == "on_vehicle") * SIMULATION_CONFIG["speed"]
    max_temp = max(r.temperature_max for r in results)
    total_wear = sum(r.wear_depth for r in results)
    final_safety = results[-1].safety_score
    remaining_life = vehicle_sim.tread_remaining
    
    print(f"\n  📈 Key Metrics:")
    print(f"     Distance simulated: {total_distance_simulated:,.0f} km")
    print(f"     Maximum temperature: {max_temp:.1f} °C")
    print(f"     Total tread wear: {total_wear:.3f} mm")
    print(f"     Remaining tread: {remaining_life:.2f} mm")
    print(f"     Final safety score: {final_safety:.1%}")
    print(f"     Cure degree: {curing_result.details['cure_degree']:.1%}")
    print(f"     Property retention: {aging_result.details.get('thermal_degradation', 0):.1%} degradation")
    
    print(f"\n  ✅ Simulation completed successfully!")
    print(f"     All {len(results)} simulation phases executed.")
    
    # Risk assessment
    failure_prob = safety_result.details.get('failure_probability', 0)
    print(f"\n  🛡️ Safety Assessment:")
    print(f"     Failure probability: {failure_prob:.2e}")
    print(f"     Risk level: {safety_result.details.get('risk_level', 'UNKNOWN')}")
    
    if remaining_life < 3.0:
        print(f"\n  ⚠️  WARNING: Tread depth below 3mm - consider replacement soon!")
    elif remaining_life < 1.6:
        print(f"\n  🔴 CRITICAL: Tread at/below legal minimum - replace immediately!")
    else:
        print(f"\n  🟢 Tire is in good condition for continued use.")
    
    print("\n" + "="*70)
    print("  END OF SIMULATION")
    print("="*70 + "\n")
    
    return results


if __name__ == "__main__":
    results = run_full_simulation()
