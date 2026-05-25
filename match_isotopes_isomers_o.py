import pandas as pd
import numpy as np
import sqlite3
import os
import time
from typing import List, Tuple, Dict, Set,Optional
from collections import defaultdict
from scipy.stats import chi2
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks
from scipy.stats import linregress
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')
class MLSeriesMatcher:
    def __init__(self, min_series_length=3, max_frequency_deviation=0.1):
        self.min_series_length = min_series_length
        self.max_frequency_deviation = max_frequency_deviation
        self.known_patterns = {
            'CH3OH': {'base_interval': 12.45, 'tolerance': 0.5},
            'CH3CN': {'base_interval': 18.4, 'tolerance': 0.2},
            'CH3CCH':{'base_interval': 17.2, 'tolerance': 0.2},
            'HC3N': {'base_interval': 9.1, 'tolerance': 0.15},
            'SO2': {'base_interval': 10.32, 'tolerance': 0.25},
            'HNCO': {'base_interval': 21.98, 'tolerance': 0.2},
        }
        
        

    def find_frequency_series(self, frequencies, amplitudes):
        """Identify series of frequencies with regular intervals."""
        freq_diff = np.diff(sorted(frequencies))
        X = StandardScaler().fit_transform(freq_diff.reshape(-1, 1))
        
        db = DBSCAN(eps=0.3, min_samples=2).fit(X)
        labels = db.labels_
        
        series_groups = {}
        for label in set(labels):
            if label != -1:
                mask = (labels == label)
                interval = np.mean(freq_diff[mask])
                series_groups[label] = {
                    'interval': interval,
                    'frequencies': sorted(frequencies[:-1][mask]),
                    'amplitudes': amplitudes[:-1][mask]
                }
        
        return series_groups

    def identify_molecular_series(self, frequencies, amplitudes, species_pool):
        """Identify molecular series based on known patterns."""
        series_matches = []
        series_groups = self.find_frequency_series(frequencies, amplitudes)
        
        for group_id, group_data in series_groups.items():
            interval = group_data['interval']
            group_freqs = group_data['frequencies']
            group_amps = group_data['amplitudes']
            
            for species, pattern in self.known_patterns.items():
                if species in species_pool:
                    if abs(interval - pattern['base_interval']) <= pattern['tolerance']:
                        if self._validate_series(group_freqs, pattern['base_interval']):
                            series_matches.append({
                                'species': species,
                                'frequencies': group_freqs,
                                'amplitudes': group_amps,
                                'interval': interval,
                                'confidence': self._calculate_series_confidence(
                                    group_freqs, group_amps, pattern['base_interval']
                                )
                            })
        
        return series_matches

    def _validate_series(self, frequencies, expected_interval):
        """Validate frequency series pattern."""
        if len(frequencies) < self.min_series_length:
            return False
            
        freq_diff = np.diff(frequencies)
        diff_std = np.std(freq_diff)
        diff_mean = np.mean(freq_diff)
        
        return (diff_std / diff_mean < self.max_frequency_deviation and
                abs(diff_mean - expected_interval) / expected_interval < self.max_frequency_deviation)

    def _calculate_series_confidence(self, frequencies, amplitudes, expected_interval):
        """Calculate confidence score for series."""
        freq_diff = np.diff(frequencies)
        interval_score = 1 - np.std(freq_diff) / np.mean(freq_diff)
        
        amplitude_score = 1.0
        if len(amplitudes) > 1:
            amp_diff = np.diff(amplitudes)
            amplitude_score = 1 - np.std(amp_diff) / np.mean(amplitudes)
        
        interval_accuracy = 1 - abs(np.mean(freq_diff) - expected_interval) / expected_interval
        
        confidence = (0.4 * interval_score + 
                     0.3 * amplitude_score + 
                     0.3 * interval_accuracy)
        
        return max(0.0, min(1.0, confidence))

    

class EnhancedMolecularLineMatcher:
    def __init__(self, vlsr: float = 0.0, confidence_level: float = 0.85, ism_molecules_path: str = '/home/anastasiaf/fitswork/DO_1/ism_mol.dat',isotopes_path: str = '/home/anastasiaf/fitswork/DO_1/iso_izo.dat'):
        self.vlsr = vlsr
        self.confidence_level = confidence_level
        self.frequency_window = 0.1
        self.series_matcher = MLSeriesMatcher()
        self.ism_molecules = self._load_ism_molecules(ism_molecules_path)
        self.isotope=self._load_isotopes_molecules(isotopes_path)

        # Enhanced isotope patterns using JPL database nomenclature
        
        
        # Modified molecular families based on JPL database
        self.molecular_families = {
            'CH3OH': ['CH3OH', '13CH3OH', 'CH2DOH', 'CD3OH'],
            'CH3CCH': ['CH3CCH', '13CH3CCH'],
            'HC3N': ['HC3N', 'DC3N', 'H13CCCN', 'HC13CCN'],
            'CH3CN': ['CH3CN', '13CH3CN', 'CH2DCN', 'CD3CN'],
            'SO2': ['SO2', '34SO2', '33SO2', 'S18O2'],
            'HNCO': ['HNCO', 'HN13CO', 'DNCO', 'H15NCO'],
            'H2CO': ['H2CO', 'H213CO', 'HDCO', 'D2CO'],
            'SiO': ['SiO', '29SiO', '30SiO', 'Si18O'],
            'CO': ['CO', '13CO', 'C18O', 'C17O', '13C18O'],
            'HCN': ['HCN', 'DCN', 'H13CN', 'HC15N'],
            'HCO+': ['HCO+', 'DCO+', 'H13CO+', 'HC18O+']
        }
        
        self.max_candidates = 10
        self.refined_window_size = 0.5
        self._initialize_matching_params()
        self.unwanted_species = ['NO2','C5N','C5N-','c-CD2CH2O','HOCH2CN','c-C5H6','C4Si','c-C3D2','HCCCH2CN','SiH3CN','HOCH2C(O)NH2','c-C2H3DO','CH3C4H','CH3OCH2OH','i-C3H7CN','C6H, v=0','KCN','MgCCH','c-CCC-13-H','c-HCC-13-CH','c-C-13-CCH','c-C6H5CN','HCOOD','HOCHCHCHO','TiO2','ethyl formate''MgCCH','c-C3D','HNO3','DNO3','C2O','C-13-N','NS-34','cis-HC-13-OO','H18ONO2','C6H','C5H','C2H5OH','AlCl','C5D','HON18OO','H2O2','H15NO3','C2H5OOCH','C3H8O2 I 13C4','HCCCH2OD','C3H8O2 I 13C3','C3H8O2 I 13C5','CH3CH2C-13-N','C3O3H6 - DHA','C2H3CHO','NH2CHO','CaNC','HCCCHO','MgCN','PO', 'C2H5OH', 'C2H5CN']
        
        self.series_params = {
            'min_series_length': 3,
            'max_frequency_gap': 50.0,  # MHz
            'intensity_variation_threshold': 0.3
        }

    def _initialize_matching_params(self):
        """Initialize matching parameters with JPL-specific tolerances."""
        self.matching_levels = [
            {'freq_tolerance': 0.5, 'max_energy': 100, 'min_einstein': 1e-7},
            {'freq_tolerance': 1.0, 'max_energy': 200, 'min_einstein': 1e-8},
            {'freq_tolerance': 2.0, 'max_energy': 300, 'min_einstein': 1e-9},
            {'freq_tolerance': 7.0, 'max_energy': 900, 'min_einstein': 1e-10}
        ]
        
        # Adjusted weights for JPL database characteristics
        self.weights = {
            'frequency': 45.0,  # Increased weight for frequency matching
            'energy': 25.0,
            'einstein': 20.0,
            'intensity': 10.0
        }
    def _load_ism_molecules(self, filepath: str) -> set:
        """Load ISM molecules from file and create a set."""
        ism_molecules = set()
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    for line in f:
                        molecule = line.strip()
                        if molecule and not molecule.startswith('#'):
                            ism_molecules.add(molecule)
            else:
                print(f"Warning: ISM molecules file not found at {filepath}")
        except Exception as e:
            print(f"Error loading ISM molecules: {e}")
        return ism_molecules
    def _load_isotopes_molecules(self, filepath: str) -> set:
        """Load isotopes from file and create a set."""
        ism_molecules = set()
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    for line in f:
                        molecule = line.strip()
                        if molecule and not molecule.startswith('#'):
                            ism_molecules.add(molecule)
            else:
                print(f"Warning: ISM molecules file not found at {filepath}")
        except Exception as e:
            print(f"Error loading ISM molecules: {e}")
        return ism_molecules
    def _is_ism_molecule(self, species: str) -> bool:
       
        if any(mol in species for mol in self.ism_molecules):
            return True
        
        '''
        isotope_prefixes = ['13', '18', 'D', 'H2', 'HD']
        for mol in self.ism_molecules:
            for prefix in isotope_prefixes:
                if f"{prefix}{mol}" in species or f"{mol}-{prefix}" in species:
                    return True
        '''
        return False
    

    def _is_matched_species(self, species: str) -> tuple:
    
    # First check ISM molecules
        if species in self.ism_molecules:
            return True, "ISM", species
    
    # Then check isotopes
        if species in self.isotope:
            return True, "ISOTOP", species
    
        return False, None, None

    def _find_candidates_for_frequency(self, freq: float, database: pd.DataFrame,
                                     freq_error: float, amplitude: float) -> List[Dict]:
        """Find candidates for a specific frequency using a hierarchical matching approach."""
        freq_window = self.refined_window_size
        mask = database['adjusted_frequency'].between(
            freq - freq_window*7,
            freq + freq_window*7
        )
        
        candidates = database[mask].copy()
        if candidates.empty:
            return []
        
        results = []
        
        # First priority: Check against primary species list
        #primary_species = {'CH3OH', 'CH3CN', 'CH3CCH', 'HC3N', 'SO2', 'HNCO','H2CS','H2CO','SO','CO','HCN','HCO+','CH','CN','SiO', 'CS','C-34-S','SiS','OCS','HCO', 'CCH','NCO','CH3NC','NH3','HDO','OH', 'H2CN','SO+', 'CO+','c-C3H', 'l-C3H'}
        primary_species={
        
    'HNCO', 'H-15-NCO', 'H-13-NCO', 'HNC-17-O', 'HNC-18-O', 
    'HCOOH', 'H-13-COOH','SiO',
    'HCO+', 'DCO+', 'HC-13-O+', 'HCO-17+', 'HCO-18+',
    'NCO', '15-NCO', 'NC-13-O', 'NCO-17', 'NCO-18',
    'C-34-S','SiO', 'Si-29-O', 'Si-30-O', 'Si-33-O', 'Si-34-O', 'SiO-17', 'SiO-18',
    
    'CH3CN', 'C-13-H3CN', 'CH2DCN',
    'c-C3H', 'c-C-13-3H', 'c-C3D',
    'HC3N',
    'CH3OH',
    'SO2', 'S-33-O2', 'S-34-O2', 'SOO-18', 'SOO-17',
    
    'HCO', 'DCO', 'HC-13-O', 'HCO-17', 'HCO-18', 'HC18O',
    
    'OCS', 'O-17-CS', 'O-18-CS', 'OC-13-S', 'OCS-33', 'OCS-34',
    'SO', 'S-33-O', 'S-34-O', 'SO-17', 'SO-18',
    
    
    'H2CO', 'H2CO-18', 'HDCO', 'H2-13-CO', 'H2CO-17',
    
    'CN', 'C-13-N', 'CN-15',
    'HNO3', 'H-15-NO3', 'H-13-NO3', 'HNO3-17', 'HNO3-18',
    'NH3', 'N-15-H3', 'NHD2',
    'SO+', 'S-33-O+', 'S-34-O+', 'SO-17+', 'SO-18+',
    'CH', 'C-13-H', 'CD',
    'CCH', 'C-13-CH', 'CC-13-H', 
    'OH', 'O-17-H', 'O-18-H', 
    'CO', 'C-13-O', 'CO-17', 'CO-18',
    'CH3CCH', 
    'CS', 'C-13-S', 'CS-33', 'C-34-S', 'CS-34',
    'HDO',
    
    'H2CS', 'D2CS', 'H2C-13-S', 'H2CS-33', 'H2CS-34',
    # Add
    
    
    
    
    
    'C4H', 'C-13-CCCH', 'CC-13-CCH', 'CCC-13-CH',
    'N2H+', 'N-15-NH+', 'NN-15-H+', 'N2D+'
}
        primary_matches = []
        for _, candidate in candidates.iterrows():
            species = str(candidate['species'])
            if species in primary_species:
                score = self._calculate_transition_score(
                    abs(float(candidate['adjusted_frequency']) - freq),
                    float(candidate['lower_level_energy']),
                    float(candidate['einstein_coefficient']),
                    freq_error,
                    1.0,
                    amplitude,
                    species
                )
                # Apply higher weight for primary species
                score *= 1.2  
                
                primary_matches.append({
                    'frequency': freq,
                    'species': species,
                    'quantum_numbers': str(candidate['upper_level_quantum_numbers']).replace(" ", "-"),
                    'database_frequency': float(candidate['adjusted_frequency']),
                    'einstein_coefficient': float(candidate['einstein_coefficient']),
                    'upper_level_energy': float(candidate['upper_level_energy']),
                    'uncertainty': float(candidate['uncertainty']),
                    'score': score,
                    'match_type': "PRIMARY",
                    'parent_molecule': species
                })
        
        if primary_matches:
            return primary_matches
        
        # Second priority: Check against secondary file
        try:
            with open('/home/anastasiaf/DATABASE/species_united.dat', 'r') as f:
                secondary_species = {line.strip() for line in f if line.strip()}
                
            secondary_matches = []
            for _, candidate in candidates.iterrows():
                species = str(candidate['species'])
                if species in secondary_species:
                    score = self._calculate_transition_score(
                        abs(float(candidate['adjusted_frequency']) - freq),
                        float(candidate['lower_level_energy']),
                        float(candidate['einstein_coefficient']),
                        freq_error,
                        1.0,
                        amplitude,
                        species
                    )
                    
                    secondary_matches.append({
                        'frequency': freq,
                        'species': species,
                        'quantum_numbers': str(candidate['upper_level_quantum_numbers']).replace(" ", "-"),
                        'database_frequency': float(candidate['adjusted_frequency']),
                        'einstein_coefficient': float(candidate['einstein_coefficient']),
                        'upper_level_energy': float(candidate['upper_level_energy']),
                        'uncertainty': float(candidate['uncertainty']),
                        'score': score,
                        'match_type': "SECONDARY",
                        'parent_molecule': species
                    })
                    
            if secondary_matches:
                return secondary_matches
        except FileNotFoundError:
            print("Warning: Secondary species file not found")
        
        # Third priority: Check against ISM molecules
        ism_matches = []
        for _, candidate in candidates.iterrows():
            species = str(candidate['species'])
            if species in self.ism_molecules:
                score = self._calculate_transition_score(
                    abs(float(candidate['adjusted_frequency']) - freq),
                    float(candidate['lower_level_energy']),
                    float(candidate['einstein_coefficient']),
                    freq_error,
                    1.0,
                    amplitude,
                    species
                )
                
                ism_matches.append({
                    'frequency': freq,
                    'species': species,
                    'quantum_numbers': str(candidate['upper_level_quantum_numbers']).replace(" ", "-"),
                    'database_frequency': float(candidate['adjusted_frequency']),
                    'einstein_coefficient': float(candidate['einstein_coefficient']),
                    'upper_level_energy': float(candidate['upper_level_energy']),
                    'uncertainty': float(candidate['uncertainty']),
                    'score': score,
                    'match_type': "ISM",
                    'parent_molecule': species
                })
        
        if ism_matches:
            return ism_matches
        
        # If no matches found in any category, return empty list
        return []
    def _create_unmatched_candidate(self, freq: float) -> Dict:
        """Create an unmatched candidate dictionary."""
        return {
            'frequency': freq,
            'species': "UNMATCHED",
            'quantum_numbers': "N/A",
            'database_frequency': 0.0,
            'einstein_coefficient': 0.0,
            'upper_level_energy': 0.0,
            'uncertainty': 0.0,
            'score': 0.0,
            'match_type': None,
            'parent_molecule': None
        }
    def _find_multiple_candidates(self, data_lines: pd.DataFrame, database: pd.DataFrame) -> List[List[Dict]]:
        """Modified to prioritize ISM molecules and their isotopologs."""
        working_db = database[~database['species'].isin(self.unwanted_species)].copy()
        working_db.loc[:, 'FREQUENCY'] = pd.to_numeric(working_db['frequency'], errors='coerce')
        
        all_candidates = []
        
        for idx, line in data_lines.iterrows():
            observed_freq = float(line['FREQUENCY'])
            candidates = self._find_candidates_for_frequency(
                observed_freq,
                working_db,
                float(line.get('center_error', 0.002)) * 1e3,
                float(line.get('amplitude', 1.0))
            )
            
            # Sort candidates by match type and score
            sorted_candidates = sorted(
                candidates,
                key=lambda x: (
                    2 if x.get('match_type') == 'ISM' else
                    1 if x.get('match_type') == 'isotopolog' else 0,
                    x.get('score', 0)
                ),
                reverse=True
            )
            
            all_candidates.append(sorted_candidates[:self.max_candidates] if sorted_candidates else
                                [self._create_unmatched_candidate(observed_freq)])
        
        return all_candidates

    

    

    def match_frequencies_with_candidates(self, data_lines: pd.DataFrame, database: pd.DataFrame) -> Tuple[List[Tuple], List[List[Dict]]]:
        
        try:
            initial_matches = self._perform_initial_matching(data_lines, database)
            candidate_matches = self._find_multiple_candidates(data_lines, database)
            
            # Identify molecular series for unmatched frequencies
            unmatched_freqs = []
            unmatched_amps = []
            unmatched_indices = []
            
            for i, match in enumerate(initial_matches):
                if match[1] == "UNMATCHED":
                    unmatched_freqs.append(match[0])
                    unmatched_amps.append(data_lines['amplitude'].iloc[i])
                    unmatched_indices.append(i)
            
            if unmatched_freqs:
                species_pool = set(database['species'].unique())
                series_matches = self.series_matcher.identify_molecular_series(
                    np.array(unmatched_freqs),
                    np.array(unmatched_amps),
                    species_pool
                )
                
                matched_results = initial_matches.copy()
                for series in series_matches:
                    if series['confidence'] > 0.6:
                        for freq, amp in zip(series['frequencies'], series['amplitudes']):
                            try:
                                freq_idx = unmatched_freqs.index(freq)
                                orig_idx = unmatched_indices[freq_idx]
                                
                                closest_transition = self._find_closest_transition(
                                    freq, series['species'], database
                                )
                                
                                if closest_transition is not None:
                                    matched_results[orig_idx] = self._create_match_tuple(
                                        freq,
                                        closest_transition,
                                        series['confidence']
                                    )
                            except ValueError:
                                continue
                
                return matched_results, candidate_matches
            
            return initial_matches, candidate_matches
            
        except Exception as e:
            print(f"Error in frequency matching: {str(e)}")
            return [], []

    
    
    def _initialize_matching_params(self):
        """Initialize matching parameters with relaxed tolerances."""
        self.matching_levels = [
            {'freq_tolerance': 1.0, 'max_energy': 100, 'min_einstein': 1e-6},
            {'freq_tolerance': 1.5, 'max_energy': 200, 'min_einstein': 1e-7},
            {'freq_tolerance': 2.0, 'max_energy': 200, 'min_einstein': 1e-8},
            {'freq_tolerance': 2.5, 'max_energy': 600, 'min_einstein': 1e-9},
            {'freq_tolerance': 3.0, 'max_energy': 1000, 'min_einstein': 1e-10}
        ]
        
        self.weights = {
            'frequency': 40.0,
            'energy': 30.0,
            'einstein': 20.0,
            'intensity': 10.0
        }

    def _calculate_doppler_shift(self, freq: float, velocity_uncertainty: float = 0.2) -> Tuple[float, float]:
        """Calculate Doppler-shifted frequency with uncertainty."""
        c = 2.99792458e5
        shifted_freq = freq*1#freq / (1 - /c)
        freq_uncertainty = freq * velocity_uncertainty/c
        return shifted_freq, freq_uncertainty

    def _calculate_transition_score(self, freq_diff: float, energy: float, 
                                 einstein: float, uncertainty: float,
                                 intensity_ratio: float, amplitude: float,species: str
                                 ) -> float:
        
        # Check species in ism.dat
        
        
        try:
            
            
            
            score=0.0
            with open('/home/anastasiaf/fitswork/DO_1/ism_mol.dat', 'r') as f:
                    if species in f.read():
                        score += 5.0
            energy_score = np.exp(-energy/100)
            einstein_score = 1 - np.exp(einstein)
            score = (self.weights['energy'] * energy_score +
                self.weights['einstein'] * einstein_score)
                
            
            
            return max(0.0, score)
        except Exception as e:
            print(f"Error in score calculation: {str(e)}")
            return 0.0
    
    

   
    def _perform_initial_matching(self, data_lines: pd.DataFrame, database: pd.DataFrame) -> List[Tuple]:
        
        working_db = database[~database['species'].isin(self.unwanted_species)].copy()
        working_db.loc[:, 'FREQUENCY'] = pd.to_numeric(data_lines['FREQUENCY'], errors='coerce')
        
        doppler_results = working_db['frequency'].apply(self._calculate_doppler_shift)
        working_db.loc[:, 'adjusted_frequency'] = doppler_results.apply(lambda x: x[0])
        working_db.loc[:, 'freq_uncertainty'] = doppler_results.apply(lambda x: x[1])
        
        matched_results = []
        remaining_lines = data_lines.copy()
        
        for criteria in self.matching_levels:
            if remaining_lines.empty:
                break
            
            new_matched_results = []
            unmatched_lines = []
            
            for idx, line in remaining_lines.iterrows():
                observed_freq = float(line['FREQUENCY'])
                freq_error = float(line.get('center_error', 0.02)) * 1e3
                
                # Use line width to define search window, but limit it to reasonable bounds
                line_width = float(line.get('width', 0.0)) * 1e3  # Convert to MHz
                # Limit the frequency window to the minimum of:
                # 1. The line width
                # 2. The criteria's frequency tolerance
                # 3. A maximum of 5 MHz
                freq_window = min(
                    max(line_width, freq_error),  # At least as large as frequency error
                    criteria['freq_tolerance'],      # No larger than tolerance
                    5.0                             # Hard limit of 5 MHz
                )
                
                mask = (
                    (working_db['adjusted_frequency'].between(
                        observed_freq - freq_window, 
                        observed_freq + freq_window
                    )) &
                    (working_db['upper_level_energy'] <= criteria['max_energy']) &
                    (working_db['einstein_coefficient'] >= criteria['min_einstein'])
                )
                
                candidates = working_db[mask].copy()
                
                if not candidates.empty:
                    candidates.loc[:, 'total_uncertainty'] = np.sqrt(
                        (freq_error * 1.5)**2 +
                        candidates['freq_uncertainty']**2 +
                        candidates['uncertainty']**2
                    )
                    
                    candidates.loc[:, 'score'] = candidates.apply(
                        lambda x: self._calculate_transition_score(
                            abs(float(x['adjusted_frequency']) - observed_freq),
                            float(x['upper_level_energy']),
                            float(x['einstein_coefficient']),
                            float(x['total_uncertainty']),
                            1.0,
                            float(line.get('amplitude', 1.0)),
                       str(candidates['species']) ),
                        axis=1
                    )
                    
                    # Increase score threshold for stricter matching
                    if candidates['score'].max() > 0.1:  # Increased from 0.3
                        best_match = candidates.loc[candidates['score'].idxmax()]
                        new_matched_results.append(
                            self._create_match_tuple(
                                observed_freq, 
                                best_match, 
                                float(best_match['score'])
                            )
                        )
                        continue
                
                # Only check molecular families if no good direct matches found
                family_match = self._check_molecular_families(
                    observed_freq, 
                    working_db, 
                    freq_window  # Use same limited window for family matching
                )
                if family_match:
                    new_matched_results.append(family_match)
                    continue
                
                unmatched_lines.append(line)
            
            matched_results.extend(new_matched_results)
            remaining_lines = pd.DataFrame(unmatched_lines)
        
        return matched_results

    def _find_closest_transition(self, frequency: float, species: str, database: pd.DataFrame) -> pd.Series:
        """Find closest matching transition in database."""
        species_data = database[database['species'] == species].copy()
        if species_data.empty:
            return None
        
        species_data['freq_diff'] = abs(species_data['frequency'] - frequency)
        return species_data.loc[species_data['freq_diff'].idxmin()]

    def _check_molecular_families(self, observed_freq: float, 
                                 database: pd.DataFrame, 
                                 freq_window: float) -> Tuple:
        """Check molecular families with improved physical constraints."""
        best_family_match = None
        best_family_score = 0.0
        
        for family, species in self.molecular_families.items():
            for sp in species:
                mask = (
                    (database['species'] == sp) &
                    (database['adjusted_frequency'].between(
                        observed_freq - freq_window*2,  # Slightly wider window for families
                        observed_freq + freq_window*2
                    ))
                )
                candidates = database[mask].sort_values('einstein_coefficient', ascending=False)
                
                if not candidates.empty:
                    candidate = candidates.iloc[0]
                    freq_diff = abs(float(candidate['adjusted_frequency']) - observed_freq)
                    score = self._calculate_transition_score(
                        freq_diff,
                        float(candidate['upper_level_energy']),
                        float(candidate['einstein_coefficient']),
                        freq_window,
                        1.0,
                        1.0,
                        candidate['species']
                    )
                    
                    if score > best_family_score:
                        best_family_score = score
                        best_family_match = candidate
        
        if best_family_match is not None and best_family_score > 0.3:
            return self._create_match_tuple(
                observed_freq,
                best_family_match,
                best_family_score * 0.8  # Penalty for family matches
            )
        return None

    def _create_match_tuple(self, observed_freq: float, match: pd.Series, 
                             score: float) -> Tuple:
        """Create standardized match tuple with ISM molecule checking."""
        try:
            species = str(match['species'])
            
            # Check if the species is in the ISM molecules list
            if not self._is_ism_molecule(species):
                species += "?"
            
            return (
                float(observed_freq),
                species,  # Modified species name
                str(match['upper_level_quantum_numbers']).replace(" ", "-"),
                float(match['adjusted_frequency']),
                float(match['einstein_coefficient']),
                float(match['upper_level_energy']),
                float(match.get('uncertainty', 0.0)),
                float(score)
            )
        except Exception as e:
            print(f"Error creating match tuple: {str(e)}")
            return self._create_unmatched_tuple(observed_freq)

    def _create_unmatched_tuple(self, observed_freq: float) -> Tuple:
        """Create unmatched line tuple."""
        return (
            float(observed_freq),
            "UNMATCHED",
            "N/A",
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        )
def get_molecular_equivalents():
    """
    Returns a dictionary of equivalent molecular notations.
    Keys are standardized forms, values are lists of alternative notations.
    """
    return {
        'CCH': ['C2H', 'ETHYNYL'],
        'CH3COOH': ['CH3CO2H', 'ACETIC ACID', 'CH3COOH'],
        'HCOOH': ['FORMIC ACID', 'HCO2H'],
        'NH2CHO': ['FORMAMIDE', 'HCONH2'],
        'CH3OCH3': ['DIMETHYL ETHER', '(CH3)2O', 'DME'],
        'CH3OH': ['METHANOL', 'CH3OH'],
        'H2CO': ['FORMALDEHYDE', 'HCHO'],
        'CH3CH2OH': ['ETHANOL', 'C2H5OH'],
        'CH3CN': ['METHYL CYANIDE', 'ACETONITRILE'],
        'HC3N': ['CYANOACETYLENE', 'HCCCN'],
        'CH3CHO': ['ACETALDEHYDE', 'CH3COH'],
        'SO2': ['SULFUR DIOXIDE'],
        'HCS+': ['THIOFORMYL CATION'],
        'C2H5CN': ['PROPIONITRILE', 'ETHYL CYANIDE'],
        'CH2CO': ['KETENE', 'H2CCO'],
        'HNCO': ['ISOCYANIC ACID'],
        'CH3CCH': ['PROPYNE', 'CH3C2H'],
        'H2CS': ['THIOFORMALDEHYDE'],
        'NH2CN': ['CYANAMIDE'],
        'CH2NH': ['METHYLENIMINE', 'CH2=NH'],
        'C2H3CN': ['VINYL CYANIDE', 'ACRYLONITRILE']
    }

def standardize_molecule_name(name, equivalents_dict):
    """
    Convert a molecule name to its standard form using the equivalents dictionary.
    
    Args:
        name (str): The molecule name to standardize
        equivalents_dict (dict): Dictionary of equivalent molecular notations
        
    Returns:
        str: Standardized molecule name
    """
    name = name.upper().strip()
    
    # Check if this name is a standard form
    if name in equivalents_dict:
        return name
        
    # Check if this name is an alternative form
    for standard_name, alternatives in equivalents_dict.items():
        if name in [alt.upper() for alt in alternatives]:
            return standard_name
            
    return name
def format_quantum_numbers_spectroscopic(quantum_str):
    """
    Convert spectroscopic quantum number notation to LaTeX format following standard spectroscopic conventions.
    Handles various quantum number formats including J/N, Ka/Kc, v, and F quantum numbers.
    
    Args:
        quantum_str (str): Quantum number string from database
    
    Returns:
        str: LaTeX formatted quantum number string
    """
    try:
        # Split the quantum numbers into parts
        parts = quantum_str.strip().split()
        if len(parts) < 2:
            parts = quantum_str.split('-')
        
        # Extract upper and lower state numbers if they exist
        if len(parts) >= 2:
            # Handle cases where numbers are grouped in threes (J, Ka, Kc)
            upper_parts = []
            lower_parts = []
            
            # Try to identify if we have grouped numbers (like 325 -> 32,5)
            for part in parts:
                if len(part) == 3 and part.isdigit():
                    # Split into J and K numbers
                    j_num = part[:2]
                    k_num = part[2]
                    if int(j_num) <= 99:
                        upper_parts.extend([j_num, k_num])
                    else:
                        # Convert numbers > 99 using uppercase letters
                        letter_num = chr(ord('A') + (int(j_num) - 100))
                        upper_parts.extend([letter_num, k_num])
                else:
                    upper_parts.append(part)
            
            # Format the quantum numbers in spectroscopic notation
            if len(upper_parts) >= 3:
                # Format for asymmetric top (J Ka Kc)
                return f"{upper_parts[0]}_{{{upper_parts[1]},{upper_parts[2]}}}"
            elif len(upper_parts) == 2:
                # Format for symmetric top (J K)
                return f"{upper_parts[0]}_{{{upper_parts[1]}}}"
            else:
                # Format for simple rotation (J)
                return upper_parts[0]
                
        return quantum_str
        
    except Exception as e:
        print(f"Error formatting quantum numbers '{quantum_str}': {e}")
        return quantum_str
            
def main():
    try:
        # Initialize matcher
        i=4
        rest_freq=243000
        matcher = EnhancedMolecularLineMatcher(vlsr=0.0, confidence_level=0.90)
        #params_file = '/home/anastasiaf/RCW120/resampled/B'+str(i)+'_resample_frequency_baseline_removed_gauss.dat'
        #params_file = '/home/anastasiaf/RCW120/resampled/gauss/B'+str(i)+'_resampled_frequency_baseline_removed_gauss_f.dat'
        params_file='/home/anastasiaf/RCW120/CORE2/CORE2_FREQ_RESAMPLED/baseline/CORE2_B'+str(i)+'_res_freq_baseline_gauss.dat'
        output_file = '/home/anastasiaf/RCW120/CORE2/CORE2_FREQ_RESAMPLED/baseline/CORE2_B'+str(i)+'_resample_frequency_baseline_removed_matched30.dat'
        candidates_file = '/home/anastasiaf/RCW120/CORE2/CORE2_FREQ_RESAMPLED/baseline/CORE2_B'+str(i)+'_resample_frequency_baseline_removed_candidates30.dat'
        latex_output_file = '/home/anastasiaf/RCW120/CORE2/CORE2_FREQ_RESAMPLED/baseline/CORE2_B'+str(i)+'_resample_frequency_baseline_removed_latex30.tex'
        # Load data
        #params_file = '/home/anastasiaf/fitswork/DO_1/B4_smoothed_spectral_line_fits_+7_afa.txt'
        #output_file = '/home/anastasiaf/fitswork/DO/spectral_line_fits_B4_smoothed_matched.txt'
        #candidates_file = '/home/anastasiaf/fitswork/DO/spectral_line_fits_B4_smoothed_candidates.txt'
        #latex_output_file = '/home/anastasiaf/fitswork/DO/spectral_line_fits_B4_smoothed_latex.tex'
        #params_file = '/home/anastasiaf/fitswork/DO_1/B2_smoothed_spectral_line_fits_+7_afa.txt'
        #output_file = '/home/anastasiaf/fitswork/DO/spectral_line_fits_RCW120_CORE2_B2_matched_+7_again.txt'
        #candidates_file = '/home/anastasiaf/fitswork/DO/spectral_line_fits_RCW120_CORE2_B2_candidates_+7_again.txt'
        #latex_output_file = '/home/anastasiaf/fitswork/DO/spectral_line_fits_RCW120_CORE2_B2_latex_+7_again.tex'
        # Get molecular equivalents
        molecular_equivalents = get_molecular_equivalents()
        
        # Read Gaussian parameters
        gaussian_params = []
        with open(params_file, 'r') as f:
            current_params = {}
            for line in f:
                try:
                    if 'Amplitude' in line:
                        parts = line.split('±')
                        
                        current_params['amplitude'] = float(parts[0].split('=')[1].strip())
                        current_params['amplitude_error'] = float(parts[1].strip()) if len(parts) > 1 else 0.0
                    elif 'Center' in line:
                        parts = line.split('±')
                        current_params['center'] = float(parts[0].split('=')[1].strip())
                        current_params['center_error'] = float(parts[1].strip()) if len(parts) > 1 else 0.0
                    elif 'Width' in line:
                        parts = line.split('±')
                        current_params['width'] = float(parts[0].split('=')[1].strip())
                        current_params['width_error'] = float(parts[1].strip()) if len(parts) > 1 else 0.0
                    elif 'Area' in line:
                        parts = line.split('±')
                        current_params['area'] = float(parts[0].split('=')[1].strip())
                        current_params['area_error'] = float(parts[1].strip()) if len(parts) > 1 else 0.0
                        gaussian_params.append(current_params.copy())
                        current_params = {}
                except ValueError as e:
                    print(f"Error parsing line: {line.strip()}, Error: {e}")
                    continue
        
        # Convert to DataFrame
        data_lines = pd.DataFrame(gaussian_params)
        print("\nData lines columns:", data_lines.columns.tolist())
        print("Data lines shape:", data_lines.shape)
        
        data_lines = data_lines.rename(columns={'center': 'FREQUENCY'})
        data_lines['FREQUENCY'] = data_lines['FREQUENCY'] * 1e3  # Convert to MHz
        data_lines['FREQUENCY'] = data_lines['FREQUENCY'] #* (1-((-7)/2.99792458e5))

        conn = sqlite3.connect('/home/anastasiaf/DATABASE/myjpl.db')
        conn_cdms = sqlite3.connect('/home/anastasiaf/CDMS_DATABASE/my_cdms.db')
                
        # The SQL query remains the same
        query = """
        SELECT DISTINCT
            species,
            frequency,
            COALESCE(uncertainty, 0.1) as uncertainty,
            einstein_coefficient,
            upper_level_energy,
            upper_level_statistical_weight,
            upper_level_quantum_numbers,
            lower_level_energy,
            lower_level_statistical_weight,
            lower_level_quantum_numbers,
            origin,
            dbsource,
            date,
            CASE 
                -- Extended isotopomers list
                
                        
                -- Carbon isotopologs
                WHEN species LIKE '%13C%' OR species LIKE '%C13%' OR 
                     species LIKE 'c13%' OR species LIKE '%c13' OR
                     species LIKE '%c-13%' OR species LIKE '%13c%' THEN 'C13_isotopolog'
                -- Oxygen isotopologs
                
                
                -- Sulfur isotopologs
                WHEN species LIKE '%34S%' OR species LIKE '%S34%' OR 
                     species LIKE 's34%' OR species LIKE '%s34' OR
                     species LIKE '%s-34%' OR species LIKE '%34s%' OR
                     species LIKE 'c34s' THEN 'S34_isotopolog'
                WHEN species LIKE '%33S%' OR species LIKE '%S33%' OR 
                     species LIKE 's33%' OR species LIKE '%s33' OR
                     species LIKE '%s-33%' OR species LIKE '%33s%' OR
                     species LIKE 'c33s' THEN 'S33_isotopolog'
                
                
                ELSE 'main_isotopolog'
            END as isotope_type
        FROM line
        WHERE upper_level_energy <= 1000
        ORDER BY species, frequency
        """

        # First, query the CDMS database only
        print("Querying CDMS database first...")
        database2 = pd.read_sql_query(query, conn_cdms)
        print(f"Found {len(database2)} entries in CDMS database")

        # Process CDMS data alone first
        database = database2.copy()
        print("\nCDMS Database columns:", database.columns.tolist())
        print("CDMS Database shape:", database.shape)

        # Standardize molecule names in the database
        database['standard_species'] = database['species'].apply(
            lambda x: standardize_molecule_name(x, molecular_equivalents)
        )

        # Group equivalent molecules together
        database['molecular_group'] = database['standard_species'].apply(
            lambda x: x.split('-')[0] if '-' in x else x
        )

        # Prepare database for matching
        database = database.drop_duplicates()
        database.loc[database['uncertainty'] <= 0, 'uncertainty'] = 0.1
        database['adjusted_frequency'] = database['frequency']

        print("\nPrepared CDMS database columns:", database.columns.tolist())

        # Match frequencies with candidates from CDMS
        print("\nAttempting frequency matching with CDMS database...")
        matched_results, candidate_matches = matcher.match_frequencies_with_candidates(data_lines, database)
        print(f"Number of matches found in CDMS: {len(matched_results)}")

        # Check if we need to use JPL database
        unmatched_indices = []
        for i, matches in enumerate(candidate_matches):
            if matches[0]['species'] == "UNMATCHED":
                if i < len(data_lines):
                    unmatched_indices.append(i)

        # If there are unmatched lines, use JPL database for those lines
        if len(unmatched_indices) > 0:
            print(f"\nFound {len(unmatched_indices)} unmatched lines, searching JPL database for these...")
            unmatched_df = data_lines.iloc[unmatched_indices].copy()
            
            # Query JPL database
            database1 = pd.read_sql_query(query, conn)
            print(f"Found {len(database1)} entries in JPL database")
            
            # Prepare JPL database
            jpl_database = database1.copy()
            jpl_database['standard_species'] = jpl_database['species'].apply(
                lambda x: standardize_molecule_name(x, molecular_equivalents)
            )
            jpl_database['molecular_group'] = jpl_database['standard_species'].apply(
                lambda x: x.split('-')[0] if '-' in x else x
            )
            jpl_database = jpl_database.drop_duplicates()
            jpl_database.loc[jpl_database['uncertainty'] <= 0, 'uncertainty'] = 0.1
            jpl_database['adjusted_frequency'] = jpl_database['frequency']
            
            # Match unmatched frequencies with JPL database
            print("\nAttempting frequency matching with JPL database for unmatched lines...")
            jpl_matched_results, jpl_candidate_matches = matcher.match_frequencies_with_candidates(unmatched_df, jpl_database)
            print(f"Number of additional matches found in JPL: {len(jpl_matched_results)}")
            
            # Merge the JPL results back into the original results
            # We need to update the original matched_results and candidate_matches with the new matches
            for i, original_idx in enumerate(unmatched_indices):
                if i < len(jpl_candidate_matches):
                    # Use the original index we already know
                    if original_idx < len(candidate_matches):
                        candidate_matches[original_idx] = jpl_candidate_matches[i]

        conn.close()
        conn_cdms.close()

        # Process results with standardized names
        results_list = []
        for i, match in enumerate(candidate_matches):
            if i < len(data_lines):
                candidates = match
                for candidate in candidates:
                    if candidate['species'] != "UNMATCHED":
                        std_species = standardize_molecule_name(
                            candidate['species'], 
                            molecular_equivalents
                        )
                        results_list.append({
                            'frequency': candidate['frequency'],
                            'amplitude': data_lines['amplitude'].iloc[i],
                            'amplitude_error': data_lines['amplitude_error'].iloc[i],
                            'original_species': candidate['species'],
                            'standard_species': std_species,
                            'quantum_numbers': candidate['quantum_numbers'],
                            'database_frequency': candidate['database_frequency'],
                            'einstein_coefficient': candidate['einstein_coefficient'],
                            'upper_level_energy': candidate['upper_level_energy'],
                            'uncertainty': candidate.get('uncertainty', 0.1),
                            'confidence_score': candidate['score'],
                            'width': data_lines['width'].iloc[i],
                            'width_error': data_lines['width_error'].iloc[i],
                            'area': data_lines['area'].iloc[i],
                            'area_error': data_lines['area_error'].iloc[i],
                            'diff': abs(candidate['frequency'] - candidate['database_frequency'])
                        })

        # Convert to DataFrame
        results_df = pd.DataFrame(results_list)
        print("\nResults DataFrame columns:", results_df.columns.tolist())
        print("Results DataFrame shape:", results_df.shape)

        if len(results_df) == 0:
            print("Warning: No results found in matching process")
            return
            
        # Sort and group results - keep the same logic as in your original code
        best_confidence_idx = results_df.groupby('frequency')['einstein_coefficient'].idxmax()
        best_matches_df = results_df.loc[best_confidence_idx]
        best_matches_df = best_matches_df.loc[best_matches_df.groupby('frequency')['upper_level_energy'].idxmin()]
        best_matches_df = best_matches_df.loc[best_matches_df.groupby('frequency')['diff'].idxmin()]
        best_matches_df = best_matches_df.loc[best_matches_df.groupby('frequency')['confidence_score'].idxmin()]

        # Write results to files
        with open(output_file, 'w') as f:
            f.write("FREQUENCY(MHz)  SPECIES  Q.Numbers   Amp±Error   Width±Error   Adj.Freq(MHz)   Score\n")

            for _, row in best_matches_df.iterrows():
                species_info = f"{row['original_species']}"
                f.write(f"{row['frequency']:.3f}  {species_info:25s}  {row['quantum_numbers']:15s}  "
                        f"{row['amplitude']:.2e}±{row['amplitude_error']:.2e}  "
                        f"{row['width']:.2f}±{row['width_error']:.2f}  "
                        f"{row['database_frequency']:.3f}  "
                        f"{row['confidence_score']:.3f}\n")

        print(f"\nResults written to {output_file}")
           
        # Write candidate matches
        with open(candidates_file, 'w') as f:
            f.write("FREQUENCY(MHz)  RANK  SPECIES  Q.Numbers  Adj.Freq(MHz)  Einstein  Upper_Energy  Score  Freq_Diff(MHz)  Isotopolog_Info\n")

            for freq_candidates in candidate_matches:
                if freq_candidates[0]['species'] != "UNMATCHED":
                    for rank, candidate in enumerate(freq_candidates, 1):
                        freq_diff = abs(candidate['frequency'] - candidate['database_frequency'])
                        isotopolog_info = ""
                        if candidate.get('is_isotopolog', False):
                            isotopolog_info = f"[{candidate['parent_species']} isotopolog]"
                        
                        f.write(f"{candidate['frequency']:.3f}  {rank:2d}  {candidate['species']:10s}  "
                               f"{candidate['quantum_numbers']:15s}  {candidate['database_frequency']:.3f}  "
                               f"{candidate['einstein_coefficient']:.2e}  {candidate['upper_level_energy']:.2f}  "
                               f"{candidate['score']:.3f}  {freq_diff:.3f}  {isotopolog_info}\n")
                    f.write("\n")
                else:
                    f.write(f"{freq_candidates[0]['frequency']:.3f}  --  UNMATCHED  {'':15s}  {'':10s}  {'':8s}  {'':8s}  0.000  0.000\n\n")
        
        
        
        def format_quantum_numbers(quantum_str):
            """
            Convert quantum number notation to LaTeX format
            Examples: 
            '20-119-6' becomes '20 --- 19'
            '13-113-1' becomes '13 --- 14'
            '5-3---2' becomes '5_{3} --- 4_{3}'
            '14-9-5-0' becomes '14 --- 13'
            '10-0101110' becomes '10 --- 9'
            """
            try:
                # First try to extract the initial number from the string
                first_num = None
                parts = quantum_str.split('-')
                if parts and parts[0].isdigit():
                    first_num = int(parts[0])
                
                # Handle cases with explicit '---' notation
                if '---' in quantum_str:
                    parts = quantum_str.split('---')
                    if len(parts) == 2:
                        left = parts[0].split('-')
                        right = parts[1].split('-')
                        if len(left) >= 2:
                            first_num = left[0]
                            second_num = left[1]
                            first_num_right = str(int(first_num) - 1)
                            second_num_right = right[0] if right else second_num
                            return f"{first_num}_{{{second_num}}} --- {first_num_right}_{{{second_num_right}}}"
                
                # Handle cases like '20-119-6'
                elif quantum_str.count('-') == 2 and len(parts[0]) <= 2:
                    second_part = parts[1]
                    if second_part.startswith('11'):
                        # For patterns like '13-113-1', increment the number
                        return f"{first_num}$ - ${first_num + 1}"
                    else:
                        # For patterns like '20-119-6', decrement the number
                        return f"{first_num}$ - ${first_num - 1}"
                
                # Default case: if we have a first number, use "number --- (number-1)" format
                elif first_num is not None:
                    return f"{first_num}$ - ${first_num - 1}"
                
                # If we couldn't parse it at all, return original
                return quantum_str
                
            except Exception as e:
                print(f"Error formatting quantum numbers '{quantum_str}': {e}")
                return quantum_str
        # Find this code in your main() function where the LaTeX table is being created and replace it with this updated version:

        # Find this code in your main() function where the LaTeX table is being created and replace it with this updated version:

        filtered_df = best_matches_df[best_matches_df['amplitude'] >= 0.016].copy()

# Function to clean molecule names by removing comma and everything after it
        def clean_molecule_name(name):
            return name.split(',')[0].strip()

# Before generating the LaTeX table, we need to add catalog information
# Create a dictionary to track which molecules are in which catalogs
        molecule_catalog_map = {}

# Process database1 (JPL)
        for species in database1['species'].unique():
            clean_species = clean_molecule_name(species)
            std_species = standardize_molecule_name(clean_species, molecular_equivalents)
            if std_species not in molecule_catalog_map:
                molecule_catalog_map[std_species] = {'jpl': True, 'cdms': False}
            else:
                molecule_catalog_map[std_species]['jpl'] = True

# Process database2 (CDMS)
        for species in database2['species'].unique():
            clean_species = clean_molecule_name(species)
            std_species = standardize_molecule_name(clean_species, molecular_equivalents)
            if std_species not in molecule_catalog_map:
                molecule_catalog_map[std_species] = {'jpl': False, 'cdms': True}
            else:
                molecule_catalog_map[std_species]['cdms'] = True

# Add catalog information to filtered_df
        filtered_df['clean_species'] = filtered_df['original_species'].apply(clean_molecule_name)
        filtered_df['clean_standard_species'] = filtered_df['clean_species'].apply(
            lambda x: standardize_molecule_name(x, molecular_equivalents)
        )

        filtered_df['catalog_source'] = filtered_df.apply(
            lambda row: 'JPL' if row.get('dbsource') == 'JPL' else 
                        ('CDMS' if row.get('dbsource') == 'CDMS' else 
                        ('JPL' if molecule_catalog_map.get(row['clean_standard_species'], {}).get('jpl', False) else 'CDMS')),
            axis=1
        )

        filtered_df['catalog_code'] = filtered_df.apply(
            lambda row: 1 if molecule_catalog_map.get(row['clean_standard_species'], {}).get('jpl', False) and not molecule_catalog_map.get(row['clean_standard_species'], {}).get('cdms', False) else
                        (2 if molecule_catalog_map.get(row['clean_standard_species'], {}).get('jpl', False) and molecule_catalog_map.get(row['clean_standard_species'], {}).get('cdms', False) else
                        (3 if not molecule_catalog_map.get(row['clean_standard_species'], {}).get('jpl', False) and molecule_catalog_map.get(row['clean_standard_species'], {}).get('cdms', False) else 0)),
            axis=1
        )

        with open(latex_output_file, 'w') as f:
    # Write LaTeX table header
            f.write("\\begin{table*}\n")
            f.write("\\centering\n")
            f.write("\\caption{ line parameters for RCW120 CORE2}\n")
            f.write("\\begin{tabular}{lccccccccl}\n")  # Added one more column for alignment
            f.write("\\hline\n")
            f.write("Молекула & Переход & $\\nu$, МГц & $E_u$ & $\\int T_{\rm mb}dV$ & $\\Delta V$ & $T_{\rm mb}$ & $v_{\rm LSR}$ & Reference & Код \\\\\n")
            f.write(" & & MHz & K & K km s$^{-1}$ & km s$^{-1}$ & K& km s$^{-1}$ & & \\\\\n")
            f.write("\\hline\n")

            for _, row in filtered_df.iterrows():
                vlsr = 299792.458 * ((rest_freq-row['frequency'])/rest_freq)
                print(vlsr)
                vlsr=vlsr-7
                row['frequency'] = rest_freq * (1 - vlsr/299792.458)
        # Calculate line width in MHz from velocity width
                if row['database_frequency'] != 0.0:
                    vlsr_mol = 299792.458 * (1-row['frequency']/row['database_frequency'])
                    
                else:
                    vlsr_mol = 0.0
                    row['database_frequency'] = row['frequency']
        
        # Format quantum numbers in LaTeX notation
                formatted_quantum_numbers = format_quantum_numbers(row['quantum_numbers'])
                formatted_quantum_numbers = f"{formatted_quantum_numbers}"
                width_error = row['width_error']
                area_error = row['area_error']
                '''
                if width_error > row['width'] / 2:
                    width_error = width_error / 6
    
                if area_error > row['area'] / 2:
                    area_error = width_error / 6
                '''
        # Format the line with proper LaTeX escaping and errors
                t_mb_mk = row['amplitude'] * 1000  # Convert to mK
                t_mb_error_mk = row['amplitude_error'] * 1000  # Convert error to mK
                        
        # Write the line with proper LaTeX escaping - now including catalog info and code
                f.write(f"{row['original_species'].replace('_', '\\_')} & "
                f"${formatted_quantum_numbers}$ & "
                f"{row['database_frequency']:.3f} & "
                f"{row['upper_level_energy']:.2f} & "
                f"{row['area']:.2f}$\\pm${area_error:.2f} & "
                f"{row['width']:.2f}$\\pm${width_error:.2f} & "
                f"{t_mb_mk:.0f}$\\pm${t_mb_error_mk:.0f} & "
                f"{vlsr_mol:.1f} & "
                f"{row.get('catalog_source', 'Unknown')} & "  # Add catalog source column
                f"{row.get('catalog_code', 0)} \\\\\n")  # Add catalog code column

    # Write LaTeX table footer
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\label{tab:molecular_parameters}\n")
            f.write("\\end{table*}\n")
    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.4f} seconds")